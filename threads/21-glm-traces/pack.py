"""Part C: pack GLM-format docs into window groups (orbit schema + boundary arrays + fit/val split).

Groups (each doc lives in one group; only boundary-free filler docs can be split across windows):
  c512          converted corpus docs without a real reasoning end (raw text/code + chat docs)
  c2048         converted corpus docs with >=1 real </think>
  c2048_traces  GLM-5.3 generated traces (pack.py --traces), gap fill from trace fillers + reserved raw pool

Items: docs with boundaries are "anchors" and never split. A doc longer than C is cut so that every chunk ends
right after a boundary and starts as early as possible, and boundary-free stretches become splittable fillers.
Anchors are packed first-fit-decreasing, then the gaps are filled exactly with filler tokens in stream order, so
there is no padding. Windows are then shuffled (seeded) within fit and within val, and val windows go last.
"""
import argparse, hashlib, json, os, random, sys
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

ROOT = os.environ.get("PACK_ROOT", "/tmp/nestquant/corpus/glm53_calib_glmfmt_v1")
CONV = "/tmp/nestquant/21-traces/conv_docs.jsonl"
DECONTAM = "/tmp/nestquant/21-traces/decontam.json"  # eval_overlap.py
SEED = 20260928
VAL_FRAC = 0.015
SEG_STRIDE = 1 << 20  # segment_id = doc_index + k*SEG_STRIDE for the k-th non-contiguous piece of a doc in a window
RESERVE_TOKENS = 200_000  # raw filler pool kept out of c512 for the traces group


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def chunk_doc(n, bnds, C):
    """-> list of (start, end, is_anchor)."""
    B = sorted(bnds); out = []; s = 0; j = 0
    while s < n:
        while j < len(B) and B[j] < s:
            j += 1
        if j == len(B):
            out.append((s, n, False)); break
        b1 = B[j]
        if b1 - s >= C:
            out.append((s, b1 - C + 1, False)); s = b1 - C + 1
        k = j
        while k + 1 < len(B) and B[k + 1] <= s + C - 1:
            k += 1
        out.append((s, B[k] + 1, True)); s = B[k] + 1; j = k + 1
    return out


def make_items(doc, C):
    ids = doc["ids"]; th, en = doc["th"], doc["en"]
    if not th and not en:
        return [], [(doc, 0, len(ids))]
    if len(ids) <= C:
        return [(doc, 0, len(ids))], []
    anchors, fillers = [], []
    for a, b, is_anchor in chunk_doc(len(ids), th + en, C):
        (anchors if is_anchor else fillers).append((doc, a, b))
    return anchors, fillers


class Deficit(Exception):
    pass


def pack(anchors, fillers, C, rng, extra_items=None):
    """anchors: non-splittable pieces; fillers: splittable pieces (consumed in order). Returns windows as lists of
    (doc, start, end), leftover filler tokens dropped count, and the unused filler pieces. With extra_items=k, extra
    full windows are only started while the stream is still inside the first k filler items (the rest of the
    stream is gap-fill only)."""
    anchors = sorted(anchors, key=lambda it: -(it[2] - it[1]))
    wins, free = [], []
    # first-fit decreasing with a size-bucketed free list (window free space is 0..C)
    buckets = [[] for _ in range(C + 1)]
    for it in anchors:
        L = it[2] - it[1]
        w = None
        for f in range(L, C + 1):
            if buckets[f]:
                w = buckets[f].pop(); break
        if w is None:
            w = len(wins); wins.append([]); free.append(C)
        wins[w].append(it); free[w] -= L
        buckets[free[w]].append(w)
    # gap fill + extra full windows from the filler stream
    fq = list(fillers); fi = 0; cur = None

    def take(space):
        nonlocal fi, cur
        got = []
        while space > 0:
            if cur is None:
                if fi == len(fq):
                    return got, space
                cur = list(fq[fi]); fi += 1
            d, a, b = cur
            n = min(space, b - a)
            got.append((d, a, a + n)); space -= n
            cur = None if a + n == b else [d, a + n, b]
        return got, 0

    n_anchor_wins = len(wins)
    for w in range(len(wins)):
        if free[w]:
            got, left = take(free[w]); wins[w].extend(got); free[w] = left
    while True:
        if extra_items is not None and (fi - 1 if cur is not None else fi) >= extra_items:
            break
        got, left = take(C)
        if not got:
            break
        wins.append(got); free.append(left)
    unused = ([tuple(cur)] if cur else []) + [tuple(x) for x in fq[fi:]]
    dropped = 0
    while len(wins) > n_anchor_wins and free[-1] > 0:  # at most the final partially-filled windows (orbit drops the partial tail)
        dropped += C - free[-1]; wins.pop(); free.pop()
    if any(f > 0 for f in free):
        raise Deficit(sum(free))
    return wins, dropped, unused


def write_group(name, fit_wins, val_wins, C, meta_extra, rng):
    out = f"{ROOT}/{name}"; os.makedirs(out, exist_ok=True)
    rng.shuffle(fit_wins); rng.shuffle(val_wins)
    wins = fit_wins + val_wins; W = len(wins)
    T = np.zeros((W, C), np.int32); S = np.zeros((W, C), np.int32)
    BT = np.zeros((W, C), np.int8); BE = np.zeros((W, C), np.int8)
    rows = []; cat_tok = {}; st = dict(think_boundaries=0, end_boundaries=0, think_rows=0, end_rows=0)
    stv = dict(st); docs_fit, docs_val = set(), set()
    for w, pieces in enumerate(wins):
        o = 0; segs = []; seen = {}; prev = None
        for d, a, b in pieces:
            n = b - a; di = d["doc_index"]
            # one segment per contiguous doc piece: a later non-contiguous piece of a doc already in this window
            # gets its own id di + k*SEG_STRIDE (so segment_id % SEG_STRIDE == doc_index)
            if prev is not None and prev[0] == di and prev[1] == a:
                sid = prev[2]
            else:
                k = seen.get(di, -1) + 1; seen[di] = k; sid = di + k * SEG_STRIDE
            prev = (di, b, sid)
            T[w, o:o + n] = d["ids"][a:b]; S[w, o:o + n] = sid
            th = [x - a for x in d["th"] if a <= x < b]; en = [x - a for x in d["en"] if a <= x < b]
            bt, be = G.bnd_arrays(n, th, en)
            BT[w, o:o + n] = bt; BE[w, o:o + n] = be
            s = st if w < len(fit_wins) else stv
            s["think_boundaries"] += len(th); s["end_boundaries"] += len(en)
            s["think_rows"] += int((bt > 0).sum()); s["end_rows"] += int((be > 0).sum())
            (docs_fit if w < len(fit_wins) else docs_val).add(d["doc_index"])
            segs.append(dict(source_id=d["source_id"], text_sha256=d["text_sha256"], category=d["category"],
                             kind=d["kind"], group=name, doc_index=di, segment_id=sid, token_offset=a, window_offset=o, tokens=n))
            cat_tok[d["category"]] = cat_tok.get(d["category"], 0) + n
            o += n
        assert o == C
        rows.append(dict(segments=segs))
    assert not (docs_fit & docs_val)
    np.save(f"{out}/tokens.npy", T); np.save(f"{out}/segments.npy", S)
    np.save(f"{out}/bnd_think_d.npy", BT); np.save(f"{out}/bnd_end_d.npy", BE)
    wb = "".join(json.dumps(r) + "\n" for r in rows).encode()
    open(f"{out}/windows.jsonl", "wb").write(wb)
    split = dict(fit=[0, len(fit_wins)], val=[len(fit_wins), W], fit_docs=len(docs_fit), val_docs=len(docs_val))
    json.dump(split, open(f"{out}/split.json", "w"), indent=1)
    man = dict(schema="nestquant-21-glmfmt-v1", group=name, role="calibration (fit) + held-out val windows",
               context=C, windows=W, tokens=W * C, fit_windows=len(fit_wins), val_windows=len(val_wins),
               documents=len(docs_fit | docs_val), fit_tokens=len(fit_wins) * C, val_tokens=len(val_wins) * C, boundary_counts_fit=st, boundary_counts_val=stv,
               category_tokens=cat_tok, seed=SEED,
               tokenizer_sha256=sha_file(f"{G.TOK_DIR}/tokenizer.json"),
               chat_template_sha256=sha_file(f"{G.TOK_DIR}/chat_template.jinja"),
               tokens_sha256=sha_file(f"{out}/tokens.npy"), segments_sha256=sha_file(f"{out}/segments.npy"),
               bnd_think_d_sha256=sha_file(f"{out}/bnd_think_d.npy"), bnd_end_d_sha256=sha_file(f"{out}/bnd_end_d.npy"),
               windows_sha256=hashlib.sha256(wb).hexdigest(), split_sha256=sha_file(f"{out}/split.json"),
               boundary_semantics="bnd_*_d[w,p] = b-p (1..32) for the next boundary token b in the same window "
               "segment, else 0; think = first </think> closing a non-empty turn-opening <think>; end = <|user|>/"
               "<|observation|>/<|endoftext|> right after an assistant turn; exclusive, nearer wins, ties -> end",
               packing="docs with boundaries never split (docs > C cut to end right after a boundary, max preceding "
               "context); first-fit-decreasing + exact gap fill with boundary-free filler pieces; no padding; "
               "attention is local to each window segment and positions reset per segment; segments.npy holds segment_id "
               "(one per contiguous doc piece; = doc_index, or doc_index + k*2^20 for the k-th further non-contiguous piece "
               "of the same doc in that window); windows.jsonl carries doc_index and segment_id per piece", **meta_extra)
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    print(name, json.dumps({k: v for k, v in man.items() if "sha" not in k and k not in ("packing", "boundary_semantics")}))
    return man


def load_docs(path, start_index=0):
    tok = G.tokenizer(); docs = []
    for i, l in enumerate(open(path)):
        d = json.loads(l)
        th, en = G.boundaries(d["ids"], chat=d["kind"] != "raw")
        text = tok.decode(d["ids"], skip_special_tokens=False)
        docs.append(dict(source_id=d["source_id"], category=d["category"], kind=d["kind"], ids=d["ids"], th=th, en=en,
                         doc_index=start_index + i, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return docs


def split_val(docs, rng, force=()):
    force = set(force)
    forced = {i for i, d in enumerate(docs) if d["source_id"] in force}
    rest = [i for i in range(len(docs)) if i not in forced]
    k = max(0, round(VAL_FRAC * len(docs)) - len(forced))
    val = forced | set(rng.sample(rest, k))
    return [d for i, d in enumerate(docs) if i not in val], [d for i, d in enumerate(docs) if i in val]


def build_group(docs, C, rng, extra_fillers=None):
    A, F = [], []
    for d in docs:
        a, f = make_items(d, C); A += a; F += f
    return A, F + list(extra_fillers or [])


def main_corpus():
    rng = random.Random(SEED)
    docs = load_docs(CONV)
    dc = json.load(open(DECONTAM)); excl = dc["exclude"]; force = dc["force_val"]
    docs = [d for d in docs if d["source_id"] not in excl]
    long_docs = [d for d in docs if d["th"]]
    short_docs = [d for d in docs if not d["th"]]
    rng.shuffle(short_docs)
    # reserve raw filler pool for the traces group (whole raw docs, not packed into c512)
    reserve, n = [], 0
    for d in short_docs:
        if d["kind"] == "raw" and not d["en"] and n < RESERVE_TOKENS and len(d["ids"]) < 4096:
            reserve.append(d); n += len(d["ids"])
    rset = {d["doc_index"] for d in reserve}
    short_docs = [d for d in short_docs if d["doc_index"] not in rset]
    lf, lv = split_val(long_docs, rng, force)
    sf, sv = split_val(short_docs, rng, force)
    # c2048: its own fillers, then raw docs borrowed (whole) from the c512 pool, taken from the end of the order
    raw_f = [d for d in sf if d["kind"] == "raw"]; raw_v = [d for d in sv if d["kind"] == "raw"]
    groups = {}
    for split, ldocs, raws in (("fit", lf, raw_f), ("val", lv, raw_v)):
        A, F = build_group(ldocs, 2048, rng)
        borrowed = []
        while True:  # borrow whole raw docs from the c512 pool (end of its order) until the FFD gaps are covered
            try:
                wins, dropped, unused = pack(A, F + [(d, 0, len(d["ids"])) for d in borrowed], 2048, rng)
                break
            except Deficit as e:
                need = e.args[0]
                while need > 0:
                    d = raws.pop(); borrowed.append(d); need -= len(d["ids"])
        # unused filler (the tail of the last borrowed doc) goes back: whole borrowed docs that were not touched
        used_idx = {p[0]["doc_index"] for w in wins for p in w}
        back = [d for d in borrowed if d["doc_index"] not in used_idx]
        raws.extend(back)
        groups[("c2048", split)] = (wins, dropped)
    bset = {p[0]["doc_index"] for s in ("fit", "val") for w in groups[("c2048", s)][0] for p in w}
    for split, sdocs in (("fit", sf), ("val", sv)):
        sdocs = [d for d in sdocs if d["doc_index"] not in bset]
        A, F = build_group(sdocs, 512, rng)
        wins, dropped, unused = pack(A, F, 512, rng)
        groups[("c512", split)] = (wins, dropped)
    # partial pieces of borrowed docs: any doc_index in both groups?
    c2 = bset
    c5 = {p[0]["doc_index"] for s in ("fit", "val") for w in groups[("c512", s)][0] for p in w}
    assert not (c2 & c5)
    mans = {}
    for g, C in (("c2048", 2048), ("c512", 512)):
        (fw, fd), (vw, vd) = groups[(g, "fit")], groups[(g, "val")]
        mans[g] = write_group(g, fw, vw, C, dict(dropped_tail_tokens=dict(fit=fd, val=vd),
                                                  source_corpus="orbit-duet runs/glm53_training_15m_v2 (converted)",
                                                  decontamination=dict(excluded_docs=len(excl), rule="docs in orbit windows >= 28784 (thread-18 nq-tail) dropped; >=5% 13-gram overlap with thread-18 eval texts or GPQA-diamond dropped; docs in orbit windows 28656..28783 (thread-19 val) forced into val")), rng)
    json.dump(dict(reserve_doc_indices=sorted(rset), reserve_tokens=n), open(f"{ROOT}/reserve_raw.json", "w"))
    return mans


TRACES = "/tmp/nestquant/21-traces/trace_docs.jsonl"   # ingest_traces.py
TRACE_DEDUP = "/tmp/nestquant/21-traces/trace_dedup.json"  # dedup_traces.py
TRACE_INDEX0 = 100_000  # trace doc_index = TRACE_INDEX0 + line (corpus doc_index < 40k; < SEG_STRIDE)
THINK_ONLY = {"claude-opus46"}  # sessions dominated by tool output: keep only the anchor pieces with a </think>
VAL_SOURCES = ("claude-opus46", "prime-glm53f", "dsv41flash")  # many small docs; tb21 / gpqa docs are few and huge


def main_traces():
    rng = random.Random(SEED + 1)
    C = 2048
    docs = load_docs(TRACES, start_index=TRACE_INDEX0)
    raw = [json.loads(l) for l in open(TRACES)]
    for d, r in zip(docs, raw):
        d.update(source=r["source"], model=r["model"], truncated=r["truncated"], conv_id=r["conv_id"],
                 gpqa_index=r.get("gpqa_index"), gpqa_record_id=r.get("gpqa_record_id"))
    dd = json.load(open(TRACE_DEDUP)); drop = dd["drop"]
    docs = [d for d in docs if d["source_id"] not in drop]
    # val: whole docs from the small-doc sources until ~VAL_FRAC of the trace tokens
    tot = sum(len(d["ids"]) for d in docs)
    cand = [d for d in docs if d["source"] in VAL_SOURCES and d["th"] and len(d["ids"]) <= 4 * C]
    rng.shuffle(cand); val, vt = [], 0
    for d in cand:
        if vt >= VAL_FRAC * tot:
            break
        val.append(d); vt += len(d["ids"])
    vset = {d["doc_index"] for d in val}
    fit = [d for d in docs if d["doc_index"] not in vset]
    # reserved raw corpus pool (not in c512/c2048): gap fill only
    rs = json.load(open(f"{ROOT}/reserve_raw.json"))["reserve_doc_indices"]
    corpus = load_docs(CONV); byi = {d["doc_index"]: d for d in corpus}
    reserve = [byi[i] for i in rs]; rng.shuffle(reserve)
    rv = reserve[:max(4, len(reserve) // 20)]; rf = reserve[len(rv):]  # disjoint raw pools for val / fit
    groups = {}; dropped_think_only = 0
    for split, ds, raws in (("fit", fit, rf), ("val", val, rv)):
        A, F = [], []
        order = list(ds); rng.shuffle(order)
        for d in order:
            a, f = make_items(d, C)
            if d["source"] in THINK_ONLY:
                keep = [it for it in a if any(it[1] <= x < it[2] for x in d["th"])]
                dropped_think_only += sum(it[2] - it[1] for it in a if it not in keep) + sum(it[2] - it[1] for it in f)
                a, f = keep, []
            A += a; F += f
        wins, dropped, unused = pack(A, F + [(d, 0, len(d["ids"])) for d in raws], C, rng, extra_items=len(F))
        groups[split] = (wins, dropped)
    used_raw = {p[0]["doc_index"] for s in groups.values() for w in s[0] for p in w if p[0]["doc_index"] < TRACE_INDEX0}
    assert used_raw <= set(rs)
    # per-source mix over the packed windows
    mix = {}
    for split, (wins, _) in groups.items():
        for w in wins:
            for d, a, b in w:
                k = d.get("source", "reserve-raw:" + d["category"])
                m = mix.setdefault(k, dict(fit_tokens=0, val_tokens=0, think_boundaries=0, end_boundaries=0, docs=set(), truncated_docs=set()))
                m[f"{split}_tokens"] += b - a
                m["think_boundaries"] += sum(a <= x < b for x in d["th"]); m["end_boundaries"] += sum(a <= x < b for x in d["en"])
                m["docs"].add(d["doc_index"])
                if d.get("truncated"):
                    m["truncated_docs"].add(d["doc_index"])
    for m in mix.values():
        m["docs"] = len(m["docs"]); m["truncated_docs"] = len(m["truncated_docs"])
    gq = sorted({(d["gpqa_index"], d["gpqa_record_id"]) for d in docs if d.get("gpqa_index") is not None})
    (fw, fd), (vw, vd) = groups["fit"], groups["val"]
    meta = dict(dropped_tail_tokens=dict(fit=fd, val=vd), source_corpus="on-box reasoning traces (ingest_traces.py) + reserved raw orbit docs as gap fill",
                source_mix=mix, think_only_dropped_tokens=dropped_think_only,
                trace_sources_excluded="Qwen (ctap2 round013, ICH eval gens) and gpt-oss-120b (rad-agent batches) excluded by user override 2026-09-28",
                gpqa_included=dict(note="GLM-5.2 hybrid GPQA-diamond traces included by user override; later GPQA evals should exclude these questions",
                                   question_index=[i for i, _ in gq], record_id=[r for _, r in gq]),
                truncated_semantics="truncated/runaway traces are included without an end token (and without </think> when cut inside the reasoning)",
                dedup=dict(rule="13-gram: >=5% overlap with GPQA/tb4/ICH/thread-18 evals dropped (GPQA traces vs GPQA and TB2.1 vs tb4 exempt by user/lead decision), >=50% overlap with the converted corpus dropped",
                           dropped=len(drop), kept_overlaps=dd.get("kept_overlaps")))
    return write_group("c2048_traces", fw, vw, C, meta, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--traces", action="store_true")
    a = ap.parse_args()
    if a.traces:
        main_traces()
    else:
        main_corpus()
