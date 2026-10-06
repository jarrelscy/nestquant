#!/usr/bin/env python3
"""T37 jF step 1: capture trace (capture37g.py --trace) -> chains + per-layer 16-token block matrices.  PRIVATE.

  blocks37.py chains            -> $OUT/chains/{train,val,test}.npz + chains.json  (layer independent)
  blocks37.py blk [NPROC]       -> $OUT/blk/{split}/L{L}.npz + meta.json            (per layer, resumable)

Trace layout (capture37g.py): per rank r of W, windows.r{r}of{W}.json = {"wins": [[group, window, split], ...], "N"}
(this rank's windows, round-robin, 2048 rows each, no padding) and L{L}.r{r}of{W}.npz = ids uint16 [N, 8],
w f16 [N, 8] (final combine weights INCLUDING routed_scaling_factor 2.5), xn f32 [N] (sum x^2 of the normalised
MoE input).  Rows are window-major in the rank's window order; attention is local to each window segment (glmfmt
packing), so every segment = one contiguous piece of one source document.

Chains (predictor state resets per chain; T32/T33 used 4 x 2048 = 8192-token chains):
  * the corpus windows are shuffled excerpts, so instead of "consecutive windows" each chain is rebuilt from the
    document pieces in source order (windows.jsonl source_id + token_offset): the capture's reasoning / agent traces
    (group traces: 160 long docs, 978 of 1049 adjacent pieces are contiguous across windows) give one chain stream
    per document, cut into J37_CHAIN-token chains;  c2048 docs (short, ~1k tokens) are concatenated in corpus order
    into one stream and cut the same way (multi-document chains, like the GLM-5.3 calib-fit chains).
  * piece starts are cold-context (segment-local attention in the capture), as in the GLM-5.3 calib traces.
  * chain lengths are truncated to a multiple of 16; a tail shorter than J37_MINTAIL tokens joins the previous chain.
Splits (by DOCUMENT, no doc in two splits):
  test  = the corpus' own held-out val windows (doc-disjoint by construction; = the capture's val split)
  val   = fit docs whose sha256(source_id) ranks first until J37_VALFRAC of the group's fit tokens (model selection)
  train = the remaining fit docs.

DECODE traces (the jF training source; selected automatically when $J37_TRACE has seqs.r{r}of{W}.json):
  per rank: L{L}.r{r}of{W}.npz (ids uint16 [N, 8], w f16 [N, 8] incl. 2.5 scaling, xn f32 [N]; same as capture37g)
  + tok.r{r}of{W}.npy int32 [N] (token id of every trace row) + seqs.r{r}of{W}.json =
  {"N": N, "seqs": [{"id": str, "rows": n, "prompt_len": p, "group": optional (split key, e.g. the prompt / task
  id so that samples of one prompt never straddle splits), "split": optional "train"|"val"|"test"}, ...]} with the
  sequences' rows concatenated in order (row = position in prompt + decode; decode rows = positions >= prompt_len).
  One chain per sequence = its decode part (serve semantics: fresh predictor per request, prefill chunks ignored),
  optionally led by J37_DEC_PROMPT prompt-tail tokens; J37_DEC_MAXCHAIN > 0 cuts long decodes (default 0: no cut).
  Splits without an explicit "split": by sha256(group) -> test J37_TESTFRAC (0.10) / val J37_VALFRAC / train.
HANDOFF eval pairs (J37_HANDOFF=1, val / test only, eval only): per decode sequence with a prompt, kind 2 = the last
J37_HO_PROMPT (4096) prompt tokens + the first J37_HO_DEC (16) decode blocks (the seeded predictor: step_chunk over the
prompt tail, one refresh at hm 0), kind 3 = the same decode blocks alone (cold: fresh predictor, floating_default).
They re-use rows of the decode / prompt chains and are excluded from training and the row-uniqueness check."""
import glob
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import jlib37 as J  # noqa: E402

CHAIN = int(os.environ.get("J37_CHAIN", "8192"))
MINTAIL = int(os.environ.get("J37_MINTAIL", "1024"))
VALFRAC = float(os.environ.get("J37_VALFRAC", "0.08"))
TESTFRAC = float(os.environ.get("J37_TESTFRAC", "0.10"))
DEC_PROMPT = int(os.environ.get("J37_DEC_PROMPT", "0"))
DEC_MAXCHAIN = int(os.environ.get("J37_DEC_MAXCHAIN", "0"))
BREAK_ON_DOC = {"traces": True, "c2048": False, "mm": False}


def rank_files(trace, kind=None):
    kind = kind or ("seqs" if is_decode(trace) else "windows")
    fs = []
    for f in glob.glob(f"{trace}/{kind}.r*of*.json"):
        r, W = map(int, re.search(kind + r"\.r(\d+)of(\d+)\.json$", f).groups())
        fs.append((r, W, f))
    fs.sort()
    assert fs and [r for r, _, _ in fs] == list(range(fs[0][1])), (trace, fs)
    return fs


def is_decode(trace):
    return bool(glob.glob(f"{trace}/seqs.r*of*.json"))


def is_tf(trace):
    return bool(glob.glob(f"{trace}/windows.r*of*.json"))


def pieces(trace):
    """all captured document pieces with their trace row (rank-concatenated order) of a capture37g trace."""
    cache, out, off = {}, [], 0
    for r, W, f in rank_files(trace, "windows"):
        j = json.load(open(f))
        assert j["N"] == len(j["wins"]) * J.SEQ, (f, j["N"], len(j["wins"]))
        for k, (g, w, split) in enumerate(j["wins"]):
            if g not in cache:
                root = J.GROUP_ROOT[g]
                cache[g] = dict(win=[json.loads(l) for l in open(f"{root}/windows.jsonl")],
                                tok=np.load(f"{root}/tokens.npy", mmap_mode="r"))
            segs = cache[g]["win"][w]["segments"]
            assert sum(s["tokens"] for s in segs) == J.SEQ, (g, w)
            for s in segs:
                out.append(dict(g=g, w=int(w), split=split, src=s["source_id"], off=int(s["token_offset"]),
                                n=int(s["tokens"]), wo=int(s["window_offset"]), row0=off + k * J.SEQ + int(s["window_offset"]),
                                cat=s.get("category", g)))
        off += j["N"]
    return out, cache, off


def assign_splits(P):
    for p in P:
        p["sp"] = "test" if p["split"] == "val" else None
    for g in sorted({p["g"] for p in P}):
        fit = [p for p in P if p["g"] == g and p["sp"] is None]
        tok = {}
        for p in fit:
            tok[p["src"]] = tok.get(p["src"], 0) + p["n"]
        order = sorted(tok, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        tot, acc, val = sum(tok.values()), 0, set()
        for s in order:
            if len(tok) < 2 or acc >= VALFRAC * tot:
                break
            val.add(s); acc += tok[s]
        for p in fit:
            p["sp"] = "val" if p["src"] in val else "train"


def cut(rows, tok, info, chain=None):
    """one stream (lists of row / token arrays) -> chains of <= chain tokens (tail < MINTAIL merged)."""
    chain = chain or CHAIN
    R = np.concatenate(rows); T = np.concatenate(tok)
    n = len(R)
    b = list(range(0, n, chain)) + [n]
    if len(b) > 2 and b[-1] - b[-2] < MINTAIL:
        b.pop(-2)
    ch = []
    for s, e in zip(b[:-1], b[1:]):
        e = s + (e - s) // J.G * J.G
        if e - s >= 5 * J.G:                    # y64 needs 4 future blocks
            ch.append((R[s:e], T[s:e], dict(info, ntok=int(e - s))))
    return ch


def seg_of(tok):
    """per position t: 0 think / 1 answer = serve state after the emitted token tok[t+1]; chain start = think."""
    nt = np.r_[tok[1:], -1]
    ev = np.where(nt == J.THINK_ID, 0, np.where(nt == J.ETHINK_ID, 1, -1))
    idx = np.where(ev >= 0, np.arange(len(tok)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, ev[np.maximum(idx, 0)], 0).astype(np.int8)


def dec_chains(traces):
    """decode traces (src index = position in `traces`) -> {split: [(rows, tok, info)]}.  Kind decode = rows >=
    prompt_len (optionally led by DEC_PROMPT prompt-tail rows); with J37_DEC_PREFILL=1 the remaining prompt rows of
    each sequence become a separate kind-prefill chain (same group -> same split).  Splits by sha256(group) weighted
    by decode tokens, over ALL decode traces jointly (one prompt / task id in two traces lands in one split)."""
    S = []
    for si, tr in enumerate(traces):
        off = 0
        for r, W, f in rank_files(tr, "seqs"):
            j = json.load(open(f))
            tk = np.load(f"{tr}/tok.r{r}of{W}.npy", mmap_mode="r")
            assert len(tk) == j["N"] == sum(int(q["rows"]) for q in j["seqs"]), (f, j["N"], len(tk))
            o = 0
            for q in j["seqs"]:
                n, p = int(q["rows"]), int(q.get("prompt_len", 0))
                assert 0 <= p <= n, q
                S.append(dict(id=str(q["id"]), key=str(q.get("group", q["id"])), split=q.get("split"), n=n, p=p,
                              tok=tk, o=o, row0=off + o, src=si))
                o += n
            off += j["N"]
    dtok = {}
    for q in S:
        dtok[q["key"]] = dtok.get(q["key"], 0) + q["n"] - q["p"]
    order = sorted(dtok, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    tot, acc, gsp = sum(dtok.values()), 0, {}
    for k in order:
        gsp[k] = "test" if acc < TESTFRAC * tot else "val" if acc < (TESTFRAC + VALFRAC) * tot else "train"
        acc += dtok[k]
    for q in S:
        q["sp"] = {"fit": None}.get(q["split"], q["split"]) or gsp[q["key"]]
        assert q["sp"] in J.SPLITS, q
    grp = {}
    for q in S:
        grp.setdefault(q["key"], set()).add(q["sp"])
    assert all(len(v) == 1 for v in grp.values()), "a group straddles splits (explicit split fields disagree)"
    out = {sp: [] for sp in J.SPLITS}
    big = DEC_MAXCHAIN or (1 << 62)
    for q in S:
        a = max(0, q["p"] - DEC_PROMPT)
        rows = np.arange(q["row0"] + a, q["row0"] + q["n"], dtype=np.int64)
        tk = np.asarray(q["tok"][q["o"] + a:q["o"] + q["n"]], np.int32)
        base = dict(src=q["src"], group=q["key"], seq=q["id"], docs=1, pieces=1)
        if len(rows) >= 5 * J.G:
            out[q["sp"]] += cut([rows], [tk], dict(base, kind=0, cat="decode", prompt_lead=int(q["p"] - a)), big)
        if J.DEC_PREFILL and a >= 5 * J.G:                     # prompt rows not used as the decode lead
            rows = np.arange(q["row0"], q["row0"] + a, dtype=np.int64)
            tk = np.asarray(q["tok"][q["o"]:q["o"] + a], np.int32)
            out[q["sp"]] += cut([rows], [tk], dict(base, kind=1, cat="dec-prompt"))
        if J.HANDOFF and q["sp"] != "train":                 # handoff eval pair (rows shared with the chains above)
            tail = min(q["p"], J.HO_PROMPT) // J.G * J.G
            dd = min(q["n"] - q["p"], J.HO_DEC * J.G) // J.G * J.G
            if tail >= J.G and dd >= J.G:
                hid = id_next()
                r0 = q["row0"] + q["p"]
                tka = np.asarray(q["tok"][q["o"] + q["p"] - tail:q["o"] + q["p"] + dd], np.int32)
                out[q["sp"]].append((np.arange(r0 - tail, r0 + dd, dtype=np.int64), tka,
                                     dict(base, kind=2, cat="handoff", hoff=tail // J.G, hid=hid, ntok=tail + dd)))
                out[q["sp"]].append((np.arange(r0, r0 + dd, dtype=np.int64), tka[tail:],
                                     dict(base, kind=3, cat="handoff-cold", hoff=0, hid=hid, ntok=dd)))
    return out, {sp: len({q["key"] for q in S if q["sp"] == sp}) for sp in J.SPLITS}


_HID = [0]


def id_next():
    _HID[0] += 1
    return _HID[0] - 1


def tf_chains(trace, si, kind):
    """teacher-forced capture37g trace -> {split: [(rows, tok, info)]} (document chains, splits by document)."""
    P, cache, N = pieces(trace)
    assign_splits(P)
    out = {}
    for sp in J.SPLITS:
        chains = []
        for g in sorted({p["g"] for p in P}):
            ps = [p for p in P if p["g"] == g and p["sp"] == sp]
            first = {}
            for p in ps:
                first.setdefault(p["src"], (p["w"], p["wo"]))
                first[p["src"]] = min(first[p["src"]], (p["w"], p["wo"]))
            docs = sorted(first, key=lambda s: first[s])
            by = {s: sorted([p for p in ps if p["src"] == s], key=lambda p: (p["off"], p["w"])) for s in docs}
            tk = lambda p: np.asarray(cache[g]["tok"][p["w"], p["wo"]:p["wo"] + p["n"]], np.int32)  # noqa: E731
            rw = lambda p: np.arange(p["row0"], p["row0"] + p["n"], dtype=np.int64)              # noqa: E731
            base = dict(src=si, kind=kind, group=g)
            if BREAK_ON_DOC[g]:
                for s in docs:
                    chains += cut([rw(p) for p in by[s]], [tk(p) for p in by[s]],
                                  dict(base, docs=1, cat=by[s][0]["cat"], pieces=len(by[s])))
            elif docs:
                al = [p for s in docs for p in by[s]]
                chains += cut([rw(p) for p in al], [tk(p) for p in al], dict(base, docs=len(docs), cat="mixed",
                                                                             pieces=len(al)))
        out[sp] = chains
    return out, {sp: len({p["src"] for p in P if p["sp"] == sp}) for sp in J.SPLITS}


def train_blocks(chains, sub):
    """train blocks a SUB-strided sampler would keep (finite y64 target: not the last 4 blocks of a chain)."""
    n = 0
    for c in chains:
        nb = len(c[0]) // J.G
        n += len(range(0, max(nb - 4, 0), sub))
    return n


def build_chains():
    """sources: J37_TRACE (':'-separated DECODE traces, required) + optional J37_PREFILL_TRACE (teacher-forced
    capture trace, kind prefill) + optional decode prompt rows (J37_DEC_PREFILL=1, kind prefill)."""
    tf_alone = os.environ.get("J37_ALLOW_TF", "0") == "1"
    src, per = [], []                                    # src: [{trace, type}]; per: (chains by split, groups by split)
    dec = [t for t in J.TRACES if is_decode(t)]
    bad = [t for t in J.TRACES if not is_decode(t)]
    if bad:
        if tf_alone and len(J.TRACES) == 1 and not J.PF_TRACE and is_tf(bad[0]):
            print(f"[blocks37] SMOKE: J37_ALLOW_TF=1, teacher-forced trace {bad[0]} used ALONE as kind decode", flush=True)
            src.append(dict(trace=bad[0], type="tf-as-decode-SMOKE"))
            per.append(tf_chains(bad[0], 0, 0))
        else:
            sys.exit(f"[blocks37] not decode traces (no seqs.r*of*.json): {bad}.  J37_TRACE must hold fp8 / NVFP4 DECODE "
                     "traces (dec37 / online-corpus TF with seqs+tok metadata).  A teacher-forced capture trace goes in "
                     "J37_PREFILL_TRACE (minority kind prefill, mixed with decode data); alone only for a smoke test "
                     "(J37_ALLOW_TF=1, no J37_PREFILL_TRACE).")
    else:
        for t in dec:
            src.append(dict(trace=t, type="decode"))
        per.append(dec_chains(dec))
    if J.PF_TRACE:
        if not is_tf(J.PF_TRACE):
            sys.exit(f"[blocks37] J37_PREFILL_TRACE {J.PF_TRACE}: no windows.r*of*.json (capture37g trace expected)")
        src.append(dict(trace=J.PF_TRACE, type="tf-prefill"))
        per.append(tf_chains(J.PF_TRACE, len(src) - 1, 1))
    allc = {sp: [c for ch, _ in per for c in ch[sp]] for sp in J.SPLITS}
    ndec = {sp: sum(c[2]["kind"] == 0 for c in allc[sp]) for sp in J.SPLITS}
    if not ndec["train"] or not ndec["val"]:
        sys.exit(f"[blocks37] refusing: no decode chains in train / val ({ndec}); jF needs decode data present "
                 "(the prefill / teacher-forced source is a minority mix-in only)")
    # train mix: prefill share of the strided train rows -> deterministic block subsample probability q
    sub = int(os.environ.get("J37_SUB", "8"))
    nd = train_blocks([c for c in allc["train"] if c[2]["kind"] == 0], sub)
    npf = train_blocks([c for c in allc["train"] if c[2]["kind"] == 1], sub)
    f = J.PF_FRAC
    if f > J.PF_MAXFRAC:
        sys.exit(f"[blocks37] J37_PREFILL_FRAC {f} > J37_PREFILL_MAXFRAC {J.PF_MAXFRAC}")
    q = min(1.0, f / (1 - f) * nd / npf) if npf and 0 < f < 1 else (1.0 if npf and f >= 1 else 0.0)
    ach = q * npf / max(nd + q * npf, 1)
    mix = dict(prefill_frac_target=f, q=q, decode_train_rows=nd, prefill_train_rows_avail=npf,
               prefill_train_rows=int(round(q * npf)), prefill_frac_achieved=ach, sub=sub)
    print(f"[blocks37] train mix: decode {nd} rows, prefill {npf} avail -> keep q {q:.4f} = {q * npf:.0f} rows "
          f"({100 * ach:.1f}% prefill, target {100 * f:.1f}%){'  (short of prefill data)' if ach + 1e-6 < f else ''}",
          flush=True)
    os.makedirs(f"{J.OUT}/chains", exist_ok=True)
    summ = {}
    for sp in J.SPLITS:
        chains = allc[sp]
        rows = np.concatenate([c[0] for c in chains]) if chains else np.zeros(0, np.int64)
        tok = np.concatenate([c[1] for c in chains]) if chains else np.zeros(0, np.int32)
        cst = np.r_[0, np.cumsum([len(c[0]) for c in chains])].astype(np.int64)
        seg = np.concatenate([seg_of(c[1]) for c in chains]) if chains else np.zeros(0, np.int8)
        csrc = np.array([c[2]["src"] for c in chains], np.int8)
        ckind = np.array([c[2]["kind"] for c in chains], np.int8)
        choff = np.array([c[2].get("hoff", 0) for c in chains], np.int32)
        chid = np.array([c[2].get("hid", -1) for c in chains], np.int64)
        np.savez(f"{J.OUT}/chains/{sp}.npz", rows=rows, tok=tok, cstart=cst, seg=seg, csrc=csrc, ckind=ckind,
                 choff=choff, chid=chid)
        info = [c[2] for c in chains]
        by_kind = {}
        for kd in range(len(J.KINDS)):
            ii = [i for i in info if i["kind"] == kd]
            by_kind[J.KINDS[kd]] = dict(chains=len(ii), tokens=sum(i["ntok"] for i in ii))
        summ[sp] = dict(chains=len(chains), tokens=int(len(rows)), blocks=int(len(rows) // J.G),
                        docs=sum(g[sp] for _, g in per), by_kind=by_kind,
                        by_source={s["type"] + f"#{k}": sum(i["ntok"] for i in info if i["src"] == k)
                                   for k, s in enumerate(src)},
                        answer_frac=float(seg.mean()) if len(seg) else 0.0, chain_info=info)
        print(f"[blocks37] {sp}: {len(chains)} chains {len(rows)} tokens groups {summ[sp]['docs']} by kind {by_kind} "
              f"answer-frac {summ[sp]['answer_frac']:.3f}", flush=True)
    for k in range(len(src)):                            # no trace row in two chains / splits
        used = []
        for sp in J.SPLITS:
            c = np.load(f"{J.OUT}/chains/{sp}.npz")
            rs = np.repeat(c["csrc"], np.diff(c["cstart"]))
            rk = np.repeat(c["ckind"], np.diff(c["cstart"]))
            used.append(c["rows"][(rs == k) & (rk < 2)])      # handoff eval pairs re-use decode / prompt rows
        used = np.concatenate(used)
        assert len(np.unique(used)) == len(used), f"row used twice in source {k}"
    json.dump(dict(sources=src, trace=J.TRACE, prefill_trace=J.PF_TRACE, dec_prefill=J.DEC_PREFILL,
                   dec_prompt=DEC_PROMPT, dec_maxchain=DEC_MAXCHAIN, chain=CHAIN, mintail=MINTAIL, testfrac=TESTFRAC,
                   valfrac=VALFRAC, mix=mix, splits=summ), open(f"{J.OUT}/chains.json", "w"), indent=1)


def load_trace(L, trace):
    parts = [np.load(f"{trace}/L{L}.r{r}of{W}.npz") for r, W, _ in rank_files(trace)]
    return tuple(np.concatenate([p[k] for p in parts]) for k in ("ids", "w", "xn"))


def block_mats(ids, w, xn, seg):
    """(= t32lib.block_mats, NE 288) per 16-token block: cnt, cnt_answer, n_answer, sal (raw w^2 xn), seg_last."""
    T = ids.shape[0]
    nb = T // J.G
    b = (np.arange(T) // J.G)[:, None].repeat(J.TOPK, 1)
    idx = (b * J.NE + ids.astype(np.int64)).ravel()
    cnt = np.bincount(idx, minlength=nb * J.NE).reshape(nb, J.NE).astype(np.float32)
    a = np.repeat(seg[:, None], J.TOPK, 1).ravel().astype(bool)
    cnta = np.bincount(idx[a], minlength=nb * J.NE).reshape(nb, J.NE).astype(np.float32)
    v = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]).ravel()
    sal = np.bincount(idx, weights=v, minlength=nb * J.NE).reshape(nb, J.NE)
    nans = seg.reshape(nb, J.G).sum(1)
    seg_last = seg.reshape(nb, J.G)[:, -1]
    return cnt, cnta, nans, sal, seg_last


def blk_job(L):
    if all(os.path.exists(f"{J.OUT}/blk/{sp}/L{L}.npz") for sp in J.SPLITS):
        return L, 0.0
    t0 = time.time()
    src = json.load(open(f"{J.OUT}/chains.json"))["sources"]
    tr = {}
    for k, s in enumerate(src):
        tr[k] = load_trace(L, s["trace"])
        assert tr[k][0].max() < J.NE and tr[k][0].shape[1] == J.TOPK, s
    for sp in J.SPLITS:
        c = np.load(f"{J.OUT}/chains/{sp}.npz")
        r = c["rows"]
        rs = np.repeat(c["csrc"], np.diff(c["cstart"]))
        ids = np.empty((len(r), J.TOPK), np.uint16); w = np.empty((len(r), J.TOPK), np.float16)
        xn = np.empty(len(r), np.float32)
        for k in tr:
            m = rs == k
            ids[m], w[m], xn[m] = tr[k][0][r[m]], tr[k][1][r[m]], tr[k][2][r[m]]
        cnt, cnta, nans, sal, segl = block_mats(ids, w, xn, c["seg"])
        f = f"{J.OUT}/blk/{sp}/L{L}.npz"
        np.savez(f + ".part.npz", bcnt=cnt.astype(np.uint8), bcnta=cnta.astype(np.uint8), nans=nans.astype(np.uint8),
                 segl=segl.astype(np.uint8), bsal=sal.astype(np.float32), slot_sal_sum=np.float64(sal.sum()),
                 slots=np.int64(len(r) * J.TOPK))
        os.replace(f + ".part.npz", f)
    return L, time.time() - t0


def build_blk(nproc):
    for sp in J.SPLITS:
        os.makedirs(f"{J.OUT}/blk/{sp}", exist_ok=True)
        c = np.load(f"{J.OUT}/chains/{sp}.npz")
        cst = c["cstart"]
        assert (cst % J.G == 0).all()
        json.dump(dict(bstart=(cst // J.G).tolist(), kind=c["ckind"].tolist(), src=c["csrc"].tolist(),
                       hoff=c["choff"].tolist(), hid=c["chid"].tolist()),
                  open(f"{J.OUT}/blk/{sp}/meta.json", "w"))
    with Pool(nproc) as p:
        for L, t in p.imap_unordered(blk_job, J.LAYERS):
            print(f"L{L} {t:.0f}s", end=" | ", flush=True)
    print(flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "chains":
        build_chains()
    else:
        build_blk(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
