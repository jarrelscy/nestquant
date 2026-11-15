"""Pack converted Inkling documents (work/<group>_ink.jsonl) into 2048-token windows, same schema as the GLM input.

- A document's split = the split of the source windows it came from (reassemble.py asserts no doc crosses splits).
- Each document is one or more fragments (one per contiguous source run); a fragment is cut into pieces of <= C
  tokens. Cuts are moved earlier (at most C/2) so that no boundary's 32-token approach window straddles a cut.
- Full-size pieces fill their own window; shorter pieces are first-fit-decreasing packed; leftover positions are
  padding (token PAD, segment -1). Windows within a split are shuffled deterministically; fit windows first.
- segment_id = doc_index, or doc_index + k*2^20 for the k-th further piece of the same doc in a window.
- bnd arrays per window segment: inkfmt.bnd_arrays on the piece with only the boundaries inside the piece.
"""
import json, sys, os, hashlib, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkfmt as IF

C = 2048
SEED = 20260928
WORK = "/tmp/nq_inkling/corpus/work"
OUTROOT = "/tmp/nq_inkling/corpus"
SRC = "/home/coder/nestquant-corpus-backup/glm53_calib_glmfmt_v1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cuts(n, bnds):
    """Piece boundaries [0, ..., n] for a fragment of n tokens."""
    out, s = [0], 0
    bnds = sorted(bnds)
    while n - s > C:
        c = s + C
        moved = True
        while moved:  # boundary b's rows are b-NB..b: never let a cut fall inside them
            moved = False
            for b in bnds:
                if b - IF.NB < c <= b and (b - IF.NB) - s >= C // 2:
                    c = b - IF.NB; moved = True; break
        out.append(c); s = c
    out.append(n)
    return out


def main(group):
    docs = [json.loads(l) for l in open(f"{WORK}/{group}_ink.jsonl")]
    src_man = json.load(open(f"{SRC}/{group}/manifest.json"))
    pieces = {"fit": [], "val": []}
    st = collections.Counter(); cat_tok = collections.Counter(); split_cat = collections.Counter()
    for d in docs:
        off = 0
        for fi, fr in enumerate(d["fragments"]):
            ids = fr["ids"]; n = len(ids)
            cs = cuts(n, fr["think"] + fr["end"])
            st["split_pieces"] += len(cs) - 2
            for a, b in zip(cs[:-1], cs[1:]):
                pieces[d["split"]].append(dict(doc=d, frag=fr, fi=fi, a=a, b=b, doc_off=off + a))
            off += n
            cat_tok[d["category"]] += n; split_cat[f"{d['split']}/{d['category']}"] += n
            st[f"{d['split']}_{fr['mode']}_fragments"] += 1
        st[f"{d['split']}_docs"] += 1
    rng = random.Random(SEED)
    wins = {}
    for sp in ("fit", "val"):
        P = pieces[sp]
        full = [[p] for p in P if p["b"] - p["a"] == C]
        rest = sorted([p for p in P if p["b"] - p["a"] < C], key=lambda p: -(p["b"] - p["a"]))
        bins, free = [], []
        for p in rest:  # first-fit decreasing
            L = p["b"] - p["a"]
            for i, f in enumerate(free):
                if f >= L:
                    bins[i].append(p); free[i] -= L; break
            else:
                bins.append([p]); free.append(C - L)
        w = full + bins
        rng.shuffle(w)
        wins[sp] = w
    W = len(wins["fit"]) + len(wins["val"])
    tokens = np.full((W, C), IF.PAD, np.int32); segs = np.full((W, C), -1, np.int32)
    bth = np.zeros((W, C), np.int8); ben = np.zeros((W, C), np.int8)
    out = f"{OUTROOT}/{group}"; os.makedirs(out, exist_ok=True)
    bc = {sp: collections.Counter() for sp in ("fit", "val")}
    tok_split = collections.Counter()
    with open(f"{out}/windows.jsonl", "w") as wf:
        wi = 0
        for sp in ("fit", "val"):
            for win in wins[sp]:
                o = 0; seen = collections.Counter(); segl = []
                for p in win:
                    d, fr, a, b = p["doc"], p["frag"], p["a"], p["b"]; L = b - a
                    k = seen[d["source_id"]]; seen[d["source_id"]] += 1
                    sid = d["doc_index"] + k * (1 << 20)
                    tokens[wi, o:o + L] = fr["ids"][a:b]; segs[wi, o:o + L] = sid
                    th = [x - a for x in fr["think"] if a <= x < b]; en = [x - a for x in fr["end"] if a <= x < b]
                    dt, de = IF.bnd_arrays(L, th, en)
                    bth[wi, o:o + L] = dt; ben[wi, o:o + L] = de
                    bc[sp]["think_boundaries"] += len(th); bc[sp]["end_boundaries"] += len(en)
                    bc[sp]["think_rows"] += int((dt > 0).sum()); bc[sp]["end_rows"] += int((de > 0).sum())
                    seg = dict(source_id=d["source_id"], text_sha256=fr["text_sha256"], category=d["category"], kind=d["kind"],
                               group=group, doc_index=d["doc_index"], segment_id=sid, token_offset=p["doc_off"],
                               window_offset=o, tokens=L, fragment=p["fi"], src_token_offset=fr["src_token_offset"])
                    if fr["mode"] == "fallback":
                        seg["fallback"] = True
                    segl.append(seg); o += L
                tok_split[sp] += o; tok_split[sp + "_pad"] += C - o
                wf.write(json.dumps(dict(segments=segl)) + "\n"); wi += 1
    nf = len(wins["fit"])
    np.save(f"{out}/tokens.npy", tokens); np.save(f"{out}/segments.npy", segs)
    np.save(f"{out}/bnd_think_d.npy", bth); np.save(f"{out}/bnd_end_d.npy", ben)
    fit_docs = sum(1 for d in docs if d["split"] == "fit"); val_docs = len(docs) - fit_docs
    json.dump({"fit": [0, nf], "val": [nf, W], "fit_docs": fit_docs, "val_docs": val_docs}, open(f"{out}/split.json", "w"), indent=1)
    cv = json.load(open(f"{WORK}/{group}_convstats.json"))
    nchat = cv.get("frag_chat", 0)
    man = dict(
        schema="nestquant-inkling2-fmt-v1", group=group, model="thinkingmachines/Inkling-2-Small",
        role="calibration (fit) + held-out val windows", context=C, windows=W, fit_windows=nf, val_windows=W - nf,
        documents=len(docs), fit_docs=fit_docs, val_docs=val_docs,
        tokens=tok_split["fit"] + tok_split["val"], fit_tokens=tok_split["fit"], val_tokens=tok_split["val"],
        pad_tokens=dict(fit=tok_split["fit_pad"], val=tok_split["val_pad"]),
        boundary_counts_fit=dict(bc["fit"]), boundary_counts_val=dict(bc["val"]),
        category_tokens=dict(cat_tok), split_category_tokens=dict(split_cat),
        fragments={k: v for k, v in st.items() if k.endswith("_fragments")},
        docs_split_across_windows_pieces=st["split_pieces"],
        conversion=dict(
            chat_fragments=nchat, raw_fragments=cv.get("frag_raw", 0), fallback_fragments=cv.get("frag_fallback", 0),
            fallback_rate=cv.get("frag_fallback", 0) / max(1, nchat + cv.get("frag_fallback", 0)),
            glm_roundtrip_pass=cv.get("roundtrip_pass", 0), glm_roundtrip_rate=cv.get("roundtrip_pass", 0) / max(1, nchat),
            head_partial_fragments=cv.get("head_partial", 0),
            tail_cut_fragments={k[5:]: v for k, v in cv.items() if k.startswith("tail_")},
            dropped_partial_tool_calls=cv.get("dropped_partial_call", 0) + cv.get("head_dropped_partial_call", 0),
            fragments_with_tool_declarations=cv.get("frag_with_tools", 0), tool_calls=cv.get("tool_calls", 0),
            glm_source_tokens=dict(chat=cv.get("srctok_chat", 0), raw=cv.get("srctok_raw", 0), fallback=cv.get("srctok_fallback", 0)),
            inkling_tokens=dict(chat=cv.get("tok_chat", 0), raw=cv.get("tok_raw", 0), fallback=cv.get("tok_fallback", 0)),
            fallback_errors=cv.get("fallback_errors", [])),
        source=dict(dir=f"{SRC}/{group}", manifest_sha256=sha(f"{SRC}/{group}/manifest.json"),
                    tokens_sha256=src_man.get("tokens_sha256"), windows_sha256=src_man.get("windows_sha256"),
                    boundary_counts_fit=src_man.get("boundary_counts_fit"), boundary_counts_val=src_man.get("boundary_counts_val"),
                    tokens=src_man.get("tokens")),
        seed=SEED, pad_token_id=IF.PAD, eos_token_id=IF.EOS,
        glm_tokenizer_sha256=sha("/tmp/nq_inkling/glm_tok/tokenizer.json"),
        glm_chat_template_sha256=sha("/tmp/nq_inkling/glm_tok/chat_template.jinja"),
        inkling_tokenizer_sha256=sha("/tmp/nq_inkling/inktok/tokenizer.json"),
        inkling_chat_template_sha256=sha("/tmp/nq_inkling/inktok/chat_template.jinja"),
        tokens_sha256=sha(f"{out}/tokens.npy"), segments_sha256=sha(f"{out}/segments.npy"),
        bnd_think_d_sha256=sha(f"{out}/bnd_think_d.npy"), bnd_end_d_sha256=sha(f"{out}/bnd_end_d.npy"),
        windows_sha256=sha(f"{out}/windows.jsonl"), split_sha256=sha(f"{out}/split.json"),
        code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                     for f in ("reassemble.py", "glmparse.py", "convert.py", "inkfmt.py", "pack.py")},
        boundary_semantics=("bnd_*_d[w,p] = b-p (1..32) for the next boundary token b in the same window segment, else 0; "
                            "think = <|end_message|> closing a non-empty <|message_model|><|content_thinking|> message; "
                            "end = <|end_message|> closing the last model message of an assistant turn (the token right "
                            "before <|content_model_end_sampling|>=EOS 200006); exclusive, nearer wins, ties -> end"),
        format=("Chats: GLM template inverted exactly (glmparse.py; GLM re-render == decoded source text for every chat "
                "fragment) and rendered with Inkling chat_template.jinja, add_generation_prompt=False, reasoning_effort "
                "0.9 if any assistant turn has non-empty reasoning else 0; the template closes every assistant turn with "
                "<|content_model_end_sampling|> (EOS). GLM tool declarations -> Inkling tool_declare; tool results get "
                "the positional tool-call name in their header; GLM arg values parsed as JSON only when non-string and "
                "byte-identical under tojson (declared string params stay strings). Docs whose source was cut keep the "
                "cut: a fragment ending mid-message ends mid-message (no <|end_message|>), one ending right after "
                "</think> ends at the thinking <|end_message|>, a turn cut after its tool calls ends without EOS; a "
                "fragment starting mid-message starts mid-message (no header, no effort preamble). Raw docs: GLM "
                "decode -> Inkling encode, no special tokens. No BOS (Inkling template uses none)."),
        packing=("one fragment per contiguous source run; fragments > C cut into <=C pieces (cut moved <=C/2 earlier to "
                 "keep each boundary's 32-row approach in one piece); full pieces own a window, the rest first-fit-"
                 "decreasing; padding = token 199999 (<|endoftext|>) with segment -1; windows shuffled (seed) within "
                 "split, fit first; positions/attention reset per window segment; segment_id = doc_index (+k*2^20 for "
                 "further pieces of one doc in a window)"),
    )
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    print(group, json.dumps({k: man[k] for k in ("windows", "fit_windows", "val_windows", "documents", "fit_tokens", "val_tokens",
                                               "pad_tokens", "boundary_counts_fit", "boundary_counts_val", "category_tokens")}, indent=1))


if __name__ == "__main__":
    for g in sys.argv[1:] or ["c2048", "c2048_traces"]:
        main(g)
