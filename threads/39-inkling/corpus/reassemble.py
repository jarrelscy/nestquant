"""Reassemble GLM-format documents from packed windows. Output work/<group>_docs.jsonl:
one line per source_id: split, category, kind, doc_index, pieces [(token_offset, n)], ids (concatenated in offset order),
contiguous (offsets form a gapless run), starts_at0."""
import json, sys, collections
import numpy as np
SRC = "/home/coder/nestquant-corpus-backup/glm53_calib_glmfmt_v1"
OUT = "/tmp/nq_inkling/corpus/work"
for g in ("c2048", "c2048_traces"):
    tok = np.load(f"{SRC}/{g}/tokens.npy", mmap_mode="r"); seg = np.load(f"{SRC}/{g}/segments.npy", mmap_mode="r")
    sp = json.load(open(f"{SRC}/{g}/split.json"))
    docs = {}
    for w, line in enumerate(open(f"{SRC}/{g}/windows.jsonl")):
        role = "fit" if sp["fit"][0] <= w < sp["fit"][1] else "val"
        for s in json.loads(line)["segments"]:
            a, n = s["window_offset"], s["tokens"]
            assert (seg[w, a:a + n] == s["segment_id"]).all()
            d = docs.setdefault(s["source_id"], dict(source_id=s["source_id"], category=s["category"], kind=s["kind"],
                                                     doc_index=s["doc_index"], split=role, parts=[]))
            assert d["split"] == role, ("doc crosses split", s["source_id"])
            assert d["category"] == s["category"] and d["kind"] == s["kind"]
            d["parts"].append((s["token_offset"], w, a, n))
    st = collections.Counter()
    with open(f"{OUT}/{g}_docs.jsonl", "w") as f:
        for sid, d in docs.items():
            parts = sorted(d.pop("parts"))
            ids, pieces, contig = [], [], True
            for off, w, a, n in parts:
                if pieces and off != pieces[-1][0] + pieces[-1][1]:
                    contig = False
                    assert off > pieces[-1][0] + pieces[-1][1], ("overlap", sid)
                pieces.append((off, n)); ids.extend(int(x) for x in tok[w, a:a + n])
            d.update(pieces=pieces, contiguous=contig, starts_at0=pieces[0][0] == 0, ids=ids)
            st[f"{d['split']}_{d['kind']}_docs"] += 1; st[f"{d['split']}_tokens"] += len(ids)
            st["noncontig"] += not contig; st["not_at0"] += pieces[0][0] != 0
            st[f"not_at0_{d['kind']}"] += pieces[0][0] != 0
            f.write(json.dumps(d) + "\n")
    print(g, len(docs), dict(st))
