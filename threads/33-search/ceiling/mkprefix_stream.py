#!/usr/bin/env python3
"""T33l stream ceiling: anchors x 8 consecutive decision points (p0 + 16j) -> $OUTC/prefixes_stream.json (PRIVATE)."""
import json
import numpy as np
import clib as C
T = C.T
rng = np.random.default_rng(34)
NPT = 8
out = []
for corpus, n in (("glm52-heldout", 12), ("calib-fit", 4)):
    tok = np.load(f"{T.CORP}/{corpus}.npy")
    nw = len(tok) // T.SEQ
    for a, w in enumerate(sorted(rng.choice(nw, n, replace=False).tolist())):
        p0 = int(rng.integers(256 // 16, (T.SEQ - 64 - 16 * (NPT - 1)) // 16 + 1)) * 16
        base = w * T.SEQ
        for j in range(NPT):
            p = p0 + 16 * j
            out.append(dict(corpus=corpus, win=int(w), p=p, gk=int((base + p) // T.G - 1), anchor=f"{corpus}#{a}", j=j,
                            prefix=tok[base:base + p].astype(int).tolist(), cont=tok[base + p:base + p + 64].astype(int).tolist()))
json.dump(out, open(f"{C.OUTC}/prefixes_stream.json", "w"))
print(len(out), np.mean([x["p"] for x in out]))
