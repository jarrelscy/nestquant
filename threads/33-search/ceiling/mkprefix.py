#!/usr/bin/env python3
"""T33l: choose N prefixes (block-aligned positions inside the 2048-token T18 windows) -> $OUTC/prefixes.json (PRIVATE)."""
import json, sys
import numpy as np
import clib as C
T = C.T
rng = np.random.default_rng(33)
out = []
for corpus, n in (("glm52-heldout", 64), ("calib-fit", 64)):
    tok = np.load(f"{T.CORP}/{corpus}.npy")
    nw = len(tok) // T.SEQ
    wins = list(range(nw)) * (n // nw) + sorted(rng.choice(nw, n - (n // nw) * nw, replace=False).tolist())
    for w in wins:
        p = int(rng.integers(256 // 16, 1984 // 16 + 1)) * 16
        base = w * T.SEQ
        out.append(dict(corpus=corpus, win=int(w), p=p, gk=int((base + p) // T.G - 1),
                        prefix=tok[base:base + p].astype(int).tolist(), cont=tok[base + p:base + p + 64].astype(int).tolist()))
# interleave corpora so every device gets both
h = [x for x in out if x["corpus"] == "glm52-heldout"]; c = [x for x in out if x["corpus"] == "calib-fit"]
rng.shuffle(h); rng.shuffle(c)
mix = [x for pair in zip(h, c) for x in pair]
json.dump(mix, open(f"{C.OUTC}/prefixes.json", "w"))
print(len(mix), np.mean([x["p"] for x in mix]))
# tiny CPU test file: 2 prefixes of 96 tokens
t = [dict(x, prefix=x["prefix"][:96], cont=x["prefix"][96:160] if len(x["prefix"]) >= 160 else x["cont"], p=96) for x in mix[:2]]
json.dump(t, open(f"{C.OUTC}/prefixes_test.json", "w"))
