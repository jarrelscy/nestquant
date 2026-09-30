#!/usr/bin/env python3
"""T33l: gen.npz vs T32 FP8 trace: top-8 set overlap of (a) last-64 prefix tokens (prefill path), (b) the teacher-forced
real continuation (decode path).  valid.py PREFIXES GEN_DIR [layers]"""
import json, sys
import numpy as np
import clib as C
T = C.T
PX = json.load(open(sys.argv[1]))
g = np.load(f"{sys.argv[2]}/gen.npz")
order, K1, steps = g["order"], int(g["k"]) + 1, int(g["steps"])
sl = g["sparse_layers"].tolist()
Ls = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else sl
cache = {}
def ov(a, b):
    return float(np.mean([len(set(x) & set(y)) / 8 for x, y in zip(a.reshape(-1, 8), b.reshape(-1, 8))]))
res = []
for L in Ls:
    j = sl.index(L)
    pa, ca, wa = [], [], []
    for q, i in enumerate(order):
        x = PX[i]
        key = (L, x["corpus"])
        if key not in cache:
            cache = {key: T.load_layer(L, x["corpus"])}
        ids, w, xn = cache[key]
        base = x["win"] * T.SEQ
        P = x["p"]
        pa.append(ov(g["pids"][j, q], ids[base + P - 64:base + P]))
        n = min(steps, len(x["cont"]))
        ca.append(ov(g["ids"][j, q * K1 + K1 - 1, :n], ids[base + P:base + P + n]))
        wa.append(float(np.abs(g["xn"][j, q * K1 + K1 - 1, :n] / xn[base + P:base + P + n] - 1).mean()))
    res.append((L, np.mean(pa), np.mean(ca), np.mean(wa)))
    print(f"L{L} prefill-overlap {np.mean(pa):.4f} decode-TF-overlap {np.mean(ca):.4f} |xn rel err| {np.mean(wa):.4f}", flush=True)
print("mean", np.mean([r[1] for r in res]), np.mean([r[2] for r in res]))
