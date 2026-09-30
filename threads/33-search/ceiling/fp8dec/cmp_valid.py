"""T33l fp8dec: compare dec.py validation runs against references (top-8 set overlap, mean over layers & tokens).
  cmp_valid.py VAL_DIR   (expects VAL_DIR/{A_force,A_tf,B_force,B_tf}/ where present + vref.npz)"""
import json
import os
import sys

import numpy as np

V = sys.argv[1]
ref = np.load(f"{V}/vref.npz")


def ov(a, b):
    a = np.sort(a.reshape(-1, 8).astype(np.int16), 1); b = b.reshape(-1, 8).astype(np.int16)
    return float(np.mean([(np.isin(b[i], a[i])).sum() / 8 for i in range(len(a))]))


def rows(run, span):
    """-> per task routing [nsp, n, 8] for span 'cont' (the 64 tf tokens) or 'tail' (last 64 prompt tokens)."""
    d = f"{V}/{run}"
    if not os.path.exists(f"{d}/index.json"):
        return None
    ix = json.load(open(f"{d}/index.json"))
    out = []
    rid = {}
    for t in ix["tasks"]:
        g = t["g"]
        if g not in rid:
            rid[g] = np.load(f"{d}/rid.g{g}.npy", mmap_mode="r")
        P = t["prompt_len"]
        if ix["mode"] == "tf":            # prompt_len includes the 64 tf tokens
            c0 = t["off"] + P - 64
        else:
            c0 = t["off"] + P
        n = 64 if ix["mode"] == "tf" else t["n_dec"]
        s = slice(c0, c0 + n) if span == "cont" else slice(c0 - 64, c0)
        out.append(np.asarray(rid[g][:, s]))
    return np.stack(out)


def rep(name, a, b):
    if a is None or b is None:
        print(f"{name:44s} n/a"); return
    n = min(a.shape[2], b.shape[2]); a, b = a[:, :, :n], b[:, :, :n]
    per_layer = [ov(a[:, j], b[:, j]) for j in range(a.shape[1])]
    print(f"{name:44s} top-8 overlap {100 * np.mean(per_layer):6.2f}%  (min layer {100 * np.min(per_layer):6.2f}%)")


Af, At = rows("A_force", "cont"), rows("A_tf", "cont")
rep("vA force(decode path) vs gen.py decode", Af, ref["refA_gen"])
rep("vA tf(prefill path) vs gen.py decode", At, ref["refA_gen"])
rep("vA force vs tf", Af, At)
rep("vA tf vs T32 trace (cont)", At, ref["trA"][:, :, 64:])
rep("vA force: prompt tail (prefill) vs trace", rows("A_force", "tail"), ref["trA"][:, :, :64])
Bf, Bt = rows("B_force", "cont"), rows("B_tf", "cont")
if Bf is None and rows("B_force8", "cont") is not None:
    Bf = rows("B_force8", "cont")[:4]                 # vB_force8 = the 4 vB tasks + duplicates (>= 1 task per device)
rep("vB (3000-token ctx) force vs tf [indexer on]", Bf, Bt)
rep("vB tf (KV carry) vs trace (no carry)", Bt, ref["trB"])
