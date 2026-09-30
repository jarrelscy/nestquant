#!/usr/bin/env python3
"""T33l task 2: realisation-noise split on the FP8 traces.  Per layer, per-token salience split into two halves
(interleaved even/odd tokens; random tokens) -> block matrices A, B.  Arms (all sync refresh-16, hysteresis grid,
scored as all-slot sal-hot share of the HALF-B served-block salience, or of the full served-block salience):
  v2, orc{32,64,128}_{full,A,B}, rate oracles, skip-served oracle.   -> $OUTC/split_CORPUS.json"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import clib as C
T = C.T
corpus = sys.argv[1]
HMS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0]   # additive relative margin dm (x 51st-best score)


def blocks(ids, v, mask):
    Tn = ids.shape[0]; nb = Tn // T.G
    b = (np.arange(Tn) // T.G)[:, None].repeat(8, 1)
    idx = (b * T.NE + ids.astype(np.int64))
    m = np.repeat(mask[:, None], 8, 1)
    s = np.bincount(idx[m], weights=v[m], minlength=nb * T.NE).reshape(nb, T.NE)
    c = np.bincount(idx[m], minlength=nb * T.NE).reshape(nb, T.NE).astype(np.float64)
    return s, c


def fut_skip(M, n):
    """blocks k+2 .. k+1+n (skip the served block k+1)."""
    F = C.fut(M, n + 1)
    return F - C.fut(M, 1)


def rate_sal(S, Cn, b=4.0):
    """count x shrunk per-hit salience (per-hit mean shrunk to the layer-window mean with b pseudo-hits)."""
    mL = S.sum(1, keepdims=True) / np.maximum(Cn.sum(1, keepdims=True), 1)
    return Cn * (S + b * mL) / (Cn + b)


def job(L):
    ids, w, xn = T.load_layer(L, corpus)
    v = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None])
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    bs = d["bsal"].astype(np.float64)
    Tn = ids.shape[0]
    full, cfull = blocks(ids, v, np.ones(Tn, bool))
    assert np.allclose(full, bs, rtol=1e-4, atol=1e-6), L
    fx, fd = C.masks(L)
    S = C.v2_scores(corpus, L)
    rng = np.random.default_rng(1000 + L)
    res = {}

    def run(name, Sc, Y):
        pts = []
        for hm in HMS:
            sv = C.replay(Sc, fx, fd, dm=hm)
            pts.append(C.metric(sv, fx, Y))
        res[name] = pts
    run("full/v2", S, full)
    for n in (1, 2, 4, 8):
        run(f"full/orc{16 * n}", C.fut(full, n), full)
    run("full/orc64_rate", rate_sal(C.fut(full, 4), C.fut(cfull, 4)), full)
    run("full/orc64_cnt", C.fut(cfull, 4), full)
    run("full/orc64_skip", fut_skip(full, 4), full)             # next-64 excluding the served block
    run("full/orc48_skip", fut_skip(full, 3), full)
    run("full/orc64_skip_past16", fut_skip(full, 3) + C.past(full, 1), full)   # k, k+2..k+4 (served block hidden)
    for split in ("par", "rnd"):
        mA = (np.arange(Tn) % 2 == 0) if split == "par" else (rng.random(Tn) < 0.5)
        A, cA = blocks(ids, v, mA); B, cB = blocks(ids, v, ~mA)
        run(f"{split}/v2", S, B)
        for n in (1, 2, 4, 8, 16):
            run(f"{split}/orc{16 * n}_A", C.fut(A, n), B)
            run(f"{split}/orc{16 * n}_B", C.fut(B, n), B)
        run(f"{split}/orc64_full", C.fut(full, 4), B)
        run(f"{split}/orc64_rate_A", rate_sal(C.fut(A, 4), C.fut(cA, 4)), B)
        run(f"{split}/orc64_rate_B", rate_sal(C.fut(B, 4), C.fut(cB, 4)), B)
        run(f"{split}/orc128_rate_A", rate_sal(C.fut(A, 8), C.fut(cA, 8)), B)
    return L, res


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "20"))) as p:
        res = dict(p.map(job, T.LAYERS))
    names = list(res[T.LAYERS[0]])
    out = {}
    for n in names:
        pts = [(float(np.mean([res[L][n][i][0] for L in T.LAYERS])), float(np.mean([res[L][n][i][1] for L in T.LAYERS])))
               for i in range(len(HMS))]
        s32, flag = C.at_churn(pts, 3.2)
        out[n] = dict(pts=pts, hms=HMS, at3p2=s32, flag=flag)
        print(f"{corpus} {n:26s} hm0 {pts[0][0]*100:6.2f}/{pts[0][1]:5.2f}  @churn3.2 {s32*100:6.2f} {flag}", flush=True)
    json.dump(out, open(f"{C.OUTC}/split_{corpus}.json", "w"), indent=1)
