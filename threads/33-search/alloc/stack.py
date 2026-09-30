#!/usr/bin/env python3
"""Task 4 stacked candidate in T33j's request-start protocol (threads/33-search/scale/reqstart.py): every full
8192-token chain is split into prompt = blocks [0,P) and decode = blocks [P,512); the predictor restarts at decode start.
  cold      state reset, start set (fixed-26 by salience, then floating_default)   (serve today)
  warmmean  16 synthetic blocks of the prompt's mean per-block counts/salience fed before decode (T33j), metrics skip them
Each arm at k fixed in {0, 26} (and 6) x hm grid; v2 scores for all 256 experts (scalelib features, bit-identical to
rows_bandall).  Metric: all-slot sal-hot over decode tokens [0,256), [0,1024), all decode; churn within decode.
  stack.py STREAM [P]  -> $A/stack_STREAM.json"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib as A
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
import scalelib as SL  # noqa: E402

stream = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 32
MODEL = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
KEYS = ("bc", "bs", "bca", "nans", "segl")
GRID = {0: [0.5, 0.6, 0.7, 0.85, 1.0, 1.25], 6: [0.5, 0.7, 0.85, 1.0], 26: [0.3, 0.4, 0.5, 0.7]}


def sub(parts):
    out = {k: [] for k in KEYS}; sg = []; n = 0
    for chain in parts:
        s = n
        for blk in chain:
            for k in KEYS:
                out[k].append(blk[k])
            n += len(blk["nans"])
        sg.append((s, n))
    D2 = {k: np.concatenate(v) for k, v in out.items()}; D2["sg"] = sg
    return D2


def scores(b, D2):
    F, _ = SL.feats(D2)
    nb = F.shape[0]
    return b.predict(F.reshape(-1, 9), num_threads=1).reshape(nb, 256).astype(np.float32)


def job(L):
    import lightgbm as lgb
    b = lgb.Booster(model_file=MODEL)
    D = SL.load(stream, L)
    chains = [(s, e) for (s, e) in D["sg"] if e - s == A.NBC]
    sl = lambda s, e: {k: D[k][s:e] for k in KEYS}
    dec = [sl(s + P, e) for s, e in chains]
    syn = []
    for s, e in chains:
        m = {k: np.repeat(D[k][s:s + P].mean(0, keepdims=True), 16, 0) for k in KEYS}
        m["segl"] = np.repeat(D["segl"][s + P - 1:s + P], 16)
        syn.append(m)
    f26 = A.f26_ranked(L)
    arms = {"cold": (sub([[d] for d in dec]), 0)}
    if os.environ.get("WARM"):      # prefill warm-start dropped (T33j: hurts on real SM120 request structure)
        arms["warmmean"] = (sub([[m, d] for m, d in zip(syn, dec)]), 16)
    res = {}
    for an, (D2, skip) in arms.items():
        S = scores(b, D2)
        keep = np.concatenate([np.arange(s + skip, e) for s, e in D2["sg"]])
        pos = np.concatenate([np.arange(e - s - skip) for s, e in D2["sg"]])
        bs = D2["bs"][keep]
        for k, hms in GRID.items():
            fx = f26[:k]
            for hm in hms:
                sv = A.sim_seg(S, fx, f26[k:] + list(A.fdef[L]), 77 - k, hm, sg=D2["sg"])
                new = np.zeros(sv.shape[0]); new[1:] = (sv[1:] & ~sv[:-1]).sum(1)
                sv[:, fx] = True
                sv, new = sv[keep], new[keep]
                h = (bs * sv).sum(1); t = bs.sum(1)
                r = {}
                for nm, (lo, hi) in (("d256", (0, 16)), ("d1024", (0, 64)), ("all", (0, 10 ** 9))):
                    mk = (pos >= lo) & (pos < hi)
                    r[nm] = float(h[mk].sum() / t[mk].sum())
                r["churn"] = float(new[pos > 0].mean())
                res[f"{an}|k{k}|{hm}"] = r
    return L, res


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "18"))) as p:
        R = dict(p.map(job, A.T.LAYERS))
    summ = {key: {m: float(np.mean([R[L][key][m] for L in A.T.LAYERS])) for m in R[3][key]} for key in R[3]}
    for key, s in summ.items():
        print(f"{key:18s} dec[0,256) {s['d256']*100:6.2f}  dec[0,1024) {s['d1024']*100:6.2f}  all {s['all']*100:6.2f}  churn {s['churn']:5.2f}", flush=True)
    json.dump(dict(stream=stream, P=P, summary=summ), open(f"{A.A}/stack_{stream}.json", "w"))
