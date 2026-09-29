"""multi-horizon blend arms: V = sum_h w_h * rate_h (rate = head prediction / horizon blocks), mult hysteresis sweep.
calib: val chains only (chain % 4 == 3).  -> $W/arms_blend_{tag}.npz"""
import sys, itertools, numpy as np
from multiprocessing import Pool
import dlib as D
HEADS = [("h16", 1), ("h64", 4), ("h128", 8), ("h256", 16)]
HMS = (0.2, 0.35, 0.5, 0.7, 1.0)
step = 0.25
W = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=4) if abs(sum(w) - 1) < 1e-9]
ARMS = [("v2", None, h) for h in HMS] + [("blend", w, h) for w in W for h in HMS]

def load(corpus, name, L):
    S = D.load_S(name, corpus, L).astype(np.float64)
    if corpus == "calib-fit":
        S = S[D.val_rows(S.shape[0])]
    return S

def job(args):
    corpus, L = args
    bs, _ = D.load_blk(corpus, L)
    if corpus == "calib-fit": bs = bs[D.val_rows(bs.shape[0])]
    fx = D.fx_mask(L); fd = D.fd_mask(L)
    R = {n: load(corpus, n, L) / hz for n, hz in HEADS}; v2 = load(corpus, "v2", L)
    out = []
    for kind, w, hm in ARMS:
        S = v2 if kind == "v2" else sum(wi * R[n] for wi, (n, _) in zip(w, HEADS) if wi)
        sv = D.replay(S, S * (1 + hm), fx, fd, D.NBC, 0)
        out.append(D.evaluate(sv, bs, fx))
    return out

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "a"
    res = {}
    with Pool(20) as pool:
        for corpus in ("calib-fit", "glm52-heldout"):
            res[corpus.replace("-", "_")] = np.array(pool.map(job, [(corpus, L) for L in D.LAYERS]))
    names = np.array([f"{k}:{'' if w is None else ','.join(f'{x:.2f}' for x in w)}:{h}" for k, w, h in ARMS])
    np.savez(f"{D.W}/arms_blend_{tag}.npz", **res, arms=names)
    # summary: per weight vector, interp at churn 3.2
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(n.rsplit(":", 1)[0], []).append(i)
    rows = []
    for g, idx in groups.items():
        c = res["calib_fit"][:, idx].mean(0); h = res["glm52_heldout"][:, idx].mean(0)
        rows.append((D.interp_at(c[:, 0], c[:, 1]), D.interp_at(h[:, 0], h[:, 1]), g, c[:, 1].min(), c[:, 1].max()))
    rows.sort(reverse=True)
    for r in rows[:15] + [r for r in rows if r[2].startswith("v2")]:
        print(f"{r[2]:32s} calib@3.2 {r[0]*100:.2f}  held@3.2 {r[1]*100:.2f}   (calib churn range {r[3]:.2f}-{r[4]:.2f})")
