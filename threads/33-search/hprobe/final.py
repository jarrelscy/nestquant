#!/usr/bin/env python3
"""final: best arm lgh|cfhpx|lam1000|a0.5 (ridge residual probe on v2, trained on calib-fit + hpx) -> heldout scores
(scores_glm52-heldout/L.npy float16 [nb,256]); eval k=26 (hm .4/.5/.6) and k=0 (T32 k0: no fixed, 77 floating,
rows_bandk0 targets, v2 k0 offset; hm .5/.6/.7) vs v2."""
import json
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
from multiprocessing import Pool  # noqa: E402
import numpy as np  # noqa: E402
import heval as H  # noqa: E402
import hplib as HL  # noqa: E402
import t32lib as T  # noqa: E402
import exp4 as E  # noqa: E402

A, LAM = 0.5, 1000.0
FDEF77 = {L: list(H.FIXED[L]) + [e for e in H.FDEF[L] if e not in set(H.FIXED[L])] for L in H.FIXED}
HK26, HK0 = (0.4, 0.5, 0.6), (0.5, 0.6, 0.7)


def k0_eval(S, L, bs, bc, hm):
    sv = T.sim_layer(S, [], FDEF77[L], nf=77, hm=hm, lag=0)
    return dict(sn=float((bs * sv).sum()), sd=float(bs.sum()), cn=float((bc * sv).sum()), cd=float(bc.sum()),
                ch=float((sv[1:] & ~sv[:-1]).sum()), chn=float(len(sv) - 1), ch_in=0.0, chn_in=1.0)


def job(L):
    import lightgbm as lgb
    CS = ("calib-fit", "glm52-heldout", "hpx")
    d = {c: E.rows(c, L) for c in CS}
    off = {c: np.log1p(np.maximum(np.load(f"{E.P}/v2S_{c}/L{L}.npy"), 0)) for c in CS}
    Y = {c: np.log1p(HL.target(d[c], L)) - off[c] for c in ("calib-fit", "hpx")}
    raw = {c: E.pool(c, L) for c in CS}
    lg = {c: np.hstack([raw[c][1], HL.bmean(raw[c][1], 4), HL.bema(raw[c][1], 256)]).astype(np.float32) for c in CS}
    h16 = {c: raw[c][0].astype(np.float32) for c in CS}
    del raw
    h64 = {c: HL.bmean(h16[c], 4) for c in CS}
    mu, V = E.pca(np.vstack([h64["calib-fit"], h64["hpx"]]), L)
    X = {c: np.hstack([lg[c], (h16[c] - mu) @ V, (h64[c] - mu) @ V]) for c in CS}
    f = E.ridge(np.vstack([X["calib-fit"], X["hpx"]]), np.vstack([Y["calib-fit"], Y["hpx"]]), LAM)
    Ph = f(X["glm52-heldout"])
    out = {}
    dh = d["glm52-heldout"]; bs, bc = dh["bsal"].astype(np.float64), dh["bcnt"].astype(np.float64)
    S = np.expm1(off["glm52-heldout"] + A * Ph).astype(np.float32)
    np.save(f"{H.HP}/scores_glm52-heldout/L{L}.npy", S.astype(np.float16))
    S0 = np.expm1(off["glm52-heldout"]).astype(np.float32)
    for hm in HK26:
        out[f"hp|k26|hm{hm}"] = H.eval_S(S, L, bs, bc, hm=hm)
        out[f"v2|k26|hm{hm}"] = H.eval_S(S0, L, bs, bc, hm=hm)
    # k=0
    dk = np.load(f"{T.OUT}/rows_bandk0/glm52-heldout/L{L}.npz")
    Xk = np.concatenate([dk["X"], np.load(f"{T.OUT}/rows_v2_bandk0/glm52-heldout/L{L}.npz")["X2"]], -1)
    b = lgb.Booster(model_file=H.V2)
    Sk = T.score_blocks(b.predict(Xk.reshape(-1, Xk.shape[-1]), num_threads=1), dk["cand"], dk["top"], dk["e256"])
    Sk1 = np.expm1(np.log1p(np.maximum(Sk, 0)) + A * Ph).astype(np.float32)
    bs0, bc0 = dk["bsal"].astype(np.float64), dk["bcnt"].astype(np.float64)
    for hm in HK0:
        out[f"hp|k0|hm{hm}"] = k0_eval(Sk1, L, bs0, bc0, hm)
        out[f"v2|k0|hm{hm}"] = k0_eval(Sk.astype(np.float32), L, bs0, bc0, hm)
    np.savez(f"{E.P}/final_probe/L{L}.npz", mu=mu, V=V)
    return L, out


if __name__ == "__main__":
    os.makedirs(f"{H.HP}/scores_glm52-heldout", exist_ok=True); os.makedirs(f"{E.P}/final_probe", exist_ok=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = dict(p.imap_unordered(job, T.LAYERS))
    summ = {}
    for n in sorted(res[T.LAYERS[0]]):
        s = H.summarise({L: res[L][n] for L in T.LAYERS}); summ[n] = s
        print(f"{n:22s} sal {s['sal']:6.2f} churn {s['churn']:5.2f}", flush=True)
    json.dump(summ, open(f"{H.HP}/final.json", "w"), indent=1)
