#!/usr/bin/env python3
"""Task 3/4: partial (hedged) level-4 upgrades.  Per layer, a level-4 budget of 77 full-P4-record equivalents (VRAM) is
spent on k fixed full experts + nF floating full (gate/up P4 + down P4) + nD floating down-only (down P4 only), with
k + nF + B_D*nD <= 77.  Selection from the v2 scores S (sync, lag 0):
  union  = top (nF + nD) non-fixed by S, residents (any plane) boosted x(1+hm_any)
  full   = top nF of the union by S, full residents boosted x(1+hm_full);  down-only = union - full
Upload churn in full-record units per layer per refresh: new down plane = B_D, new gate/up plane = B_GU.
Metrics (all-slot, mean over layers):
  sal_k{kap}  benefit-weighted coverage sum sal * frac / sum sal, frac = 1 (full) | share_down(kappa) (down-only) | 0
              share_down = dp_down / (dp_down + kappa*(dp_gate+dp_up)), per expert from the manifest proxy_rot (kappa=1
              = T31's rms3 decomposition; README: measured kappa ~0.9-1.0)
  dsal_k1     same, weighted additionally by the expert's delta (T31) -> error energy removed / removable
  full        plain sal-hot of the full tier (+fixed); any = sal-hot counting down-only as hot
  hedge.py CORPUS [TAG]  (env GRID=small|big) -> $A/hedge_CORPUS[_TAG].json"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib as A

corpus = sys.argv[1] if len(sys.argv) > 1 else ""
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
B_GU, B_D = 1585152 / 2558208, (915460 + 57344) / 2558208      # tp4 p4rec segments: gu.p4+gu.d4 | dn.p4+dn.d4+lr4
KAPS = tuple(float(x) for x in os.environ.get("KAPS", "0.7,0.9,1.0,1.4").split(","))
PD = dict(np.load(f"{A.A}/proj_delta.npz"))          # eager: a lazy NpzFile shared across fork corrupts
LI = {int(L): i for i, L in enumerate(PD["layers"])}


def configs():
    g = os.environ.get("GRID", "small")
    KS = [int(x) for x in os.environ.get("KS", "0,26").split(",")]
    out = []
    for k in KS:
        tot = 77 - k
        for nF in sorted(({tot, tot - 8, tot - 16, tot - 24, tot - 32} if g == "small" else {tot, tot - 8, tot - 16}) if g != "big" else range(tot, tot - 41, -4),
                         reverse=True):
            if nF < 0:
                continue
            nD = int(np.floor((tot - nF) / B_D + 1e-9))
            hms_a = [0.3, 0.5, 0.7, 1.0] if g != "big" else [0.2, 0.35, 0.5, 0.7, 1.0, 1.4]
            hms_f = [0.5] if nD == 0 else ([0.3, 0.6, 1.0] if g != "big" else [0.2, 0.4, 0.7, 1.0, 1.5])
            for ha, hf in itertools.product(hms_a, hms_f):
                out.append((k, nF, nD, ha, hf if nD else ha))
    return out


def sim2(S, fx, start, nF, nD, ha, hf, nbc=A.NBC):
    """-> full [nb,256] bool, down-only [nb,256] bool (floating only; fixed excluded)."""
    nb = S.shape[0]
    fxm = np.zeros(256, bool); fxm[fx] = True
    st = [e for e in start if not fxm[e]]
    f0 = np.zeros(256, bool); f0[st[:nF]] = True
    d0 = np.zeros(256, bool); d0[st[nF:nF + nD]] = True
    FU = np.zeros((nb, 256), bool); DO = np.zeros((nb, 256), bool)
    nu = nF + nD
    for c0 in range(0, nb, nbc):
        full, down = f0.copy(), d0.copy()
        for k in range(c0, min(c0 + nbc, nb)):
            FU[k], DO[k] = full, down
            s = S[k]
            if np.where(fxm, 0, np.maximum(s, 0)).sum() <= 0:
                continue
            anyr = full | down
            v = np.where(fxm, -np.inf, s).astype(np.float32)
            va = np.where(anyr, v * np.float32(1 + ha), v)
            u = np.zeros(256, bool); u[np.argsort(-va, kind="stable")[:nu]] = True
            if nD:
                vf = np.where(u, np.where(full, v * np.float32(1 + hf), v), -np.inf)
                nf_ = np.zeros(256, bool); nf_[np.argsort(-vf, kind="stable")[:nF]] = True
            else:
                nf_ = u
            full, down = nf_ & ~fxm, u & ~nf_ & ~fxm
    return FU, DO


def job(L):
    S, bs, bc = A.load(corpus, L)
    nb = S.shape[0]
    i = LI[L]
    dp = PD["dp"][i]
    delta = PD["delta"][i]
    shd = {kap: dp[:, 2] / (dp[:, 2] + kap * (dp[:, 0] + dp[:, 1])) for kap in KAPS}
    f26 = A.f26_ranked(L)
    tot = bs.sum()
    wsd = bs * delta[None]
    totd = wsd.sum()
    out = {}
    for (k, nF, nD, ha, hf) in configs():
        fx = f26[:k]
        FU, DO = sim2(S, fx, f26[k:] + list(A.fdef[L]), nF, nD, ha, hf)
        chb = float(((FU[1:] & ~FU[:-1]) * B_GU + ((FU[1:] | DO[1:]) & ~(FU[:-1] | DO[:-1])) * B_D).sum(1).mean())
        FU[:, fx] = True
        r = dict(churn_b=chb, churn_e=float(((FU[1:] | DO[1:]) & ~(FU[:-1] | DO[:-1])).sum(1).mean()),
                 full=float((bs * FU).sum() / tot), any=float((bs * (FU | DO)).sum() / tot))
        for kap in KAPS:
            fr = FU + DO * shd[kap][None]
            r[f"sal_k{kap}"] = float((bs * fr).sum() / tot)
        r["dsal_k1.0"] = float((wsd * (FU + DO * shd[1.0][None])).sum() / totd)
        out[f"{k}|{nF}|{nD}|{ha}|{hf}"] = r
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "18"))) as p:
        res = dict(p.map(job, A.T.LAYERS))
    summ = {key: {m: float(np.mean([res[L][key][m] for L in A.T.LAYERS])) for m in res[3][key]} for key in res[3]}
    for key, s in summ.items():
        print(f"{key:22s} churnB {s['churn_b']:5.2f} full {s['full']*100:6.2f} any {s['any']*100:6.2f} " +
              " ".join(f"k{kap} {s[f'sal_k{kap}']*100:6.2f}" for kap in KAPS) + f" dsal {s['dsal_k1.0']*100:6.2f}",
              flush=True)
    json.dump(dict(corpus=corpus, B_GU=B_GU, B_D=B_D, summary=summ), open(f"{A.A}/hedge_{corpus}{'_' + TAG if TAG else ''}.json", "w"))
