#!/usr/bin/env python3
"""sm120tf (and any scalelib stream): k x hm sweep with v2 (all 256 experts scored; scalelib features) and the serve
default gbdt_x_mps (native band rebuilt per k from scalelib e256), plus the 69-full + 21-down-only hedge.  Chains =
the stream's own segments (one predictor state per task = session carry-over across its requests); churn within chains.
  seval.py STREAM  -> $A/seval_STREAM.json"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import alib as A
import hedge as H
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
import scalelib as SL  # noqa: E402

stream = sys.argv[1]
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
S5 = "/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt"
KS = [int(x) for x in os.environ.get("KS", "0,6,13,26").split(",")]
HMS = [float(x) for x in os.environ.get("HMS", "0.2,0.3,0.4,0.5,0.6,0.7,0.85,1.0,1.25,1.5,2.0,3.0").split(",")]
HEDGE = [(69, 21, ha, hf) for ha in (0.5, 0.7, 1.0) for hf in (0.6, 1.0)] if not os.environ.get("NOHEDGE") else []


def bsum(L):
    return float(np.load(f"{SL.BLK}/{stream}/L{L}.npz")["bsal"].astype(np.float64).sum())


def job(L):
    import lightgbm as lgb
    D = SL.load(stream, L)
    sg = D["sg"]
    F, e256 = SL.feats(D)
    nb = F.shape[0]
    bs, bc = D["bs"], D["bc"]
    Sv2 = lgb.Booster(model_file=V2).predict(F.reshape(-1, 9), num_threads=1).reshape(nb, 256).astype(np.float32)
    b5 = lgb.Booster(model_file=S5)
    f26 = A.f26_ranked(L)
    i = H.LI[L]; dp = H.PD["dp"][i]
    shd = {kap: dp[:, 2] / (dp[:, 2] + kap * (dp[:, 0] + dp[:, 1])) for kap in (0.7, 0.9, 1.0, 1.4)}
    tot = bs.sum()
    out = {}
    for k in KS:
        fx = f26[:k]
        fxm = np.zeros(256, bool); fxm[fx] = True
        order = np.argsort(-np.where(fxm[None], -np.inf, e256), 1, kind="stable")
        c, top = order[:, 20:121], order[:, :20]
        pr = b5.predict(np.take_along_axis(F[..., :5], c[..., None], 1).reshape(-1, 5), num_threads=1).reshape(c.shape)
        pr = pr * np.take_along_axis(F[..., 8], c, 1)
        Ss5 = np.zeros((nb, 256), np.float32)
        np.put_along_axis(Ss5, c, pr.astype(np.float32), 1)
        np.put_along_axis(Ss5, top, (1e3 + np.take_along_axis(e256, top, 1)).astype(np.float32), 1)
        for pn, S in (("v2", Sv2), ("s5", Ss5)):
            for hm in HMS:
                sv = A.sim_seg(S, fx, f26[k:] + list(A.fdef[L]), 77 - k, hm, sg=sg)
                ch = A.churn_seg(sv, sg)
                sv[:, fx] = True
                out[f"{pn}|{k}|{hm}"] = dict(sal=float((bs * sv).sum() / tot), cnt=float((bc * sv).sum() / bc.sum()),
                                              churn=ch)
        if k == 0 and os.environ.get("ALLOC"):
            st = f26 + list(A.fdef[L]); cal = A.load("calib-fit", L)[1].sum(0)
            st = st + [int(e) for e in np.argsort(-cal, kind="stable") if e not in set(st)]
            for af in os.environ["ALLOC"].split(","):
                nm = os.path.basename(af)[6:-5]
                nfL = json.load(open(af))[str(L)]
                for hm in HMS:
                    sv = A.sim_seg(Sv2, [], st, nfL, hm, sg=sg)
                    out[f"v2{nm}|0|{hm}"] = dict(sal=float((bs * sv).sum() / tot), cnt=float((bc * sv).sum() / bc.sum()),
                                                 churn=A.churn_seg(sv, sg), tot=float(tot))
        if k == 0:
            for (nF, nD, ha, hf) in HEDGE:
                FU = np.zeros((nb, 256), bool); DO = np.zeros((nb, 256), bool)
                for (s, e) in sg:
                    FU[s:e], DO[s:e] = H.sim2(Sv2[s:e], fx, f26 + list(A.fdef[L]), nF, nD, ha, hf, nbc=e - s)
                any_ = FU | DO
                nw = 0.0; n = 0
                for (s, e) in sg:
                    nw += ((FU[s + 1:e] & ~FU[s:e - 1]) * H.B_GU + (any_[s + 1:e] & ~any_[s:e - 1]) * H.B_D).sum()
                    n += e - s - 1
                r = dict(churn=float(nw / n), full=float((bs * FU).sum() / tot))
                for kap, sh in shd.items():
                    r[f"sal_k{kap}"] = float((bs * (FU + DO * sh[None])).sum() / tot)
                r["sal"] = r["sal_k0.9"]
                out[f"hedge{nF}_{nD}|0|{ha}_{hf}"] = r
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "18"))) as p:
        R = dict(p.map(job, A.T.LAYERS))
    summ = {key: {m: float(np.mean([R[L][key][m] for L in A.T.LAYERS])) for m in R[3][key]} for key in R[3]}
    tl = {L: bsum(L) for L in A.T.LAYERS}
    for key in summ:
        summ[key]["pooled"] = float(sum(R[L][key]["sal"] * tl[L] for L in tl) / sum(tl.values()))
    for key, s in summ.items():
        print(f"{key:22s} sal {s['sal']*100:6.2f} churn {s['churn']:5.2f}" +
              f" pooled {s['pooled']*100:6.2f}" + ("".join(f" k{kap} {s[f'sal_k{kap}']*100:6.2f}" for kap in (0.7, 0.9, 1.0, 1.4)) if "sal_k1.0" in s else ""), flush=True)
    json.dump(dict(stream=stream, summary=summ, per_layer=R), open(f"{A.A}/seval_{stream}{os.environ.get('TAG', '')}.json", "w"))
