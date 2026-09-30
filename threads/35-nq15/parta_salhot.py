"""T35 Part A (CPU, PRIVATE data): jF (k0, hm 0.7, refresh 16, sync) all-slot salience-hot (pooled over layers and
layer-mean), 4-bit slot-hit share and churn vs hot count H on sm120tf decode traces.  Out-of-sample: each sm120tf
task is scored by the fold net that held it out (jF1/jF2/jF3, T33i), scores precomputed in
/tmp/nestquant/33-search/joint/scores/jF{f}_sm120tf.  Replay = jlib.replay with n_float as a parameter.
  LAYOUT=k0 python parta_salhot.py [H list] -> /tmp/nestquant/35-nq15/parta_salhot.json"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/joint")
os.environ["LAYOUT"] = "k0"
import jlib as J

HS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,10,20,30,40,50,60,77,100").split(",")]
HM = float(os.environ.get("HM", "0.7"))
SC = "/tmp/nestquant/33-search/joint/scores"
FOLD = {"embedding-drift-monitor": 1, "fin-saccr-rwa": 1, "formal-crypto": 2, "sound-change-cascade": 2,
        "freight-dispatch-shift": 3, "pretrain-shard-corruption": 3}
_m = json.load(open(f"{J.BLK}/sm120tf/meta.json"))
TASKS = [n for n, a, b in zip(_m["chains"], _m["bstart"][:-1], _m["bstart"][1:]) if b > a]
NE = 256


def replay(S, fd, sg, nf, hm):
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    if nf == 0:
        return serve
    for s, e in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            v = S[k].astype(np.float32)
            v = np.where(want, v * np.float32(1 + hm), v)
            if np.maximum(S[k], 0).sum() <= 0:
                continue
            nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
            want = nw
    return serve


def job(L):
    d = J.load("sm120tf", L)
    Sf = {f: np.load(f"{SC}/jF{f}_sm120tf/L{L}.npy", mmap_mode="r") for f in (1, 2, 3)}
    fdall = [int(e) for e in J.FDEF[L]]
    bs = d["bsal"].astype(np.float64); bc = d["bcnt"].astype(np.float64)
    out = {}
    for H in HS:
        fd = np.zeros(NE, bool); fd[fdall[:H]] = True
        r = dict(num=0.0, den=0.0, snum=0.0, sden=0.0, cs=0.0, cn=0)
        for t, (s, e) in enumerate(d["sg"]):
            S = np.asarray(Sf[FOLD[TASKS[t]]][s:e], np.float32)
            sv = replay(S, fd, [(0, e - s)], H, HM)
            r["num"] += float((bs[s:e] * sv).sum()); r["den"] += float(bs[s:e].sum())
            r["snum"] += float((bc[s:e] * sv).sum()); r["sden"] += float(bc[s:e].sum())
            r["cs"] += float((sv[1:] & ~sv[:-1]).sum()); r["cn"] += e - s - 1
        out[H] = r
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        R = dict(p.map(job, J.LAYERS))
    res = {}
    for H in HS:
        per = [R[L][H] for L in J.LAYERS]
        res[H] = dict(sal_pooled=sum(p["num"] for p in per) / sum(p["den"] for p in per),
                      sal_lmean=float(np.mean([p["num"] / p["den"] for p in per])),
                      slot_share=sum(p["snum"] for p in per) / sum(p["sden"] for p in per),
                      churn=float(np.mean([p["cs"] / max(p["cn"], 1) for p in per])),
                      per_layer_sal={L: R[L][H]["num"] / R[L][H]["den"] for L in J.LAYERS})
        print(f"H={H:3d} sal_pooled {res[H]['sal_pooled']:.4f} lmean {res[H]['sal_lmean']:.4f} "
              f"slot {res[H]['slot_share']:.4f} churn {res[H]['churn']:.2f}", flush=True)
    os.makedirs("/tmp/nestquant/35-nq15", exist_ok=True)
    json.dump(dict(hm=HM, corpus="sm120tf (6 tasks, fold out-of-sample jF1-3)", res=res),
              open(os.environ.get("OUTJ", "/tmp/nestquant/35-nq15/parta_salhot.json"), "w"), indent=1)
