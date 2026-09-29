"""eval models on a stream, per-block metrics saved for post-hoc splits.
  evalall.py STREAM TAG name=path[@stuck] ... [--hm 0.3,0.5,0.7] [--oracle] [--chains i,j]
-> private/res/STREAM/TAG/L{L}.npz  keys 'name|hm' -> [nb,5] (covered sal, total sal, covered hits, hits, churn)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
stream, tag = args[0], args[1]
models = dict(x.split("=", 1) for x in args[2:])
HMS = [float(h) for h in opt.get("--hm", "0.5").split(",")]
out = f"{S.OUT}/private/res/{stream}/{tag}"
os.makedirs(out, exist_ok=True)

def job(L):
    import lightgbm as lgb
    if os.path.exists(f"{out}/L{L}.npz"):
        return L
    D = S.load(stream, L)
    F, e256 = S.feats(D)
    need_stuck = any(p.endswith("@stuck") for p in models.values())
    Fst = S.mem_state(D["bc"], D["bca"], D["nans"], D["segl"], D["sg"], stuck=True) if need_stuck else None
    res = {}
    for n, p in models.items():
        path, _, how = p.partition("@")
        b = lgb.Booster(model_file=path)
        FF = F
        if how == "stuck":
            FF = F.copy(); FF[..., 2] = Fst
        Sm = S.predict_S(b, FF, L)
        del FF
        for hm in HMS:
            res[f"{n}|{hm}"] = S.block_metrics(S.replay(Sm, L, D["sg"], hm=hm), D, L).astype(np.float32)
    if "--oracle" in opt:
        O = S.oracle_S(D, 4)
        res["orc64|0"] = S.block_metrics(S.replay(O, L, D["sg"], hm=0), D, L).astype(np.float32)
        res["orc64cap3|0.5"] = S.block_metrics(S.replay(O, L, D["sg"], hm=0.5, cap=3), D, L).astype(np.float32)
    np.savez(f"{out}/L{L}.npz", **res)
    return L

if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        for L in p.imap_unordered(job, S.LAYERS):
            pass
    R = {L: dict(np.load(f"{out}/L{L}.npz")) for L in S.LAYERS}
    for k in R[S.LAYERS[0]]:
        s = S.summarize({L: R[L][k] for L in S.LAYERS})
        print(f"{stream} {tag} {k:28s} sal-hot {s['sal']:6.2f} hits {s['cnt']:6.2f} churn {s['churn']:5.2f}", flush=True)
