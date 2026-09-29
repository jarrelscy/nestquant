"""evalp.py CORPUS name=SPEC ...  (CORPUS: glm52-heldout | calib-val | sm120tf)
SPEC: path to a LightGBM model (v2 = serve model) or sa:FEAT[*mps] (standalone score).  HMS env = hm sweep.
all-slot sal-hot %, churn (global, = hot_eval), sync lag 0.  SAVE=name -> scores_CORPUS/L{L}.npy (f16 [nb,256])."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import plib as P, pfeat as Q, feats as FE
corpus = sys.argv[1]
specs = dict(a.split("=", 1) for a in sys.argv[2:])
HMS = [float(x) for x in os.environ.get("HMS", "0.5").split(",")]
SAVE = os.environ.get("SAVE", "")


def job(L):
    import lightgbm as lgb
    src_c = "calib-fit" if corpus == "calib-val" else corpus
    D = P.load(src_c, L)
    segs = D["segs"]
    if corpus == "calib-val":
        segs = [sg for i, sg in enumerate(segs) if Q.val_chain(i)]
    ff = f"{P.OUT}/private/feat/{src_c}/L{L}.npz"
    if os.path.exists(ff):
        z = np.load(ff); PFd = {k: z[k].astype(np.float32) for k in z.files}
    else:
        PFd = FE.feats(D, L)
    src = dict(D["F"]); src.update(PFd)
    rows = np.concatenate([np.arange(s, e) for s, e in segs])
    lsegs, o = [], 0
    for s, e in segs:
        lsegs.append((o, o + e - s)); o += e - s
    bs, bc = D["bsal"][rows], D["bcnt"][rows]
    res = {}
    for n, sp in specs.items():
        if sp.startswith("sa:"):
            f = sp[3:]
            mps = f.endswith("*mps"); f = f.removesuffix("*mps")
            S = src[f][rows].copy()
            if f.startswith("kf"):
                S = np.exp(S)
            if mps:
                S = S * src["mps128"][rows]
            S = S.astype(np.float32)
        else:
            b = lgb.Booster(model_file=P.V2 if sp == "v2" else sp)
            names = b.feature_name()
            X = np.stack([src[c][rows] for c in names], -1).reshape(-1, len(names))
            S = b.predict(X, num_threads=1).reshape(len(rows), P.NE).astype(np.float32)
        if n == SAVE:
            os.makedirs(f"{P.OUT}/scores_{corpus}", exist_ok=True)
            np.save(f"{P.OUT}/scores_{corpus}/L{L}.npy", S.astype(np.float16))
        res[n] = {str(hm): P.metric(P.replay(S, L, lsegs, hm=hm), L, bs, bc, lsegs) for hm in HMS}
    return L, res


if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        r = dict(p.map(job, P.T.LAYERS))
    f = f"{P.OUT}/eval_{corpus}.json"
    allr = json.load(open(f)) if os.path.exists(f) else {}
    for n in specs:
        for hm in HMS:
            s = {k: float(np.mean([r[L][n][str(hm)][k] for L in r])) for k in ("sal", "cnt", "churn_g", "churn")}
            allr.setdefault(n, {})[str(hm)] = dict(s, spec=specs[n], per_layer=[r[L][n][str(hm)]["sal"] for L in P.T.LAYERS])
            print(f"{corpus} {n:16s} hm {hm:4.2f}  sal-hot {s['sal']*100:6.2f}  churn {s['churn_g']:5.2f} (in-chain {s['churn']:5.2f})  routes {s['cnt']*100:6.2f}", flush=True)
    json.dump(allr, open(f, "w"), indent=1)
    print(f"{time.time()-t0:.0f}s")
