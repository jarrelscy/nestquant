"""g33eval.py CORPUS NAME=SPEC ... [--save NAME]   CORPUS: calib-val | glm52-heldout | sm120tf
SPEC: model.txt path (feature names from the booster; 'v2' column = v2 prediction; meta init=v2 -> stacked) or
      bands:A|B|C (per-band models L3-6 | L7-40 | L41-77), or v2 (T32 reference).  Scores all experts (band all).
-> hm sweep sal-hot / churn, matched-churn sal at 2.78 / 3.2; json merged into $W/eval_CORPUS.json"""
import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
import glib as g

corpus = sys.argv[1]
args = [x for x in sys.argv[2:] if not x.startswith("--")]
save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
models = dict(x.split("=", 1) for x in args if "=" in x)
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
HMS = tuple(float(x) for x in os.environ.get("HMS", "0,0.2,0.3,0.4,0.5,0.7,1.0").split(","))
PZ = dict(np.load(f"{g.W}/priors.npz"))
HMS0 = tuple(float(x) for x in os.environ.get("HMS0", "0.4,0.5,0.6,0.7,0.9").split(","))
CH = 400000   # rows per predict chunk


def resolve(spec, L):
    if spec == "v2":
        return V2
    if spec.startswith("bands:"):
        ps = spec[6:].split("|")
        return ps[0] if L <= 6 else ps[1] if L <= 40 else ps[2]
    return spec


def predict(bst, X, raw=False):
    out = np.empty(X.shape[0], np.float64)
    for s in range(0, X.shape[0], CH):
        out[s:s + CH] = bst.predict(X[s:s + CH], num_threads=int(os.environ.get("PT", "1")), raw_score=raw)
    return out


def job(L):
    li = g.T.LAYERS.index(L)
    if corpus.startswith("sm120"):
        nch = len(json.load(open(f"{g.T.OUT}/private/sm120/blk/{corpus}/meta.json"))["chains"])
        acc = None
        for ci in range(nch):
            b = g.full_feats(corpus, L, PZ["P_all"][li], chain=ci)
            if b["bcnt"].shape[0] == 0:
                continue
            o, _ = piece(b, L)
            del b
            if acc is None:
                acc = o
            else:
                for n in o:
                    for hm in o[n]:
                        for k in ("num", "den", "chs", "chn"):
                            acc[n][hm][k] += o[n][hm][k]
        for n in acc:
            for hm, v in acc[n].items():
                v["sal"] = v["num"] / v["den"]; v["churn"] = v["chs"] / v["chn"]
        return L, acc
    src = "calib-fit" if corpus == "calib-val" else corpus
    b = g.full_feats(src, L, PZ["P_all"][li])
    if corpus == "calib-val":
        s0 = b["segs"][g.NTRAIN_CH][0]
        for k in ("XA", "bcnt", "bsal", "ysal"):
            b[k] = b[k][s0:]
        b["segs"] = [(s - s0, e - s0) for s, e in b["segs"][g.NTRAIN_CH:]]
    out, S_save = piece(b, L)
    if S_save is not None:
        d = f"{g.W}/scores_{corpus}"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/L{L}.npy", S_save.astype(np.float16))
    return L, out


def piece(b, L):
    import lightgbm as lgb
    XA = b.pop("XA"); nb = XA.shape[0]
    XA = XA.reshape(nb * g.NE, -1)
    v2b = lgb.Booster(model_file=V2)
    v2raw = predict(v2b, XA[:, :9], raw=True)
    out, S_save = {}, None
    for n, spec in models.items():
        path = resolve(spec, L)
        bst = v2b if path == V2 else lgb.Booster(model_file=path)
        names = bst.feature_name()
        meta = json.load(open(path + ".meta.json")) if os.path.exists(path + ".meta.json") and path != V2 else {}
        cols = []
        for f in names:
            cols.append(np.exp(v2raw).astype(np.float32) if f == "v2" else XA[:, g.ALLX.index(f)])
        X = np.stack(cols, 1)
        r = predict(bst, X, raw=True)
        if meta.get("init") == "v2":
            r = r + v2raw
        S = np.exp(r).reshape(nb, g.NE).astype(np.float32)
        out[n] = g.metrics(S, L, b, hms=HMS)
        out[n + "@k0"] = g.metrics(S, L, b, hms=HMS0, layout="k0")
        if save == n:
            S_save = S
    return out, S_save


if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        res = dict(p.map(job, g.T.LAYERS))
    f = f"{g.W}/eval_{corpus}.json"
    allr = json.load(open(f)) if os.path.exists(f) else {}
    for n in [m + sfx for m in models for sfx in ("", "@k0")]:
        s = g.summarize({L: res[L][n] for L in g.T.LAYERS})
        m278, m32 = g.at_churn(s, 2.78), g.at_churn(s, 3.2)
        if n.endswith("@k0"):
            m278 = g.at_churn(s, 3.16)
        allr[n] = dict(spec=models[n.split("@")[0]], sweep={str(k): v for k, v in s.items()}, at278=m278, at32=m32,
                       per_layer={str(L): res[L][n][0.5]["sal"] for L in g.T.LAYERS})
        print(f"{n:22s} @2.78 {m278:6.2f}  @3.2 {m32:6.2f}  | " +
              "  ".join(f"hm{k:g}:{v['sal']:.2f}/{v['churn']:.2f}" for k, v in s.items()) +
              f"  | bands@0.5 " + " ".join(f"{bn} {s[0.5][bn]:.1f}" for bn in g.BANDS) if 0.5 in s else "", flush=True)
    json.dump(allr, open(f, "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f}s")
