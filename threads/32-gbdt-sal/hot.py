#!/usr/bin/env python3
"""T32 all-slot hot fraction: share of ALL routed slots (fixed + floating) that land on a level-4 expert, and the
salience-weighted share (sum w^2|x|^2 on hot routes / all routes), per layer L3-77, for the serve's policies.
Idealised executor (the target set is level 4 as soon as chosen: no byte budget / slots / in-flight delay), chains of
4x2048 tokens (= request; state reset, start from floating_default).
  hot.py CORPUS  -> $OUT/hot_CORPUS.json + table on stdout"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402
from sim import oracle_serve  # noqa: E402

corpus = sys.argv[1]
fixed, fdef = T.serve_sets()
OLD = "/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt"
V2 = f"{T.OUT}/models/v2_sal_tweedie1.5.txt"
NBC = T.CHAIN * T.SEQ // T.G


def ema_serve(ids, fx, fd, half_life=512, refresh=16):
    """scheduler.Scheduler predictor='ema': per-token score = score * a + counts, every `refresh` tokens the top-51
    non-fixed by score become the target (layers with no counts yet keep floating_default); applies from the next token."""
    a = 0.5 ** (1 / half_life)
    Tn = ids.shape[0]
    nb = Tn // T.G
    wj = a ** (T.G - 1 - (np.arange(Tn) % T.G))                 # decay to the block end
    b = (np.arange(Tn) // T.G)[:, None].repeat(8, 1)
    wc = np.bincount((b * T.NE + ids.astype(np.int64)).ravel(), weights=np.repeat(wj, 8),
                     minlength=nb * T.NE).reshape(nb, T.NE)
    serve = np.zeros((nb, T.NE), bool)
    rb = refresh // T.G
    for c0 in range(0, nb, NBC):
        s = np.zeros(T.NE); want = fd.copy()
        for k in range(c0, min(c0 + NBC, nb)):
            serve[k] = want
            s = s * a ** T.G + wc[k]
            if (k - c0 + 1) % rb == 0 and s.sum() > 0:
                sc = np.where(fx, -np.inf, s)
                want = np.zeros(T.NE, bool); want[np.argsort(-sc, kind="stable")[:51]] = True
    return serve


def free_oracle(M, n=77):
    out = np.zeros(M.shape, bool)
    for c0 in range(0, M.shape[0], NBC):
        s = M[c0:c0 + NBC]
        cs = np.vstack([np.zeros((1, T.NE)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        sc = cs[np.minimum(k + T.H // T.G, s.shape[0])] - cs[k]
        np.put_along_axis(out[c0:c0 + NBC], np.argsort(-sc, 1, kind="stable")[:, :n], True, 1)
    return out


def job(L):
    import lightgbm as lgb
    fx = np.zeros(T.NE, bool); fx[fixed[L]] = True
    fd = np.zeros(T.NE, bool); fd[[e for e in fdef[L] if not fx[e]][:51]] = True
    d = np.load(f"{T.OUT}/rows/{corpus}/L{L}.npz")
    da = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    X2 = np.load(f"{T.OUT}/rows_v2/{corpus}/L{L}.npz")["X2"]
    ids = T.load_layer(L, corpus)[0]
    old = lgb.Booster(model_file=OLD)
    p_old = old.predict(d["X"].reshape(-1, 5), num_threads=1)
    S_old = T.score_blocks(p_old, d["cand"], d["top"], d["e256"])
    S_mps = T.score_blocks(p_old * X2[..., 3].ravel(), d["cand"], d["top"], d["e256"])
    v2 = lgb.Booster(model_file=V2)
    X2a = np.load(f"{T.OUT}/rows_v2_bandall/{corpus}/L{L}.npz")["X2"]
    p_v2 = v2.predict(np.concatenate([da["X"], X2a], -1).reshape(-1, 9), num_threads=1)
    S_v2 = T.score_blocks(p_v2, da["cand"], da["top"], da["e256"])
    nb = bc.shape[0]
    sets = {
        "ema_r16": ema_serve(ids, fx, fd, refresh=16),
        "ema_r64": ema_serve(ids, fx, fd, refresh=64),
        "gbdt_nr": T.sim_layer(S_old, fixed[L], fdef[L], lag=1),
        "gbdt_sync": T.sim_layer(S_old, fixed[L], fdef[L], lag=0),
        "gbdt_x_mps_nr": T.sim_layer(S_mps, fixed[L], fdef[L], lag=1),
        "gbdt_x_mps_sync": T.sim_layer(S_mps, fixed[L], fdef[L], lag=0),
        "v2sal_sync_bandall": T.sim_layer(S_v2, fixed[L], fdef[L], lag=0),
        "static10_fixed26": np.zeros((nb, T.NE), bool),
        "static30_fixed26+fdef51": np.repeat(fd[None], nb, 0),
        "orc_count_26+51": oracle_serve(bc, fx, NBC),
        "orc_sal_26+51": oracle_serve(bs, fx, NBC),
        "orc_count_free77": free_oracle(bc),
        "orc_sal_free77": free_oracle(bs),
    }
    out = {}
    for n, sv in sets.items():
        hot = sv | fx if "free77" not in n else sv
        out[n] = dict(cnt=float((bc * hot).sum() / bc.sum()), sal=float((bs * hot).sum() / bs.sum()),
                      nhot=float(hot.sum(1).mean()))
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        res = dict(p.map(job, T.LAYERS))
    names = list(res[T.LAYERS[0]])
    bands = {"all L3-77": T.LAYERS, "L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}
    summ = {n: {b: {k: float(np.mean([res[L][n][k] for L in Ls])) for k in ("cnt", "sal")} for b, Ls in bands.items()}
            for n in names}
    print(f"{corpus}: % of ALL routed slots on a level-4 expert  [routes / salience-weighted]")
    print(f"{'arm':26s}" + "".join(f"{b:>18s}" for b in bands))
    for n in names:
        print(f"{n:26s}" + "".join(f"   {100 * summ[n][b]['cnt']:5.1f} / {100 * summ[n][b]['sal']:5.1f}" for b in bands))
    json.dump({"corpus": corpus, "summary": summ, "per_layer": res}, open(f"{T.OUT}/hot_{corpus}.json", "w"), indent=1)
