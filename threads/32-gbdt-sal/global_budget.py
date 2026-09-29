#!/usr/bin/env python3
"""T32 global floating budget (offline, sync-mode v2 GBDT, band all).  Same total 75*51 floating slots (+ the 26 fixed
per layer), allocated across layers each refresh (16-token block, lag 0) by a cross-layer score instead of 51 per layer.
  global_budget.py CORPUS [MODEL] [NPROC]  -> $OUT/global_CORPUS.json
Scores (p = model prediction = next-64 salience / m_L, m_L = calib-fit mean salience per routed slot (train meta)):
  uni        51 per layer (= sim.py v2 sync band all)
  a_raw      p * m_L                       raw predicted salience (w^2 |x|^2, units differ per layer)
  b_div      p * s_L                       s_L = T18 per-layer rel-div increment, nq2 (passA1) - nq4 (passB), 3-layer
                                           running mean, floored at 5e-4: divergence growth per unit of layer salience
  b_drel     p * m_L * drel_e              T31 delta/G (relative error-energy drop 2->4) per expert
  b_delta    p * m_L * delta_e             T31 delta (absolute error-energy drop per unit input energy) = SM120 benefit
  c_*        b_* with per-layer floor 24 / cap 128
Hysteresis as the serve (incumbent scores x 1.5), chains start from floating_default (51/layer).
Metrics (non-fixed salience covered by floating): U = mean over layers (uniform), S = s_L-weighted mean over layers,
D = pooled delta_e * salience covered (absolute error energy), R = pooled raw salience; + all-slot hot % (routes, sal)
and the per-layer mean slot count."""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NF, HM, FLOOR, CAP = 51, 0.5, 24, 128
fixed, fdef = T.serve_sets()
NL = len(T.LAYERS)


def sens():
    a = json.load(open("/tmp/nestquant/18-e2e/results/passA1.json"))["per_layer"]["nq2"]["rel_div_increment"]
    b = json.load(open("/tmp/nestquant/18-e2e/results/passB.json"))["per_layer"]["nq4"]["rel_div_increment"]
    s = np.array([a[str(L)] - b[str(L)] for L in T.LAYERS])
    s = np.convolve(np.pad(s, 1, mode="edge"), np.ones(3) / 3, "valid")
    return np.maximum(s, 5e-4)


def job(L):
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    b = lgb.Booster(model_file=MODEL)
    Xm = T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d)
    S = T.score_blocks(b.predict(Xm, num_threads=1), d["cand"], d["top"], d["e256"])
    uni = T.sim_layer(S, fixed[L], fdef[L], nf=NF, hm=HM, lag=0)
    return L, S, d["bcnt"].astype(np.float32), d["bsal"].astype(np.float32), uni


def alloc(v, floor, cap, budget):
    """v [NL, NE] (fixed = -inf) -> bool [NL, NE]: per-layer top-floor, then global top by v under cap."""
    out = np.zeros(v.shape, bool)
    order = np.argsort(-v, 1, kind="stable")
    if floor:
        np.put_along_axis(out, order[:, :floor], True, 1)
    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(NE)[None], 1)
    ok = (~out) & (rank < cap) & np.isfinite(v)
    need = budget - out.sum()
    fl = np.where(ok, v, -np.inf).ravel()
    pick = np.argpartition(-fl, need)[:need]
    out.ravel()[pick] = True
    return out


def sim_global(S, w, fx, fd, floor, cap, nbc):
    """S [NL, nb, NE] predictions, w [NL, NE] cross-layer multiplier -> serve [NL, nb, NE] bool (sync, hysteresis)."""
    nb = S.shape[1]
    serve = np.zeros(S.shape, bool)
    for c0 in range(0, nb, nbc):
        want = fd.copy()
        for k in range(c0, min(c0 + nbc, nb)):
            serve[:, k] = want
            v = np.where(fx, -np.inf, S[:, k] * w)
            v = np.where(want, v * (1 + HM), v)
            want = alloc(v, floor, cap, NL * NF) & ~fx
    return serve


NE = T.NE

if __name__ == "__main__":
    with Pool(NPROC) as p:
        res = sorted(p.map(job, T.LAYERS), key=lambda r: r[0])
    S = np.stack([r[1] for r in res]); BC = np.stack([r[2] for r in res]).astype(np.float64)
    BS = np.stack([r[3] for r in res]).astype(np.float64); UNI = np.stack([r[4] for r in res])
    fx = np.zeros((NL, NE), bool); fd = np.zeros((NL, NE), bool)
    for i, L in enumerate(T.LAYERS):
        fx[i, fixed[L]] = True
        fd[i, [e for e in fdef[L] if e not in set(fixed[L])][:NF]] = True
    meta = MODEL + ".meta.json" if os.path.exists(MODEL + ".meta.json") else \
        "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"      # = streaming/gbdt_v2sal_p64.txt
    mL = json.load(open(meta))["sal_norm_mL"]
    mL = np.array([mL[str(L)] for L in T.LAYERS])[:, None]
    t31 = np.load("/tmp/nestquant/31-delta/delta_table.npz")
    assert list(t31["layers"]) == T.LAYERS
    s = sens()[:, None]
    W = {"uni_check": (np.ones((NL, 1)), NF, NF), "a_raw": (mL, 0, NE),
         "b_div": (s, 0, NE), "b_drel": (mL * t31["drel"], 0, NE), "b_delta": (mL * t31["delta"], 0, NE),
         "c_raw": (mL, FLOOR, CAP), "c_div": (s, FLOOR, CAP), "c_drel": (mL * t31["drel"], FLOOR, CAP),
         "c_delta": (mL * t31["delta"], FLOOR, CAP)}
    nbc = T.CHAIN * T.SEQ // T.G
    nf = ~fx[:, None, :]
    dl = t31["delta"][:, None, :]
    out = {"corpus": corpus, "model": MODEL, "sens_s_L": s.ravel().tolist(), "floor": FLOOR, "cap": CAP, "arms": {}}

    def metrics(sv):
        cov = (BS * sv).sum((1, 2)) / (BS * nf).sum((1, 2))
        allc = (BC * (sv | fx[:, None])).sum() / BC.sum()
        alls = (BS * (sv | fx[:, None])).sum((1, 2)) / BS.sum((1, 2))
        return dict(U=float(cov.mean()), S=float((cov * s.ravel()).sum() / s.sum()),
                    D=float((BS * dl * sv).sum() / (BS * dl * nf).sum()),
                    R=float((BS * sv).sum() / (BS * nf).sum()),
                    hot_routes=float(allc), hot_sal_U=float(alls.mean()),
                    hot_sal_S=float((alls * s.ravel()).sum() / s.sum()),
                    slots_mean=sv.sum(2).mean(1).round(1).tolist(), cov_L=cov.round(4).tolist())
    out["arms"]["uni"] = metrics(UNI)
    for name, (w, fl, cp) in W.items():
        sv = sim_global(S, w, fx, fd, fl, cp, nbc)
        if name == "uni_check":
            out["uni_check_mismatch_blocks"] = int((sv != UNI).any((0, 2)).sum())
            continue
        out["arms"][name] = metrics(sv)
        print(name, {k: round(v, 4) for k, v in out["arms"][name].items() if not isinstance(v, list)}, flush=True)
    print("uni", {k: round(v, 4) for k, v in out["arms"]["uni"].items() if not isinstance(v, list)},
          "uni_check mismatched blocks", out["uni_check_mismatch_blocks"])
    json.dump(out, open(f"{T.OUT}/global_{corpus}.json", "w"), indent=1)
