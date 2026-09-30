"""streaming GBDTPredictorV2 (drop-in model file) vs offline scalelib score matrices on sm120tf decode blocks.
Serve today passes no token ids -> stuck flag; offline feats(stuck=True) is the matching path.  K0=1 for the k=0 layout.
  parity_model.py MODEL"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
sys.path.insert(0, "/home/coder/git/nestquant/streaming")
from gbdt_predictor_v2 import GBDTPredictorV2
import lightgbm as lgb
MODEL = sys.argv[1]
b = lgb.Booster(model_file=MODEL)
ND = 96; worst = 0.0; nset = nsame = 0
for L in (3, 27, 51, 75):
    D0 = S.load(os.environ.get("STREAM", "sm120tf"), L)
    for ci, (s, e) in enumerate(D0["sg"][:3]):
        D = {k: D0[k][s + 500:s + 500 + ND] for k in ("bc", "bs", "bca", "nans", "segl")}; D["sg"] = [(0, ND)]
        F, _ = S.feats(D, stuck=os.environ.get("STUCK", "1") == "1")
        So = S.predict_S(b, F, L)
        p = GBDTPredictorV2([L], {L: S.FIXED[L]}, model_path=MODEL, n_float=S.NF, mode="sync", num_threads=1, rlo=0, rhi=256)
        Son = []
        for k in range(ND):
            p.step(D["bc"][k][None], ntok=16, sal=D["bs"][k][None])
            Son.append(p.S[0].copy())
        Son = np.array(Son)
        fx = np.zeros(S.NE, bool); fx[S.FIXED[L]] = True
        d = np.abs(Son - So)[:, ~fx].max() / np.abs(So[:, ~fx]).max()
        worst = max(worst, d)
        top = lambda A: set(np.argsort(-np.where(fx, -np.inf, A), kind="stable")[:S.NF])
        for k in range(ND):
            nset += 1; nsame += top(Son[k]) == top(So[k])
        p.close()
print(f"{os.path.basename(MODEL)} K0={os.environ.get('K0','0')} max rel |S_stream - S_offline| {worst:.2e}; top-{S.NF} identical in {nsame}/{nset}")
