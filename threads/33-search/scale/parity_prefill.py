"""offline (scalelib.feats + predict_S on the 'warmmean' synthetic stream) vs streaming GBDTPredictorV2P score
matrices on heldout chains: prompt = first 512 tokens (sums), decode blocks 32..95 of each chain."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
from gbdt_predictor_v2p import GBDTPredictorV2P
import lightgbm as lgb
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
b = lgb.Booster(model_file=V2)
P, ND = 32, 64
worst = 0.0; nset = 0; nsame = 0
for L in (3, 20, 45, 77):
    D = S.load("glm52-heldout", L)
    for ci, (s, e) in enumerate(D["sg"][:3]):
        m = {k: np.repeat(D[k][s:s + P].mean(0, keepdims=True), 16, 0) for k in ("bc", "bs", "bca", "nans", "segl")}
        dec = {k: D[k][s + P:s + P + ND] for k in m}
        D2 = {k: np.concatenate([m[k], dec[k]]) for k in m}; D2["sg"] = [(0, 16 + ND)]
        F, _ = S.feats(D2)
        Soff = S.predict_S(b, F, L)
        sv_off = S.replay(Soff, L, D2["sg"])
        p = GBDTPredictorV2P([L], {L: S.FIXED[L]}, model_path=V2, mode="sync", num_threads=1, rlo=0, rhi=256)
        p.prefill(D["bc"][s:s + P].sum(0)[None], D["bs"][s:s + P].sum(0)[None], P * 16, reset=True)
        Son = [p.S[0].copy()]
        for k in range(ND - 1):
            p.step(dec["bc"][k][None], ntok=16, sal=dec["bs"][k][None])
            Son.append(p.S[0].copy())
        Son = np.array(Son); So = Soff[15:15 + ND]
        fx = np.zeros(S.NE, bool); fx[S.FIXED[L]] = True
        d = np.abs(Son - So)[:, ~fx].max() / np.abs(So[:, ~fx]).max()
        worst = max(worst, d)
        top = lambda A: set(np.argsort(-np.where(fx, -np.inf, A), kind="stable")[:51])
        for k in range(ND):
            nset += 1; nsame += top(Son[k]) == top(So[k])
        p.close()
print(f"max rel |S_stream - S_offline| {worst:.2e}; top-51 identical in {nsame}/{nset} decisions")
