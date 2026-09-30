"""streaming GBDTPredictorD0 vs offline (scalelib.feats base 9 + T33g glib.dyn_feats dyn0 9) at k=0.
Per stream, 4 layers x 3 chains x ND decode blocks (fresh state at the slice start on both sides), both flag modes:
  stuck   step(counts, 16, sal=...) with no token ids  <-> offline feats(stuck=True)
  correct each block fed as a think step (THINK id) + an answer step (ETHINK id) so the predictor rebuilds the block's
          answer counts / answer-token count / last-token segment  <-> offline feats(stuck=False)
Checks: max |S_stream - S_offline| (all 256 experts), top-77 sets of the raw scores, and the hysteresis target
sequence (p.target(resident) chained, hm 0.6) vs scalelib.replay.
  K0=1 parity_d0.py MODEL [STREAM ...]"""
import os, sys
import numpy as np
assert os.environ.get("K0") == "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/gbdt")
import scalelib as S
import glib
from gbdt_predictor_d0 import GBDTPredictorD0, DYN0
from gbdt_predictor import THINK_ID, ETHINK_ID
import lightgbm as lgb

MODEL = sys.argv[1]
streams = sys.argv[2:] or ["sm120tf", "glm52-heldout"]
b = lgb.Booster(model_file=MODEL)
ND, HM = 96, 0.6
idx = [glib.DYN.index(n) for n in DYN0]
pri = np.zeros((S.NE, 13), np.float32); pri[:, 0] = 1          # dyn0 uses no priors (dummy for the unused columns)


def feed(p, D, k, correct):
    c, s = D["bc"][k], D["bs"][k]
    if not correct:
        return p.step(c[None], ntok=16, sal=s[None])
    ca, na, sl = D["bca"][k], int(D["nans"][k]), int(D["segl"][k])
    th = (c - ca, 16 - na, THINK_ID); an = (ca, na, ETHINK_ID)
    parts = [an, th] if sl == 0 else [th, an]                # the block's last token decides the segment
    z = np.zeros_like(s)
    p.step(parts[0][0][None], ntok=parts[0][1], token_ids=[parts[0][2]], sal=z[None])
    return p.step(parts[1][0][None], ntok=parts[1][1], token_ids=[parts[1][2]], sal=s[None])


for stream in streams:
    for mode in ("stuck", "correct"):
        worst = 0.0; nset = nsame = ntg = ntsame = 0; nb_ans = 0
        for L in (3, 27, 51, 75):
            D0 = S.load(stream, L)
            for (s0, e0) in D0["sg"][:3]:
                o = s0 + min(500, e0 - s0 - ND)
                D = {k: D0[k][o:o + ND] for k in ("bc", "bs", "bca", "nans", "segl")}; D["sg"] = [(0, ND)]
                nb_ans += int((D["nans"] > 0).sum())
                F, _ = S.feats(D, stuck=mode == "stuck")
                Dy = glib.dyn_feats(D["bc"], D["bs"], D["sg"], pri)[..., idx]
                So = S.predict_S(b, np.concatenate([F, Dy], -1), L)
                serve_o = S.replay(So, L, D["sg"], hm=HM)
                p = GBDTPredictorD0([L], {L: []}, model_path=MODEL, n_float=S.NF, hm=HM, mode="sync", num_threads=1,
                                    rlo=0, rhi=256)
                fd = np.zeros(S.NE, bool); fd[S.FDEF[L][:S.NF]] = True
                want = fd.copy()
                for k in range(ND):
                    ntg += 1; ntsame += bool((want == serve_o[k]).all())
                    assert feed(p, D, k, mode == "correct")
                    Sk = p.S[0]
                    worst = max(worst, float(np.abs(Sk - So[k]).max() / np.abs(So[k]).max()))
                    top = lambda A: set(np.argsort(-A, kind="stable")[:S.NF])  # noqa: E731
                    nset += 1; nsame += top(Sk) == top(So[k])
                    want = p.target(want[None])[0]
                p.close()
        print(f"{stream:14s} {mode:7s} {os.path.basename(MODEL)}: max rel |S_stream-S_off| {worst:.2e}; "
              f"top-{S.NF} identical {nsame}/{nset}; hm{HM} target sets identical {ntg and ntsame}/{ntg}; "
              f"blocks with answer tokens {nb_ans}", flush=True)
