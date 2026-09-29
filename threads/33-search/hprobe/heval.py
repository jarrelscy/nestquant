#!/usr/bin/env python3
"""T33c hprobe eval: all-slot salience-hot % (26 fixed + 51 floating), sync lag 0, refresh 16, band all, chains 4x2048,
floating_default at chain start; churn = new floating experts per layer per refresh.  Same arithmetic as
threads/32-gbdt-sal/hot_eval.py (t32lib.sim_layer / score_blocks / feature_matrix), but score matrices are generic:
  score_fn(L) -> S [nb, 256] float32.   Outputs only to /tmp/nestquant/33-search/hprobe."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np  # noqa: E402
import t32lib as T  # noqa: E402

FIXED, FDEF = T.serve_sets()
HP = "/tmp/nestquant/33-search/hprobe"
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"


def rows(corpus, L):
    return np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")


def v2_scores(corpus, L, d=None, model=V2):
    import lightgbm as lgb
    d = rows(corpus, L) if d is None else d
    b = lgb.Booster(model_file=model)
    return T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                          d["cand"], d["top"], d["e256"])


def eval_S(S, L, bsal, bcnt, hm=0.5, mask=None):
    """-> dict(salnum, salden, cnum, cden, churn_sum, churn_n) (mask: bool [nb] blocks included in the sums)."""
    sv = T.sim_layer(S, FIXED[L], FDEF[L], nf=51, hm=hm, lag=0)
    new = (sv[1:] & ~sv[:-1]).sum(1).astype(np.float64)
    nbc = T.CHAIN * T.SEQ // T.G
    k = np.arange(1, sv.shape[0])
    ok = (k % nbc) != 0                                  # hot_eval counts chain-boundary transitions too; keep both
    sv[:, FIXED[L]] = True
    m = np.ones(sv.shape[0], bool) if mask is None else mask
    return dict(sn=float((bsal * sv)[m].sum()), sd=float(bsal[m].sum()), cn=float((bcnt * sv)[m].sum()),
                cd=float(bcnt[m].sum()), ch=float(new[m[1:]].sum()), chn=float(m[1:].sum()),
                ch_in=float(new[m[1:] & ok].sum()), chn_in=float((m[1:] & ok).sum()))


def summarise(res):
    """res {L: dict} -> sal-hot (mean over layers), cnt-hot, churn (hot_eval definition: mean incl. boundaries)."""
    Ls = sorted(res)
    return dict(sal=100 * float(np.mean([res[L]["sn"] / res[L]["sd"] for L in Ls])),
                cnt=100 * float(np.mean([res[L]["cn"] / res[L]["cd"] for L in Ls])),
                churn=float(np.mean([res[L]["ch"] / res[L]["chn"] for L in Ls])),
                churn_in=float(np.mean([res[L]["ch_in"] / res[L]["chn_in"] for L in Ls])))
