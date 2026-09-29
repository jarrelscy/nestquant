#!/usr/bin/env python3
"""T32 gap decomposition: v2 GBDT (sync, band all) vs the salience oracle, all-slot hot % (26 fixed + 51 floating) of
routed salience (and routes), mean over layers.  Oracle at a decision made at the end of block k scores experts by the
TRUE salience of blocks k+1 .. k+n (the blocks it will serve); GBDT scores = streaming/gbdt_v2sal_p64.txt on band-all
rows.  Generic replay: refresh every r blocks, lag (0 sync / 1 next_refresh), hysteresis hm, optional swap cap,
optional candidate pool restriction.
  decomp.py CORPUS [NPROC]  -> $OUT/decomp_CORPUS.json"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MODEL = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
fixed, fdef = T.serve_sets()
NBC = T.CHAIN * T.SEQ // T.G
NF = 51


def replay(S, fx, fd, r=1, lag=0, hm=0.0, cap=None, pool=None):
    """S [nb, NE] score at decision end-of-block k -> serve [nb, NE] floating bool (fixed excluded)."""
    nb = S.shape[0]
    serve = np.zeros((nb, T.NE), bool)
    for c0 in range(0, nb, NBC):
        want = fd.copy()
        for k in range(c0, min(c0 + NBC, nb)):
            serve[k] = want
            if (k + 1 - c0) % r or k - c0 < lag:
                continue
            v = np.where(fx, -np.inf, S[k - lag]).astype(np.float64)
            if pool is not None:
                v = np.where(pool[k - lag], v, -np.inf)
            if hm:
                v = np.where(want, v * (1 + hm), v)
            order = np.argsort(-v, kind="stable")
            nw = np.zeros(T.NE, bool); nw[order[:NF]] = True
            nw &= np.isfinite(v)
            if cap is not None:
                new = np.nonzero(nw & ~want)[0]
                if len(new) > cap:
                    keep_new = new[np.argsort(-v[new], kind="stable")[:cap]]
                    inc = np.nonzero(want)[0]
                    keep_inc = inc[np.argsort(-v[inc], kind="stable")[:NF - cap]]
                    nw = np.zeros(T.NE, bool); nw[keep_new] = True; nw[keep_inc] = True
            want = nw & ~fx
    return serve


def fut(M, n):
    """sum of M over blocks k+1 .. k+n within the chain (truncated at chain end)."""
    out = np.zeros(M.shape)
    for c0 in range(0, M.shape[0], NBC):
        s = M[c0:c0 + NBC]
        cs = np.vstack([np.zeros((1, T.NE)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        out[c0:c0 + NBC] = cs[np.minimum(k + 1 + n, s.shape[0])] - cs[k + 1]
    return out


def past(M, n):
    """sum of M over blocks k-n+1 .. k within the chain."""
    out = np.zeros(M.shape)
    for c0 in range(0, M.shape[0], NBC):
        s = M[c0:c0 + NBC]
        cs = np.vstack([np.zeros((1, T.NE)), np.cumsum(s, 0)])
        k = np.arange(s.shape[0])
        out[c0:c0 + NBC] = cs[k + 1] - cs[np.maximum(k + 1 - n, 0)]
    return out


def topk_mask(S, fx, K):
    v = np.where(fx[None], -np.inf, S)
    o = np.argsort(-v, 1, kind="stable")[:, :K]
    m = np.zeros(S.shape, bool); np.put_along_axis(m, o, True, 1)
    return m


def job(L):
    import lightgbm as lgb
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    dflt = np.load(f"{T.OUT}/rows/{corpus}/L{L}.npz")
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    fx = np.zeros(T.NE, bool); fx[fixed[L]] = True
    fd = np.zeros(T.NE, bool); fd[[e for e in fdef[L] if e not in set(fixed[L])][:NF]] = True
    b = lgb.Booster(model_file=MODEL)
    G = T.score_blocks(b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1),
                       d["cand"], d["top"], d["e256"]).astype(np.float64)
    O = {n: fut(bs, n) for n in (1, 2, 4, 8, 16)}
    arms = {}
    # reference points
    arms["gbdt_sync_hm"] = replay(G, fx, fd, hm=0.5)
    arms["gbdt_sync_nohm"] = replay(G, fx, fd)
    arms["gbdt_lag1_hm"] = replay(G, fx, fd, lag=1, hm=0.5)
    # oracle horizon (sync, r=16 tokens, no hysteresis); + pure-history "oracle" (true past salience)
    for n in (1, 2, 4, 8, 16):
        arms[f"orc_h{16 * n}"] = replay(O[n], fx, fd)
    for n in (1, 4, 16):
        arms[f"past_h{16 * n}"] = replay(past(bs, n), fx, fd)
    # local-rate oracles: true salience in a window centred on the served blocks (rate knowledge, less realisation
    # noise of the specific next tokens): past 128 + next 128 / past 512 + next 512
    arms["orc_c256"] = replay(past(bs, 8) + O[8], fx, fd)
    arms["orc_c1024"] = replay(past(bs, 32) + fut(bs, 32), fx, fd)
    arms["orc_h64_hm_cap3"] = replay(O[4], fx, fd, hm=0.5, cap=3)
    # refresh timing (decision every r blocks; oracle window = max(r, 4) blocks ahead)
    for r in (2, 4, 8):
        arms[f"orc_r{16 * r}"] = replay(O[max(r, 4)] if r <= 4 else O[r], fx, fd, r=r)
        arms[f"gbdt_r{16 * r}_hm"] = replay(G, fx, fd, r=r, hm=0.5)
    arms["orc_lag1"] = replay(O[4], fx, fd, lag=1)                  # decided one block early (stale)
    # hysteresis / churn cap on the oracle
    arms["orc_hm"] = replay(O[4], fx, fd, hm=0.5)
    arms["orc_cap3"] = replay(O[4], fx, fd, cap=3)
    arms["orc_hm_cap3"] = replay(O[4], fx, fd, hm=0.5, cap=3)
    # ranking vs recall (sync, no hysteresis)
    for K in (64, 102, 153):
        arms[f"gpool{K}_orcorder"] = replay(O[4], fx, fd, pool=topk_mask(G, fx, K))
    for K in (64, 102):
        arms[f"opool{K}_gorder"] = replay(G, fx, fd, pool=topk_mask(O[4], fx, K))
    # metrics
    hot = {}
    for n, sv in arms.items():
        h = sv | fx
        hot[n] = dict(sal=float((bs * h).sum() / bs.sum()), cnt=float((bc * h).sum() / bc.sum()),
                      churn=float((sv[1:] & ~sv[:-1]).sum(1).mean()))
    # band 20-120: oracle-set salience outside what default-band GBDT can score (non-fixed EMA256 rank >= 121;
    # ranks < 20 are force-served by gbdt_old)
    cand = dflt["cand"].astype(np.int64); top = dflt["top"].astype(np.int64)
    inb = np.zeros(bs.shape, bool)
    np.put_along_axis(inb, cand, True, 1); np.put_along_axis(inb, top, True, 1)
    inb = np.vstack([fd[None] | True, inb[:-1]])            # decision at end of k-1 -> serves k
    so = arms["orc_h64"]
    extra = dict(orc_outside_band_share=float((bs * so * ~inb).sum() / max((bs * so).sum(), 1e-30)))
    # surprise: non-fixed (block, expert) with no hit in the previous 256 tokens (16 blocks) of the chain
    ph = np.vstack([np.zeros((1, T.NE)), past(bc, 16)[:-1]])
    ph[::NBC] = 0
    sur = (ph == 0) & ~fx[None]
    g, o = arms["gbdt_sync_hm"], so
    extra.update(surprise_share=float((bs * sur).sum() / bs.sum()),
                 surprise_orc=float((bs * sur * o).sum() / bs.sum()), surprise_gbdt=float((bs * sur * g).sum() / bs.sum()))
    # heavy tail: per-block gap (oracle - gbdt covered salience, layer-share units)
    gap_b = (bs * (o.astype(float) - g.astype(float))).sum(1) / bs.sum()
    tot_b = bs.sum(1)
    n1 = max(1, len(tot_b) // 100)
    extra.update(gap=float(gap_b.sum()), gap_top1pct_by_sal=float(gap_b[np.argsort(-tot_b)[:n1]].sum()),
                 gap_top1pct_by_gap=float(np.sort(gap_b)[::-1][:n1].sum()),
                 gap_top10pct_by_gap=float(np.sort(gap_b)[::-1][:10 * n1].sum()),
                 sal_top1pct_blocks=float(np.sort(tot_b)[::-1][:n1].sum() / tot_b.sum()))
    return L, hot, extra


if __name__ == "__main__":
    with Pool(NPROC) as p:
        res = sorted(p.map(job, T.LAYERS), key=lambda r: r[0])
    bands = {"all": T.LAYERS, "L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}
    idx = {L: i for i, L in enumerate(T.LAYERS)}
    names = list(res[0][1])
    summ = {n: {b: [float(np.mean([res[idx[L]][1][n][k] for L in r])) for k in ("sal", "cnt")]
                for b, r in bands.items()} for n in names}
    for n in names:
        summ[n]["churn"] = float(np.mean([res[i][1][n]["churn"] for i in range(len(res))]))
    ex = {k: {b: float(np.mean([res[idx[L]][2][k] for L in r])) for b, r in bands.items()} for k in res[0][2]}
    for n in names:
        print(f"{n:22s} " + "  ".join(f"{b} {v[0] * 100:5.1f}/{v[1] * 100:5.1f}" for b, v in summ[n].items()
                                      if b != "churn") + f"  churn {summ[n]['churn']:.2f}")
    for k, v in ex.items():
        print(f"{k:24s} " + "  ".join(f"{b} {x:.4f}" for b, x in v.items()))
    json.dump({"corpus": corpus, "summary": summ, "extra": ex,
               "per_layer": {L: {"hot": h, "extra": e} for L, h, e in res}},
              open(f"{T.OUT}/decomp_{corpus}.json", "w"), indent=1)
