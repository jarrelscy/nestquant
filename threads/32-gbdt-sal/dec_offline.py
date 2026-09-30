#!/usr/bin/env python3
"""T32 decode-text offline arms on the passT32Kd subset (private/corpora/sm120tfk: 6 SM120 tasks x 3 consecutive
all-decode windows = one chain per task), FP8 teacher-forced routing from private/trace_sm120 (sm120tf windows
src_win).  Same arms as the KLD pass: k26_v2 (manifest fixed-26, v2 sync band all, hm 0.5), k0 (no fixed, nf 77)
at hm 0.5 / 0.6 / 1.0, k6 (top-6 by calib salience fixed, nf 71, hm 0.55).  All-slot sal-hot / routes-hot / churn,
mean L3-77.  PRIVATE inputs; writes only aggregate numbers.   dec_offline.py  -> $OUT/dec_offline.json"""
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

O = T.OUT
P = f"{O}/private"
V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
mp = np.load(f"{P}/corpora/sm120tfk.map.npz")
SRCW, TASK = mp["src_win"], mp["task"]
TOK = np.load(f"{P}/corpora/sm120tfk.npy").astype(np.int64)
NW = len(SRCW)
CW = 3                                   # windows per chain (task)
NBC = CW * T.SEQ // T.G
fixed26, fdef = T.serve_sets()
MAN = {k: json.load(open(f"{O}/k{k}_manifest.json")) for k in (0, 6)}
ARMS = [("k26_v2", 26, 0.5), ("k0_hm06", 0, 0.6), ("k6_hm055", 6, 0.55), ("k0_hm05", 0, 0.5), ("k0_hm10", 0, 1.0)]


def seg_chain(tok):
    s = np.zeros(len(tok), np.int8)
    for c0 in range(0, len(tok), CW * T.SEQ):
        cur = 0
        for t in range(c0, c0 + CW * T.SEQ):
            nt = tok[t + 1] if t + 1 < len(tok) else -1
            cur = 0 if nt == T.THINK_ID else 1 if nt == T.ETHINK_ID else cur
            s[t] = cur
    return s


SEG = seg_chain(TOK)


def job(L):
    import lightgbm as lgb
    ids, w, xn = T.load_layer(L, "sm120tf", trace=f"{P}/trace_sm120")
    sl = np.concatenate([np.arange(s * T.SEQ, (s + 1) * T.SEQ) for s in SRCW])
    ids, w, xn = ids[sl], w[sl], xn[sl]
    cnt, cnta, nans, sal, seg_last = T.block_mats(ids, w, xn, SEG)
    b = lgb.Booster(model_file=V2)
    S = {}
    for tag, fx in (("f26", {L: fixed26[L]}), ("nofix", {L: []})):
        parts = []
        for c0 in range(0, cnt.shape[0], NBC):
            s = slice(c0, c0 + NBC)
            X, cand, top, e = T.chain_features(L, fx, cnt[s], cnta[s], nans[s], seg_last[s], 0, 256)
            X2 = T.v2_features(cnt[s], sal[s], cand, nbc=NBC)
            pr = b.predict(np.concatenate([X, X2], -1).reshape(-1, 9), num_threads=1)
            parts.append(T.score_blocks(pr, cand, top, e))
        S[tag] = np.concatenate(parts)
    bc, bs = cnt.astype(np.float64), sal.astype(np.float64)
    out = {}
    for name, k, hm in ARMS:
        if k == 26:
            fx, st, nf, Sx = fixed26[L], fdef[L], 51, S["f26"]
        else:
            m = MAN[k]
            fx, st, nf, Sx = m["default_allocation"][str(L)], m["floating_default"][str(L)], 77 - k, S["nofix"]
        sv = T.sim_layer(Sx, fx, st, nf=nf, hm=hm, nbc=NBC, lag=0)
        ch = [float((sv[c0 + 1:c0 + NBC] & ~sv[c0:c0 + NBC - 1]).sum(1).mean()) for c0 in range(0, len(sv), NBC)]
        sv = sv.copy(); sv[:, fx] = True
        out[name] = dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()),
                         churn=float(np.mean(ch)))
    return L, out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "4"))) as p:
        res = dict(p.map(job, T.LAYERS))
    summ = {n: {k: float(np.mean([res[L][n][k] for L in T.LAYERS])) for k in ("sal", "cnt", "churn")}
            for n, _, _ in ARMS}
    for n, s in summ.items():
        print(f"[sm120tfk] {n:10s} sal-hot {s['sal'] * 100:6.2f}  routes-hot {s['cnt'] * 100:6.2f}  churn {s['churn']:5.2f}")
    json.dump(dict(summary=summ, per_layer={str(L): res[L] for L in T.LAYERS}), open(f"{O}/dec_offline.json", "w"),
              indent=1)
