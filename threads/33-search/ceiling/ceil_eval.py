#!/usr/bin/env python3
"""T33l task 3: conditional-expectation ceiling from K sampled continuations per prefix.

One decision per (prefix, layer): the set chosen at the end of the prefix (block gk) serves block gk+1 = continuation
tokens 0..15.  Incumbent = v2's served set at block gk (T32 replay, hm0.4 = the ~3.2-churn v2 reference).  Every arm
picks its new 51 from the same incumbent with the additive margin dm (clib.replay rule); dm is swept and the arm is
read at mean churn 3.2 (new experts vs incumbent).  Targets: the real (teacher-forced) continuation, and each sampled
continuation held out (leave-one-out ceiling over the other K-1).
    ceil_eval.py GEN_DIR [PREFIXES]"""
import json
import sys
from multiprocessing import Pool
import numpy as np
import os
import clib as C
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/alloc")
import alib as A
T = C.T
KF = int(os.environ.get("KF", "26"))        # 26 = manifest fixed set (51 floating); 0 = 77 floating over all 256
NFL = 77 - KF

GD = sys.argv[1]
PX = json.load(open(sys.argv[2] if len(sys.argv) > 2 else f"{C.OUTC}/prefixes.json"))
g = np.load(f"{GD}/gen.npz")
order, K, steps = g["order"], int(g["k"]), int(g["steps"])
K1 = K + 1
SL = g["sparse_layers"].tolist()
NP = len(order)
DMS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3, 6, 1e9]
HM_INC = 0.4 if KF == 26 else 0.6          # the ~3.2-churn v2 reference at each k


def masks(L):
    if KF == 26:
        return C.masks(L)
    f26 = A.f26_ranked(L)
    fx = np.zeros(T.NE, bool); fx[f26[:KF]] = True
    fd = np.zeros(T.NE, bool); fd[[e for e in f26[KF:] + list(A.fdef[L]) if not fx[e]][:NFL]] = True
    return fx, fd


def ymats(n):
    """[nsp, nseq, NE] salience of continuation tokens 0..n-1."""
    n = min(n, g["ids"].shape[2])
    ids = g["ids"][:, :, :n].astype(np.int64)            # nsp, nseq, n, 8
    v = g["w"][:, :, :n].astype(np.float64) ** 2 * g["xn"][:, :, :n, None].astype(np.float64)
    ns, nq = ids.shape[:2]
    idx = (np.arange(ns * nq)[:, None, None] * T.NE + ids.reshape(ns * nq, n, 8)).ravel()
    return np.bincount(idx, weights=v.ravel(), minlength=ns * nq * T.NE).reshape(ns, nq, T.NE)


def job(L):
    """v2 incumbent (serve[gk]), v2 served next (serve[gk+1]), v2 score S[gk], trace Y16/Y64 at gk for each prefix."""
    fx, fd = masks(L)
    out = {}
    for corpus in sorted({PX[i]["corpus"] for i in order}):
        qs = [q for q, i in enumerate(order) if PX[i]["corpus"] == corpus]
        S = C.v2_scores(corpus, L)
        sv = C.replay(S, fx, fd, hm=HM_INC) if KF == 26 else \
            A.sim_seg(S, list(np.nonzero(fx)[0]), A.f26_ranked(L)[KF:] + list(A.fdef[L]), NFL, HM_INC)
        ids, w, xn = T.load_layer(L, corpus)
        for q in qs:
            x = PX[order[q]]
            gk = x["gk"]
            b0 = x["win"] * T.SEQ + x["p"]
            tr = {}
            for n in (16, 64):
                y = np.zeros(T.NE)
                np.add.at(y, ids[b0:b0 + n].astype(np.int64).ravel(),
                          (w[b0:b0 + n].astype(np.float64) ** 2 * xn[b0:b0 + n, None]).ravel())
                tr[n] = y
            out[q] = (sv[gk], sv[gk + 1], S[gk], tr[16], tr[64])
    return L, out


def pick(V, inc, fx, dm):
    """V [n,NE] scores, inc [n,NE] incumbent floating mask -> new floating mask (top-51 with additive margin)."""
    v = np.where(fx[None], -np.inf, V.astype(np.float64))
    v51 = np.partition(v, T.NE - NFL, axis=1)[:, T.NE - NFL]
    vmax = np.max(np.where(np.isfinite(v), v, 0), axis=1)
    sc = np.maximum(np.maximum(v51, 1e-6 * vmax), 1e-30)[:, None]
    v = v + (dm * sc + 1e-9 * vmax[:, None]) * inc          # + tiny tie-break toward incumbents (sparse scores)
    o = np.argpartition(-v, NFL - 1, axis=1)[:, :NFL]
    m = np.zeros(v.shape, bool)
    np.put_along_axis(m, o, True, 1)
    return m & ~fx[None]


def share(m, fx, Y):
    return float((Y * (m | fx[None])).sum() / Y.sum())


def main():
    Y16, Y64 = ymats(16), ymats(min(64, steps))
    with Pool(int(os.environ.get("NPROC", "4"))) as p:
        R = dict(p.map(job, SL))
    corp = [PX[i]["corpus"] for i in order]
    groups = {c: np.array([q for q in range(NP) if corp[q] == c]) for c in sorted(set(corp))}
    real = np.arange(NP) * K1 + K
    res = {}
    for c, qs in groups.items():
        acc = {}   # arm/target -> list over layers of [(share16, share64, churn) per dm]

        def add(name, V, inc, fx, Yt16, Yt64):
            acc.setdefault(name, []).append([(share(m, fx, Yt16), share(m, fx, Yt64),
                                              float((m & ~inc).sum(1).mean()))
                                             for m in (pick(V, inc, fx, dm) for dm in DMS)])
        chk = []
        for L in SL:
            j = SL.index(L)
            fx, _ = masks(L)
            inc = np.stack([R[L][q][0] for q in qs])
            v2next = np.stack([R[L][q][1] for q in qs])
            S = np.stack([R[L][q][2] for q in qs])
            tr16 = np.stack([R[L][q][3] for q in qs])
            tr64 = np.stack([R[L][q][4] for q in qs])
            yr16, yr64 = Y16[j, real[qs]], Y64[j, real[qs]]
            chk.append((share(v2next, fx, tr16), share(v2next, fx, yr16),
                        float(np.abs(yr16 - tr16).sum() / tr16.sum())))
            smp = np.array([[q * K1 + k for k in range(K)] for q in qs])       # nq, K
            s16, s64 = Y16[j][smp], Y64[j][smp]                                 # nq, K, NE
            # ---- target: real continuation
            add("real/v2", S, inc, fx, yr16, yr64)
            add("real/ceil16_K", s16.mean(1), inc, fx, yr16, yr64)
            add("real/ceil64_K", s64.mean(1), inc, fx, yr16, yr64)
            for kk in (1, 2, 4):
                add(f"real/ceil64_K{kk}", s64[:, :kk].mean(1), inc, fx, yr16, yr64)
            add("real/orc64_self", yr64, inc, fx, yr16, yr64)
            add("real/trace_orc64", tr64, inc, fx, tr16, tr64)
            add("real/trace_v2", S, inc, fx, tr16, tr64)
            # ---- target: each sampled continuation held out (LOO over the other K-1)
            tot16, tot64 = s16.sum(1), s64.sum(1)
            incK = np.repeat(inc, K, 0)
            t16, t64 = s16.reshape(-1, T.NE), s64.reshape(-1, T.NE)
            add("loo/v2", np.repeat(S, K, 0), incK, fx, t16, t64)
            add("loo/ceil16_K-1", ((tot16[:, None] - s16) / (K - 1)).reshape(-1, T.NE), incK, fx, t16, t64)
            add("loo/ceil64_K-1", ((tot64[:, None] - s64) / (K - 1)).reshape(-1, T.NE), incK, fx, t16, t64)
            add("loo/real64", np.repeat(yr64, K, 0), incK, fx, t16, t64)       # the real text's routing
            add("loo/orc64_self", t64, incK, fx, t16, t64)
        out = {}
        for name, per in acc.items():
            a = np.array(per)                    # nL, ndm, 3
            m = a.mean(0)
            pts16 = [(m[i, 0], m[i, 2]) for i in range(len(DMS))]
            pts64 = [(m[i, 1], m[i, 2]) for i in range(len(DMS))]
            v16, f16 = C.at_churn(pts16, 3.2)
            v64, f64 = C.at_churn(pts64, 3.2)
            out[name] = dict(dm0=list(m[0]), at3p2_blk16=v16, at3p2_next64=v64, flag=f16, grid=m.tolist())
            print(f"{c:14s} {name:22s} dm0 {100*m[0,0]:6.2f}/{100*m[0,1]:6.2f} churn {m[0,2]:5.2f}  "
                  f"@3.2: blk16 {100*v16:6.2f} next64 {100*v64:6.2f} {f16}", flush=True)
        ck = np.array(chk).mean(0)
        print(f"{c:14s} check: v2 hm{HM_INC} served next on trace {100*ck[0]:.2f} / on gen-TF {100*ck[1]:.2f}; "
              f"|Y16 gen-trace|/Y {ck[2]:.4f}; N={len(qs)} K={K}", flush=True)
        out["check"] = ck.tolist()
        out["N"] = int(len(qs))
        res[c] = out
    json.dump(res, open(f"{GD}/ceil_eval_k{KF}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
