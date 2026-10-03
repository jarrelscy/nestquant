"""T36 Spark landing replay (CPU, PRIVATE inputs from prep.py): b175 H hot slots per layer on 2x DGX Spark with the
4-bit residuals streamed from NVMe.  Token-granular event model, all 75 layers share one FIFO queue per node (TP2:
each node reads its half of every residual, the two nodes run in parallel, so one queue models both).
  fetch: start = max(issue, queue_free); queue_free = start + bytes/B; land = queue_free + lat;
         usable from the first token whose start time >= land (tokens before that run the 1.75-bit base).
Hot-set sources (want lists, per layer per block, ordered by priority):
  jF   : k0 hm-hysteresis replay of the OOS jF scores (parta_salhot.replay semantics; set for block k uses
         scores up to block k-1), issued at block start.
  orc  : perfect block oracle (top-H by the block's own true salience; spare slots keep the previous set).
Arms: perfect landing | real landing | lead d (set for block k decided+issued at block k-d, the incoming copies wait in
a staging buffer) | per-token top-up (causal: experts routed so far in the block; oracle: next-W tokens' routing),
k swaps per layer per token, victims = lowest in-block (or future) salience, then lowest jF rank.
Metric: pooled 4-bit salience share (sum over layers of w^2|x|^2 on calls served at 4 bit / all), routed-call share,
fetches per layer per block, mean landing delay, peak staging.  Aggregates only are written (results JSON)."""
import json
import os
import sys
import time

import numpy as np
from numba import njit, prange

G, NE = 16, 256
P = "/tmp/nestquant/36-spark-land/private"
BLK = "/tmp/nestquant/32-gbdt-sal/private/sm120/blk/sm120tf"
LAYERS = list(range(3, 78))
REC = 11.42e6                      # bytes per expert residual (all 3 matrices); TP2 -> half per node


@njit(cache=True)
def replay_jf(S, fd, bs, be, H, hm):
    """-> want [nb, H] u8, ordered by priority (block k uses the list at k)."""
    nb = S.shape[0]
    W = np.zeros((nb, H), np.uint8)
    for c in range(len(bs)):
        want = fd.copy()
        inw = np.zeros(NE, np.bool_)
        for e in want:
            inw[e] = True
        for k in range(bs[c], be[c]):
            W[k] = want
            v = S[k].astype(np.float32).copy()
            pos = 0.0
            for e in range(NE):
                if v[e] > 0:
                    pos += v[e]
                if inw[e]:
                    v[e] = v[e] * np.float32(1 + hm)
            if pos <= 0:
                continue
            o = np.argsort(-v, kind="mergesort")
            want = o[:H].astype(np.uint8)
            inw[:] = False
            for e in want:
                inw[e] = True
    return W


@njit(cache=True)
def oracle_blocks(bsal, fd, bs, be, H):
    """perfect block oracle: top-H experts by the block's true salience; spare slots keep previous members."""
    nb = bsal.shape[0]
    W = np.zeros((nb, H), np.uint8)
    for c in range(len(bs)):
        prev = fd.copy()
        for k in range(bs[c], be[c]):
            o = np.argsort(-bsal[k], kind="mergesort")
            n = 0
            take = np.zeros(NE, np.bool_)
            for i in range(H):
                if bsal[k, o[i]] > 0:
                    W[k, n] = o[i]; take[o[i]] = True; n += 1
            for e in prev:
                if n >= H:
                    break
                if not take[e]:
                    W[k, n] = e; take[e] = True; n += 1
            prev = W[k].copy()
    return W


@njit(cache=True)
def simulate(ids, sal, WANT, t0s, t1s, b0s, H, tau, sreq, lat, perfect, lead, kup, upmode, Wf, cap):
    """one chain.  ids/sal [Lc, N, 8] (chain tokens), WANT [Lc, nb, H].  Returns per-layer accumulators."""
    Lc, N = ids.shape[0], ids.shape[1]
    nb = N // G
    hs = np.zeros(Lc); ts = np.zeros(Lc); hc = np.zeros(Lc); tc = np.zeros(Lc)
    nf = np.zeros(Lc); delay = 0.0; ndel = 0; peak_stage = 0; skipped = 0
    act = np.zeros((Lc, NE), np.bool_)
    usable = np.zeros((Lc, NE), np.int64)        # first token index at which the 4-bit copy is usable
    landpend = np.zeros((Lc, NE), np.int64)      # lead: land token of the copy fetched when e entered want
    rank = np.full((Lc, NE), H, np.int64)
    bsalc = np.zeros((Lc, NE))                   # in-block salience so far (causal top-up)
    fsal = np.zeros((Lc, NE))                    # next-Wf-token salience (oracle top-up)
    credit = np.zeros(Lc)
    qfree = 0.0
    for L in range(Lc):
        for i in range(H):
            act[L, WANT[L, 0, i]] = True
            rank[L, WANT[L, 0, i]] = i
    nsth = np.zeros(nb, np.int64)
    for k in range(nb):
        tb = k * G
        now = tb * tau
        # --- block boundary: issue fetches for want[k] (active at k+lead), activate want[k-lead]
        nstage = 0
        for L in range(Lc):
            if k > 0 and lead == 0:
                # in-place: every want member not resident (incl. ones a top-up evicted) is fetched now
                inw = np.zeros(NE, np.bool_)
                for i in range(H):
                    inw[WANT[L, k, i]] = True
                for i in range(H):
                    e = WANT[L, k, i]
                    if not act[L, e]:
                        if perfect:
                            usable[L, e] = tb
                        else:
                            st = max(now, qfree)
                            qfree = st + sreq
                            lt = qfree + lat
                            usable[L, e] = int(np.ceil(lt / tau - 1e-9))
                            delay += lt - now; ndel += 1
                        nf[L] += 1
                act[L, :] = inw
                rank[L, :] = H
                for i in range(H):
                    rank[L, WANT[L, k, i]] = i
            elif k > 0:
                # lead d: want[k] fetched now into staging, becomes the active set at block k+d
                for i in range(H):
                    e = WANT[L, k, i]
                    new = True
                    for j in range(H):
                        if WANT[L, k - 1, j] == e:
                            new = False
                            break
                    if new:
                        if perfect:
                            landpend[L, e] = tb + lead * G
                        else:
                            st = max(now, qfree)
                            qfree = st + sreq
                            lt = qfree + lat
                            landpend[L, e] = int(np.ceil(lt / tau - 1e-9))
                            delay += lt - now; ndel += 1
                        nstage += 1
                        nf[L] += 1
                ka = max(k - lead, 0)
                prevact = act[L].copy()
                act[L, :] = False
                rank[L, :] = H
                for i in range(H):
                    e = WANT[L, ka, i]
                    act[L, e] = True
                    rank[L, e] = i
                    if not prevact[e]:
                        usable[L, e] = max(landpend[L, e], tb)
            bsalc[L, :] = 0.0
        nsth[k] = nstage
        w = 0
        for q in range(max(0, k - lead + 1), k + 1):
            w += nsth[q]
        if w > peak_stage:
            peak_stage = w                       # copies held in staging (fetched, not yet active)
        # --- tokens of the block
        for j in range(G):
            t = tb + j
            for L in range(Lc):
                for s in range(8):
                    e = ids[L, t, s]
                    v = sal[L, t, s]
                    ts[L] += v; tc[L] += 1
                    if act[L, e] and usable[L, e] <= t:
                        hs[L] += v; hc[L] += 1
                    bsalc[L, e] += v
            if kup <= 0 or j == G - 1:
                continue
            tend = (t + 1) * tau
            for L in range(Lc):
                credit[L] += kup
                if upmode == 2:              # oracle: salience of the next Wf tokens (within the chain)
                    fsal[L, :] = 0.0
                    for u in range(t + 1, min(t + 1 + Wf, N)):
                        for s in range(8):
                            fsal[L, ids[L, u, s]] += sal[L, u, s]
                    score = fsal[L]
                else:
                    score = bsalc[L]
                while credit[L] >= 1.0:
                    if not perfect and qfree > tend + cap * tau:
                        skipped += 1
                        credit[L] = 0.0
                        break
                    cb, cv = -1, 0.0
                    for e in range(NE):
                        if not act[L, e] and score[e] > cv:
                            cv = score[e]; cb = e
                    if cb < 0:
                        credit[L] = min(credit[L], 1.0)
                        break
                    vb, vv, vr = -1, 1e300, -1
                    for e in range(NE):
                        if act[L, e]:
                            if score[e] < vv or (score[e] == vv and rank[L, e] > vr):
                                vv = score[e]; vb = e; vr = rank[L, e]
                    if vb < 0 or vv >= cv:
                        credit[L] = min(credit[L], 1.0)
                        break
                    act[L, vb] = False
                    act[L, cb] = True
                    rank[L, cb] = H
                    if perfect:
                        usable[L, cb] = t + 1
                    else:
                        st = max(tend, qfree)
                        qfree = st + sreq
                        lt = qfree + lat
                        usable[L, cb] = int(np.ceil(lt / tau - 1e-9))
                        delay += lt - tend; ndel += 1
                    nf[L] += 1
                    credit[L] -= 1.0
    return hs, ts, hc, tc, nf, delay, ndel, peak_stage, skipped


def load_all(H, hm, src):
    import importlib.util
    sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/joint")
    os.environ["LAYOUT"] = "k0"
    import jlib as J
    meta = json.load(open(f"{BLK}/meta.json"))
    bs = np.array([a for a, b in zip(meta["bstart"][:-1], meta["bstart"][1:]) if b > a], np.int64)
    be = np.array([b for a, b in zip(meta["bstart"][:-1], meta["bstart"][1:]) if b > a], np.int64)
    ids, sal, WANT = [], [], []
    for L in LAYERS:
        z = np.load(f"{P}/tok/L{L}.npz")
        ids.append(z["ids"]); sal.append(z["sal"])
        fd = np.array([int(e) for e in J.FDEF[L]][:H], np.uint8)
        if src == "jF":
            WANT.append(replay_jf(np.load(f"{P}/sc/L{L}.npy"), fd, bs, be, H, hm))
        else:
            WANT.append(oracle_blocks(np.load(f"{BLK}/L{L}.npz")["bsal"].astype(np.float64), fd, bs, be, H))
    return np.stack(ids), np.stack(sal), np.stack(WANT), bs, be


def run(ids, sal, WANT, bs, be, H, rate, bw, lat, perfect, lead, kup, upmode, Wf=16, cap=2.0):
    tau = 1.0 / rate
    sreq = REC / 2 / bw
    acc = None
    for c in range(len(bs)):
        t0, t1 = bs[c] * G, be[c] * G
        r = simulate(ids[:, t0:t1], sal[:, t0:t1], WANT[:, bs[c]:be[c]], 0, 0, 0, H, tau, sreq, lat,
                     perfect, lead, kup, upmode, Wf, cap)
        if acc is None:
            acc = [np.array(x, np.float64) if np.ndim(x) else float(x) for x in r]
            acc[7] = r[7]
        else:
            for i in range(7):
                acc[i] = acc[i] + r[i]
            acc[7] = max(acc[7], r[7]); acc[8] += r[8]
    hs, ts, hc, tc, nf, delay, ndel, peak, skipped = acc
    nb = float((be - bs).sum())
    sal_p = float(hs.sum() / ts.sum())
    return dict(sal_pooled=sal_p, sal_lmean=float(np.mean(hs / ts)), call_share=float(hc.sum() / tc.sum()),
                fetch_per_layer_block=float(nf.sum() / len(LAYERS) / nb), mean_land_delay_ms=1e3 * delay / max(ndel, 1),
                ssd_util=float(nf.sum() * REC / 2 / bw / (nb * G / rate)), peak_staging_experts=int(peak),
                topup_skipped=int(skipped), kld_est=0.0158 + (1 - sal_p) * 0.084)


@njit(cache=True)
def oracle_budget(idsL, salL, fd, bs, be, H, M):
    """bandwidth-limited block oracle for one layer: from the previous set, swap in at most M experts per block,
    greedily (best non-member by the block's true salience vs worst member) while the swap gains."""
    nb = idsL.shape[0] // G
    W = np.zeros((nb, H), np.uint8)
    for c in range(len(bs)):
        act = np.zeros(NE, np.bool_)
        for e in fd:
            act[e] = True
        for k in range(bs[c] - bs[c], be[c] - bs[c]):
            kk = bs[c] + k
            b = np.zeros(NE)
            for t in range(kk * G, kk * G + G):
                for s in range(8):
                    b[idsL[t, s]] += salL[t, s]
            for m in range(M):
                cb, cv, vb, vv = -1, 0.0, -1, 1e300
                for e in range(NE):
                    if act[e]:
                        if b[e] < vv:
                            vv = b[e]; vb = e
                    elif b[e] > cv:
                        cv = b[e]; cb = e
                if cb < 0 or cv <= vv:
                    break
                act[vb] = False; act[cb] = True
            n = 0
            for e in range(NE):
                if act[e]:
                    W[kk, n] = e; n += 1
    return W
