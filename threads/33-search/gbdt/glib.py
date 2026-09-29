"""T33g gbdt: shared lib. Per-layer block feature matrices [nb, 256, F] for calib-fit / glm52-heldout (from T32
rows_bandall + rows_v2_bandall) and sm120tf (T32 sm120.feats, real salience); eval = all-slot sal-hot, sync lag 0,
hysteresis sweep, churn; matched-churn interpolation."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np  # noqa: E402
import t32lib as T  # noqa: E402

W = "/tmp/nestquant/33-search/gbdt"
NE, G = 256, 16
NBC = T.CHAIN * T.SEQ // G      # 512 blocks per chain (calib / heldout)
BASE9 = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128"]
FIXED, FDEF = T.serve_sets()
BANDS = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}


def segs_of(corpus, nb):
    if corpus in ("calib-fit", "glm52-heldout"):
        return [(s, min(s + NBC, nb)) for s in range(0, nb, NBC)]
    m = json.load(open(f"{T.OUT}/private/sm120/blk/{corpus}/meta.json"))
    b = m["bstart"]
    return [(s, e) for s, e in zip(b[:-1], b[1:]) if e > s]


def load_base(corpus, L):
    """-> dict: X9 [nb,256,9] f32, bcnt [nb,256] f64, bsal f64, segs, ysal (next-64 sal, nan where invalid)"""
    if corpus in ("calib-fit", "glm52-heldout"):
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
        X2 = np.load(f"{T.OUT}/rows_v2_bandall/{corpus}/L{L}.npz")["X2"]
        Xc = np.concatenate([d["X"], X2], -1)                       # candidate order -> expert order
        X9 = np.empty_like(Xc)
        np.put_along_axis(X9, d["cand"].astype(np.int64)[..., None], Xc, 1)
        bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
        sg = segs_of(corpus, bc.shape[0])
        return dict(X9=X9, bcnt=bc, bsal=bs, segs=sg, ysal=fut(bs, sg), mL=float(d["slot_sal_sum"] / d["slots"]))
    import sm120 as S
    d, meta = S.load_blk(corpus, L)
    F = S.feats(d, meta, L, set(BASE9) | {"e256"})
    X9 = np.stack([F[n] for n in BASE9], -1).astype(np.float32)
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    sg = segs_of(corpus, bc.shape[0])
    return dict(X9=X9, bcnt=bc, bsal=bs, segs=sg, ysal=fut(bs, sg), mL=float(bs.sum() / (bc.sum())))


def fut(M, sg, k=4):
    """F[b] = sum of blocks b+1..b+k within chain, nan beyond."""
    F = np.full(M.shape, np.nan, np.float32)
    for s, e in sg:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        b = np.arange(max(e - s - k, 0))
        F[s + b] = cs[b + 1 + k] - cs[b + 1]
    return F


def replay(S, L, segs, hm=0.5, nf=51):
    """= t32lib.sim_layer lag 0 with arbitrary chain segments -> serve [nb,NE] bool (floating only)."""
    fixed = np.zeros(NE, bool); fixed[FIXED[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in FDEF[L] if e not in set(FIXED[L])][:nf]] = True
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    V = np.where(fixed[None], -np.inf, S).astype(np.float32)
    pos = (np.where(fixed[None], 0, np.maximum(S, 0)).sum(1) > 0)
    f1 = np.float32(1 + hm)
    for s, e in segs:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            if pos[k]:
                v = np.where(want, V[k] * f1, V[k])
                nw = np.zeros(NE, bool); nw[np.argpartition(-v, nf)[:nf]] = True   # ties: rare (float scores)
                want = nw
    return serve


def metrics(S, L, base, hms=(0.5,)):
    bs, bc = base["bsal"], base["bcnt"]
    out = {}
    for hm in hms:
        sv = replay(S, L, base["segs"], hm)
        # churn: new floating per refresh, within chain (chain boundary transitions excluded like T32? T32 includes all)
        ch = (sv[1:] & ~sv[:-1]).sum(1).astype(np.float64)
        sv[:, FIXED[L]] = True
        out[hm] = dict(sal=float((bs * sv).sum() / bs.sum()), cnt=float((bc * sv).sum() / bc.sum()), churn=float(ch.mean()),
                       num=float((bs * sv).sum()), den=float(bs.sum()), chs=float(ch.sum()), chn=len(ch))
    return out


def summarize(res, layers=T.LAYERS):
    """res {L: {hm: dict}} -> {hm: (sal%, churn, bands)}"""
    hms = list(res[layers[0]])
    s = {}
    for hm in hms:
        s[hm] = dict(sal=100 * np.mean([res[L][hm]["sal"] for L in layers]),
                     churn=np.mean([res[L][hm]["churn"] for L in layers]),
                     **{b: 100 * np.mean([res[L][hm]["sal"] for L in r if L in res]) for b, r in BANDS.items()})
    return s


def at_churn(s, target):
    """linear interpolation of sal-hot at a churn target over the hm sweep."""
    pts = sorted((v["churn"], v["sal"]) for v in s.values())
    c = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    if target < c[0] or target > c[-1]:
        return float("nan")
    return float(np.interp(target, c, y))


# ------------------------------------------------------------------------------------------------ extended features
from scipy.signal import lfilter  # noqa: E402

HLS = (16, 32, 64, 128, 256, 512, 1024)          # half-lives (tokens) for the adaptive EMA
NTRAIN_CH = 26                                   # calib chains 0..25 train, 26..31 val
PRI = ["p_rate", "p_sph", "p_burst", "p_ac1", "p_ac4", "p_ac16", "p_hl", "p_shl", "p_delta", "p_drank", "p_G",
       "p_zero", "layer"]
DYN = ["e256", "sema256", "ema64", "sema64", "ema512", "sema512", "r_sema128", "r_e256", "ema_best", "sema_best",
       "gap_rate", "sal_share"]
ALLX = BASE9 + DYN + PRI


def ema_seg(M, h, segs):
    a = 0.5 ** (G / h)
    E = np.empty(M.shape, np.float32)
    for s, e in segs:
        E[s:e] = lfilter([1.0], [1.0, -a], M[s:e], axis=0) * ((1 - a) / G)
    return E


def priors_from(bc, bs, segs, ys, L):
    """per-expert static priors [NE, k] from the given chains (train chains only)."""
    nbk = sum(e - s for s, e in segs)
    cnt = np.concatenate([bc[s:e] for s, e in segs]); sal = np.concatenate([bs[s:e] for s, e in segs])
    lay_sph = sal.sum() / max(cnt.sum(), 1)
    rate = cnt.mean(0) / G
    sph = np.where(cnt.sum(0) > 0, sal.sum(0) / np.maximum(cnt.sum(0), 1) / lay_sph, 1.0)
    burst = cnt.var(0) / np.maximum(cnt.mean(0), 1e-6)
    zero = (cnt == 0).mean(0)
    m = cnt.mean(0)
    acs = []
    for k in (1, 4, 16):
        num = np.zeros(NE); den = np.zeros(NE)
        for s, e in segs:
            x = bc[s:e] - m
            num += (x[:-k] * x[k:]).sum(0); den += (x * x).sum(0)
        acs.append(num / np.maximum(den, 1e-9))
    # best half-life: corr(EMA_h, next-64) per expert, count and salience EMAs
    y = np.concatenate([ys[s:e] for s, e in segs]); ok = np.isfinite(y[:, 0])
    y = y[ok]
    best = []
    for M in (bc, bs):
        cs = []
        for h in HLS:
            E = np.concatenate([ema_seg(M[s:e], h, [(0, e - s)]) for s, e in segs])[ok].astype(np.float64)
            E = E - E.mean(0); yy = y - y.mean(0)
            cs.append((E * yy).sum(0) / np.sqrt(np.maximum((E * E).sum(0) * (yy * yy).sum(0), 1e-30)))
        best.append(np.argmax(np.stack(cs), 0).astype(np.float32))
    dt = np.load("/tmp/nestquant/31-delta/delta_table.npz")
    li = list(dt["layers"]).index(L)
    delta = dt["delta"][li]; Gg = dt["G"][li]
    nfx = np.ones(NE, bool); nfx[FIXED[L]] = False
    drank = np.argsort(np.argsort(-np.where(nfx, delta, -np.inf))).astype(np.float32)
    dn = delta / np.median(delta[nfx])
    return np.stack([rate, sph, burst, acs[0], acs[1], acs[2], best[0], best[1], dn, drank, Gg / np.median(Gg[nfx]), zero,
                     np.full(NE, L)], 1).astype(np.float32)


def dyn_feats(bc, bs, segs, pri):
    nb = bc.shape[0]
    nrm = ema_seg(bs, 256, segs).sum(1) / np.maximum(ema_seg(bc, 256, segs).sum(1), 1e-30)
    nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
    F = {}
    F["e256"] = ema_seg(bc, 256, segs); F["sema256"] = ema_seg(bs, 256, segs) / nrm
    F["ema64"] = ema_seg(bc, 64, segs); F["sema64"] = ema_seg(bs, 64, segs) / nrm
    F["ema512"] = ema_seg(bc, 512, segs); F["sema512"] = ema_seg(bs, 512, segs) / nrm
    s128 = ema_seg(bs, 128, segs)
    F["r_sema128"] = np.argsort(np.argsort(-s128, 1, kind="stable"), 1).astype(np.float32)
    F["r_e256"] = np.argsort(np.argsort(-F["e256"], 1, kind="stable"), 1).astype(np.float32)
    hc, hs = pri[:, 6].astype(int), pri[:, 7].astype(int)
    eb = np.zeros((nb, NE), np.float32); sb = np.zeros((nb, NE), np.float32)
    for i, h in enumerate(HLS):
        mc, ms = hc == i, hs == i
        if mc.any():
            eb[:, mc] = ema_seg(bc[:, mc], h, segs)
        if ms.any():
            sb[:, ms] = (ema_seg(bs[:, ms], h, segs) / nrm)
    F["ema_best"], F["sema_best"] = eb, sb
    F["gap_rate"] = F["ema64"] / np.maximum(pri[None, :, 0], 1e-6)          # current vs calib base rate
    F["sal_share"] = s128 / np.maximum(s128.sum(1, keepdims=True), 1e-30)
    return np.stack([F[n] for n in DYN], -1).astype(np.float32)


def full_feats(corpus, L, pri):
    """-> base dict + XA [nb, NE, len(ALLX)]"""
    b = load_base(corpus, L)
    D = dyn_feats(b["bcnt"], b["bsal"], b["segs"], pri)
    P = np.broadcast_to(pri[None], (D.shape[0], NE, pri.shape[1]))
    b["XA"] = np.concatenate([b.pop("X9"), D, P], -1)
    return b
