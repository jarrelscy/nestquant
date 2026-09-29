#!/usr/bin/env python3
"""T32 SM120 agent-trace test (count-only, routed ids): the GLM-5.3 Terminal-Bench / domain-probe traces from the SM120
serve (serving/predictor/traces: ex [N,75,8], tok, dec, req; no gate weights or norms), plus calib-fit / glm52-heldout
via their routing.  Metric: all-slot hits-hot % (26 fixed + 51 floating), sync (lag 0), hysteresis 1.5, churn, and
return-event coverage (routed hits whose previous hit on the same expert in the stream is >= 256 / 1024 tokens back).
Streams (chain = one predictor state, reset at chain start; new_request = think state at each request start):
  sm120dec   decode rows of the 7 tasks with decode flags (the stream the serve's predictor sees), one chain per task
  sm120all   all kept rows, one chain per file (14 tasks + 65 probes)
  probemix   the 65 probes (all rows) in a fixed shuffled order as ONE chain (domains return after other domains)
  calib-fit glm52-heldout   8192-token chains (rows_bandall bcnt; returns from rows_lh)
Blocks = 16 consecutive rows of the stream.  Count features (vectorised, same definitions as the serve /
t32lib / build_x / build_lh; salience substituted by counts: sema=ema, sal16=hits16, mps128=1):
  ema32 ema128 mem_cur_state tok_since_hit hits16 | ema512 ema2048 ema8192 | cum_rate pers8 pers32 pers128 gap_mean
  gap_cv dom_max dom_busy                                                                       target: next-64 hits
PRIVATE: all outputs under $OUT/private/sm120 (per-token-derived) except the trained tree files.
  sm120.py prep [STREAM ...]
  sm120.py rows STREAM SUB           sampled training rows (non-fixed experts, valid horizon)
  sm120.py train NAME STREAM FEATS [fold]      fold A|B = sm120dec chain fold, -A = all but fold A, '' = all
  sm120.py eval STREAM NAME=MODEL[@native|@v2c] ...   (+ ema, orc_count built in)"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

import t32lib as T  # noqa: E402

G, NE = T.G, T.NE
SRC = f"{T.OUT}/sm120_traces/serving/predictor/traces"
PD = f"{T.OUT}/private/sm120"
BLK = f"{PD}/blk"
MD = f"{T.OUT}/models/sm120"
THINK, ETHINK = 154841, 154842
FOLD = {"A": ("fin-saccr-rwa", "formal-crypto", "sound-change-cascade", "cad-model"),
        "B": ("pretrain-shard-corruption", "freight-dispatch-shift", "embedding-drift-monitor")}
F5 = ["ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16"]
LO = ["ema512", "ema2048"]
LH = ["ema8192", "cum_rate", "pers8", "pers32", "pers128", "gap_mean", "gap_cv", "dom_max", "dom_busy"]
ALLF = F5 + LO + LH
fixed, fdef = T.serve_sets()


# ------------------------------------------------------------------------------------------------ prep
def seg_tokens(tok, req_start):
    """per row: 0 think / 1 answer = state after the emitted token tok[t+1] (serve step() semantics); think at each
    request start."""
    s = np.zeros(len(tok), np.int8)
    cur = 0
    nxt = np.r_[tok[1:], -1]
    rs = np.r_[req_start[1:], True]                     # next row starts a new request -> no emitted token known
    for t in range(len(tok)):
        if req_start[t]:
            cur = 0
        nt = -1 if rs[t] else nxt[t]
        if nt == THINK:
            cur = 0
        elif nt == ETHINK:
            cur = 1
        s[t] = cur
    return s


def load_stream(name):
    """-> ex [N,75,8] u8, seg [N] i8, chain names, chain row starts."""
    files = sorted(glob.glob(f"{SRC}/*.npz"))
    tasks = [f for f in files if not os.path.basename(f).startswith("probe-")]
    probes = [f for f in files if os.path.basename(f).startswith("probe-")]
    if name == "sm120dec":
        sel = [(f, "dec") for f in tasks]
    elif name == "sm120all":
        sel = [(f, "all") for f in tasks + probes]
    elif name == "probemix":
        order = np.random.default_rng(0).permutation(len(probes))
        sel = [(probes[i], "all") for i in order]
    exs, segs, names, starts, n = [], [], [], [], 0
    one = name == "probemix"
    for f, how in sel:
        z = np.load(f)
        m = z["dec"] if how == "dec" else np.ones(len(z["tok"]), bool)
        if m.sum() < 16 * G:
            continue
        tok, req = z["tok"][m], z["req"][m]
        rst = np.r_[True, req[1:] != req[:-1]]
        segs.append(seg_tokens(tok, rst))
        exs.append(z["ex"][m])
        if not one or not names:
            names.append("probemix" if one else os.path.basename(f)[:-4]); starts.append(n)
        n += int(m.sum())
    ex, seg = np.concatenate(exs), np.concatenate(segs)
    return ex, seg, names, np.asarray(starts + [n], np.int64)


def blocks_of(starts):
    """row starts of chains -> per chain (row0, nblk)."""
    return [(int(a), int((b - a) // G)) for a, b in zip(starts[:-1], starts[1:])]


def ret_events(ids, chain_of_row, rows):
    """ids [N,8] (rows in stream order), -> per block (nb over the kept block rows) return counts gap>=256,1024."""
    t = np.repeat(np.arange(len(ids)), 8); e = ids.astype(np.int64).ravel()
    o = np.lexsort((t, e)); t, e = t[o], e[o]
    same = np.r_[False, (e[1:] == e[:-1]) & (chain_of_row[t[1:]] == chain_of_row[t[:-1]])]
    gap = np.r_[0, t[1:] - t[:-1]]
    out = []
    for g in (256, 1024):
        r = same & (gap >= g) & (rows[t] >= 0)
        C = np.zeros((int(rows.max()) + 1, NE), np.float32)
        np.add.at(C, (rows[t[r]], e[r]), 1.0)
        out.append(C)
    return out


def prep_sm(name):
    t0 = time.time()
    ex, seg, names, starts = load_stream(name)
    cb = blocks_of(starts)
    blkrow = np.full(len(seg), -1, np.int64)          # row -> block index (rows in a chain's partial tail block: -1)
    chain_of_row = np.zeros(len(seg), np.int32)
    bstart, nbt = [], 0
    for ci, (r0, nb) in enumerate(cb):
        blkrow[r0:r0 + nb * G] = nbt + np.arange(nb * G) // G
        chain_of_row[r0:starts[ci + 1]] = ci
        bstart.append(nbt); nbt += nb
    keep = blkrow >= 0
    segk = seg[keep]
    nans = segk.reshape(-1, G).sum(1).astype(np.uint8)
    segl = segk.reshape(-1, G)[:, -1].astype(np.uint8)
    os.makedirs(f"{BLK}/{name}", exist_ok=True)
    json.dump(dict(chains=names, bstart=bstart + [nbt], rows=int(keep.sum()), all_rows=int(len(seg)),
                   prep_s=time.time() - t0), open(f"{BLK}/{name}/meta.json", "w"), indent=1)
    global _S
    _S = (ex, keep, segk, blkrow, chain_of_row, nans, segl, name)
    with Pool(int(os.environ.get("NPROC", "24"))) as p:
        for r in p.imap_unordered(_prep_layer, range(75)):
            pass
    print(name, "blocks", nbt, "rows", int(keep.sum()), f"{time.time() - t0:.0f}s", flush=True)


def _prep_layer(i):
    ex, keep, segk, blkrow, chain_of_row, nans, segl, name = _S
    L = T.LAYERS[i]
    ids = ex[:, i, :]
    nb = int(keep.sum()) // G
    b = np.repeat(blkrow[keep], 8); e = ids[keep].astype(np.int64).ravel()
    bc = np.bincount(b * NE + e, minlength=nb * NE).reshape(nb, NE)
    a = np.repeat(segk.astype(bool), 8)
    bca = np.bincount(b[a] * NE + e[a], minlength=nb * NE).reshape(nb, NE)
    r256, r1024 = ret_events(ids, chain_of_row, blkrow)
    np.savez(f"{BLK}/{name}/L{L}.npz", bcnt=bc.astype(np.uint8), bcnta=bca.astype(np.uint8), nans=nans, segl=segl,
             ret256=r256[:nb].astype(np.uint8), ret1024=r1024[:nb].astype(np.uint8))
    return L


def _prep_calib_layer(args):
    corpus, L = args
    ids, _, _ = T.load_layer(L, corpus)
    tok = T.tokens(corpus, ids.shape[0] // T.SEQ)
    seg = T.seg_of(tok)
    nb = ids.shape[0] // G
    b = np.repeat(np.arange(ids.shape[0]) // G, 8); e = ids.astype(np.int64).ravel()
    bc = np.bincount(b * NE + e, minlength=nb * NE).reshape(nb, NE)
    a = np.repeat(seg.astype(bool), 8)
    bca = np.bincount(b[a] * NE + e[a], minlength=nb * NE).reshape(nb, NE)
    lh = np.load(f"{T.OUT}/rows_lh/{corpus}/L{L}.npz")
    np.savez(f"{BLK}/{corpus}/L{L}.npz", bcnt=bc.astype(np.uint8), bcnta=bca.astype(np.uint8),
             nans=seg.reshape(nb, G).sum(1).astype(np.uint8), segl=seg.reshape(nb, G)[:, -1].astype(np.uint8),
             ret256=lh["ret256_cnt"].astype(np.uint8), ret1024=lh["ret1024_cnt"].astype(np.uint8))
    return nb


def prep_calib(corpus):
    os.makedirs(f"{BLK}/{corpus}", exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "24"))) as p:
        nbs = p.map(_prep_calib_layer, [(corpus, L) for L in T.LAYERS])
    nbc = T.CHAIN * T.SEQ // G
    nb = nbs[0]
    json.dump(dict(chains=[f"{corpus}#{k}" for k in range(nb // nbc)], bstart=list(range(0, nb + 1, nbc)),
                   rows=nb * G), open(f"{BLK}/{corpus}/meta.json", "w"), indent=1)
    print(corpus, "blocks", nb, flush=True)


# ------------------------------------------------------------------------------------------------ features
def segs(meta):
    bs = meta["bstart"]
    return list(zip(bs[:-1], bs[1:]))


def ema(M, h, sg):
    a = 0.5 ** (G / h)
    E = np.empty(M.shape, np.float32)
    for s, e in sg:
        E[s:e] = lfilter([1.0], [1.0, -a], M[s:e], axis=0) * ((1 - a) / G)
    return E


def gaps(h, W=128):
    """= build_lh.gaps: h [nb, NE] bool (one chain) -> mean, cv of inter-hit block gaps among hit blocks in (b-W, b]."""
    nb, ne = h.shape
    gm = np.full((nb, ne), np.nan); gc = np.full((nb, ne), np.nan)
    k = np.arange(nb)
    for e in range(ne):
        hb = np.nonzero(h[:, e])[0]
        if len(hb) < 2:
            continue
        g = np.diff(hb).astype(np.float64)
        Q1 = np.concatenate([[0.0], np.cumsum(g)]); Q2 = np.concatenate([[0.0], np.cumsum(g * g)])
        lo = np.searchsorted(hb, k - W + 1, "left"); hi = np.searchsorted(hb, k, "right")
        n = hi - lo - 1
        ok = n >= 1
        hc, lc = np.maximum(hi - 1, 0), np.minimum(lo, len(hb) - 1)
        s1 = Q1[hc] - Q1[lc]; s2 = Q2[hc] - Q2[lc]
        m = np.where(ok, s1 / np.maximum(n, 1), np.nan)
        var = np.maximum(np.where(ok, s2 / np.maximum(n, 1), np.nan) - m * m, 0)
        gm[:, e] = m; gc[:, e] = np.sqrt(var) / m
    return gm, gc


def feats(d, meta, L, need):
    """-> dict name -> [nb, NE] float32 (+ e256)."""
    sg = segs(meta)
    bc = d["bcnt"].astype(np.float64)
    nb = bc.shape[0]
    F = {}
    for h in (32, 128, 256, 512, 2048, 8192):
        n = "e256" if h == 256 else f"ema{h}"
        if n in need or (n == "ema128" and ("dom_max" in need)):
            F[n] = ema(bc, h, sg)
    k = np.arange(nb)
    if "tok_since_hit" in need:
        tsh = np.empty((nb, NE), np.float32)
        for s, e in sg:
            last = np.maximum.accumulate(np.where(bc[s:e] > 0, k[s:e, None] - s, -10 ** 6), 0)
            tsh[s:e] = np.minimum(G * (k[s:e, None] - s + 1 - last), 1e5)
        F["tok_since_hit"] = tsh
    if "bsal" in d.files and need & {"sema32", "sema128", "sal16", "mps128"}:     # real salience (sm120tf recapture)
        bs = d["bsal"].astype(np.float64)
        a256, a128 = (1 - 0.5 ** (G / 256)) / G, (1 - 0.5 ** (G / 128)) / G
        nrm = ema(bs, 256, sg).sum(1) / np.maximum(ema(bc, 256, sg).sum(1), 1e-30)
        nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
        F["sema32"] = (ema(bs, 32, sg) / nrm).astype(np.float32)
        s128, c128 = ema(bs, 128, sg), ema(bc, 128, sg)
        F["sema128"] = (s128 / nrm).astype(np.float32)
        F["sal16"] = (bs / nrm).astype(np.float32)
        F["mps128"] = np.where(c128 / a128 > 1e-3, s128 / np.maximum(c128, 1e-30) / nrm, 1.0).astype(np.float32)
        del a256
    if "hits16" in need:
        F["hits16"] = bc.astype(np.float32)
    if "mem_cur_state" in need:
        ca = d["bcnta"].astype(np.float32); c32 = bc.astype(np.float32)
        na, sl = d["nans"].astype(np.float64), d["segl"]
        sa = 0.5 ** (1 / 2048)
        out = np.empty((nb, NE), np.float32)
        for s, e in sg:
            Et = np.zeros(NE, np.float32); Ea = np.zeros(NE, np.float32); wt = wa = 0.0
            for j in range(s, e):
                nt = G - na[j]; dt = np.float32(sa ** nt); da = np.float32(sa ** na[j])
                Et = Et * dt + (c32[j] - ca[j]); wt = wt * dt + nt; Ea = Ea * da + ca[j]; wa = wa * da + na[j]
                out[j] = Et / max(wt, 1e-6) if sl[j] == 0 else Ea / max(wa, 1e-6)
        F["mem_cur_state"] = out
    if "cum_rate" in need:
        cr = np.empty((nb, NE), np.float32)
        for s, e in sg:
            cr[s:e] = np.cumsum(bc[s:e], 0) / ((np.arange(e - s) + 1) * G)[:, None]
        F["cum_rate"] = cr
    h = bc > 0
    for n in (8, 32, 128):
        if f"pers{n}" in need:
            P = np.empty((nb, NE), np.float32)
            for s, e in sg:
                cs = np.vstack([np.zeros((1, NE)), np.cumsum(h[s:e], 0)])
                kk = np.arange(e - s); lo = np.maximum(kk + 1 - n, 0)
                P[s:e] = (cs[kk + 1] - cs[lo]) / (kk + 1 - lo)[:, None]
            F[f"pers{n}"] = P
    if "gap_mean" in need or "gap_cv" in need:
        gm = np.empty((nb, NE), np.float32); gc = np.empty((nb, NE), np.float32)
        for s, e in sg:
            gm[s:e], gc[s:e] = gaps(h[s:e])
        F["gap_mean"], F["gap_cv"] = gm, gc
    if "dom_max" in need or "dom_busy" in need:
        a = 0.5 ** (G / 8192)
        r = F["ema128"]; busy = np.where(bc >= 2, bc / G, np.nan)
        mx = np.empty((nb, NE), np.float32); db = np.empty((nb, NE), np.float32)
        for s, e in sg:
            m = np.zeros(NE); num = np.zeros(NE); den = np.zeros(NE)
            for j in range(s, e):
                m = np.maximum(m * a, r[j]); mx[j] = m
                bb = ~np.isnan(busy[j])
                num = num * a + np.where(bb, busy[j], 0); den = den * a + bb
                db[j] = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        F["dom_max"], F["dom_busy"] = mx, db
    return F


def target64(bc, meta):
    Y = np.full(bc.shape, np.nan, np.float32)
    for s, e in segs(meta):
        cs = np.vstack([np.zeros((1, NE)), np.cumsum(bc[s:e], 0)])
        kb = np.arange(max(e - s - 4, 0))
        Y[s + kb] = cs[kb + 5] - cs[kb + 1]
    return Y


def load_blk(stream, L):
    return np.load(f"{BLK}/{stream}/L{L}.npz"), json.load(open(f"{BLK}/{stream}/meta.json"))


def chain_fold(meta):
    """per block: fold letter of its chain ('' for chains in neither fold)."""
    f = np.full(meta["bstart"][-1], "", "<U1")
    for ci, (s, e) in enumerate(segs(meta)):
        for k, v in FOLD.items():
            if meta["chains"][ci] in v:
                f[s:e] = k
    return f


# ------------------------------------------------------------------------------------------------ training rows
def _rows_layer(args):
    stream, L, sub = args
    d, meta = load_blk(stream, L)
    F = feats(d, meta, L, set(ALLF))
    y = target64(d["bcnt"].astype(np.float64), meta)
    nb = y.shape[0]
    rng = np.random.default_rng(L)
    pick = np.flatnonzero((np.arange(nb) % sub == (L % sub)) & np.isfinite(y).all(1))
    nf = np.ones(NE, bool); nf[fixed[L]] = False
    X = np.stack([F[n][pick][:, nf] for n in ALLF], -1).reshape(-1, len(ALLF))
    fo = np.repeat(chain_fold(meta)[pick], nf.sum())
    del rng
    return L, X.astype(np.float32), y[pick][:, nf].ravel().astype(np.float32), fo


def make_rows(stream, sub):
    out = f"{PD}/rows/{stream}"
    os.makedirs(out, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        Xs, ys, fs = [], [], []
        for L, X, y, fo in p.imap(_rows_layer, [(stream, L, sub) for L in T.LAYERS]):
            Xs.append(X); ys.append(y); fs.append(fo)
    np.save(f"{out}/X.npy", np.concatenate(Xs)); np.save(f"{out}/y.npy", np.concatenate(ys))
    np.save(f"{out}/fold.npy", np.concatenate(fs))
    json.dump(dict(feats=ALLF, sub=sub, n=int(sum(len(y) for y in ys))), open(f"{out}/meta.json", "w"))
    print(stream, "rows", sum(len(y) for y in ys), flush=True)


def train(name, stream, feats_, fold=""):
    import lightgbm as lgb
    t0 = time.time()
    fl = feats_.split(",")
    rd = f"{PD}/rows/{stream}"
    X = np.load(f"{rd}/X.npy", mmap_mode="r"); y = np.load(f"{rd}/y.npy")
    m = np.ones(len(y), bool)
    if fold:
        fo = np.load(f"{rd}/fold.npy")
        m = (fo != fold[1:]) if fold.startswith("-") else (fo == fold)
    cols = [ALLF.index(n) for n in fl]
    Xm = np.ascontiguousarray(X[m][:, cols]) if not m.all() else np.ascontiguousarray(X[:, cols])
    ym = y[m]
    p = dict(objective="poisson", metric="poisson", learning_rate=0.3, num_leaves=15, min_data_in_leaf=500,
             bagging_fraction=0.5, bagging_freq=1, bagging_seed=3, feature_fraction=0.9, seed=0,
             num_threads=int(os.environ.get("THR", "32")), verbosity=-1, max_bin=255)
    bst = lgb.train(p, lgb.Dataset(Xm, ym, feature_name=fl, free_raw_data=True), num_boost_round=60)
    os.makedirs(MD, exist_ok=True)
    bst.save_model(f"{MD}/{name}.txt")
    json.dump(dict(features=fl, target="cnt64", obj="poisson", params=p, iters=60, train=stream, fold=fold,
                   n_train=int(len(ym)), y_mean=float(ym.mean()), wall_s=time.time() - t0),
              open(f"{MD}/{name}.txt.meta.json", "w"), indent=1)
    print(f"saved {name} rows {len(ym)} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ eval
def replay(S, fx, fd, sg, hm=0.5, nf=51):
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    for s, e in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            v = np.where(fx, -np.inf, S[k]).astype(np.float32)
            r = want & ~fx
            v = np.where(r, v * np.float32(1 + hm), v)
            if np.where(fx, 0, np.maximum(S[k], 0)).sum() <= 0:
                nw = r
            else:
                nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
            want = nw & ~fx
    return serve


def _eval_layer(args):
    stream, L, models = args
    import lightgbm as lgb
    d, meta = load_blk(stream, L)
    sg = segs(meta)
    bc = d["bcnt"].astype(np.float64)
    nb = bc.shape[0]
    fx = np.zeros(NE, bool); fx[fixed[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in fdef[L] if e not in set(fixed[L])][:51]] = True
    boosters = {n: (lgb.Booster(model_file=p.split("@")[0]), p.split("@")[1] if "@" in p else "")
                for n, p in models.items()}
    need = {"e256", "ema128"}
    real = "bsal" in d.files
    sub = {} if real else {"sema32": "ema32", "sema128": "ema128", "sal16": "hits16"}
    for b, how in boosters.values():
        for f in b.feature_name():
            need.add(sub.get(f, f) if f != "mps128" or real else "ema128")
        if how == "mps":
            need.add("mps128")
    F = feats(d, meta, L, need)
    nfx = np.flatnonzero(~fx)
    Ss = {"ema": np.where(fx, 0, F["ema128"]).astype(np.float32)}
    for n, (b, how) in boosters.items():
        names = b.feature_name()
        S = np.zeros((nb, NE), np.float32)
        if how in ("native", "mps"):                      # serve band: EMA256 ranks 20..120 scored, top 20 forced
            sc = np.where(fx, -np.inf, F["e256"])
            order = np.argsort(-sc, 1, kind="stable")
            cand, top = order[:, 20:121], order[:, :20]
        else:
            cand = np.broadcast_to(nfx, (nb, len(nfx)))
            top = None
        CH = 20000
        for c0 in range(0, nb, CH):
            ci = cand[c0:c0 + CH]
            cols = []
            for f in names:
                if f == "mps128" and not real:
                    cols.append(np.ones(ci.shape, np.float32))
                else:
                    cols.append(np.take_along_axis(F[sub.get(f, f)][c0:c0 + CH], ci, 1))
            X = np.stack(cols, -1).reshape(-1, len(names))
            pr = b.predict(X, num_threads=int(os.environ.get("PT", "4"))).reshape(ci.shape)
            if how == "mps":                              # gbdt_x_mps: predicted hits x mps128
                pr = pr * np.take_along_axis(F["mps128"][c0:c0 + CH], ci, 1)
            np.put_along_axis(S[c0:c0 + CH], ci, pr.astype(np.float32), 1)
        if top is not None:
            np.put_along_axis(S, top, (1e3 + np.take_along_axis(F["e256"], top, 1)).astype(np.float32), 1)
        Ss[n] = S
    Ss["orc_count"] = None
    bsal = d["bsal"].astype(np.float64) if real else None
    if real:
        Ss["orc_sal"] = None
    r256, r1024 = d["ret256"].astype(np.float64), d["ret1024"].astype(np.float64)
    res = {}
    for n, S in Ss.items():
        if n in ("orc_count", "orc_sal"):                 # oracle = top-51 of the actual next 64 (no hysteresis)
            M = bc if n == "orc_count" else bsal
            sv = np.zeros((nb, NE), bool)
            # serve block k with the top-51 of the next 64 tokens starting AT block k (T18 _core_oracle)
            So = np.zeros((nb, NE))
            for s, e in sg:
                cs = np.vstack([np.zeros((1, NE)), np.cumsum(M[s:e], 0)])
                kk = np.arange(e - s)
                So[s:e] = cs[np.minimum(kk + 4, e - s)] - cs[kk]
            So[:, fx] = -np.inf
            o = np.argsort(-So, 1, kind="stable")[:, :51]
            np.put_along_axis(sv, o, True, 1)
        else:
            sv = replay(S, fx, fd, sg)
        ch = (sv[1:] & ~sv[:-1]).sum(1).astype(np.float64)
        hot = sv | fx
        per = []
        for s, e in sg:
            per.append([float((bc[s:e] * hot[s:e]).sum()), float(bc[s:e].sum()), float(ch[s:e - 1].sum()),
                        float(max(e - s - 1, 0)), float((r256[s:e] * hot[s:e]).sum()), float(r256[s:e].sum()),
                        float((r1024[s:e] * hot[s:e]).sum()), float(r1024[s:e].sum())]
                       + ([float((bsal[s:e] * hot[s:e]).sum()), float(bsal[s:e].sum())] if real else []))
        res[n] = per
    return L, res


def evaluate(stream, models):
    t0 = time.time()
    meta = json.load(open(f"{BLK}/{stream}/meta.json"))
    with Pool(int(os.environ.get("NPROC", "12"))) as p:
        res = dict(p.map(_eval_layer, [(stream, L, models) for L in T.LAYERS]))
    f = f"{PD}/eval_{stream}.json"                         # per chain per layer sums (private: per-task routing stats)
    allr = json.load(open(f)) if os.path.exists(f) else {"chains": meta["chains"], "res": {}}
    for n in res[T.LAYERS[0]]:
        allr["res"][n] = {str(L): res[L][n] for L in T.LAYERS}
    json.dump(allr, open(f, "w"))
    print(f"eval {stream} {len(models)} models {time.time() - t0:.0f}s", flush=True)
    report(stream)


def summarise(allr, name, chains=None):
    """mean over layers of per-layer pooled (over the selected chains) hits-hot, churn, ret coverage."""
    ch = allr["chains"]
    sel = [i for i, c in enumerate(ch) if chains is None or c in chains]
    hh, cu, r2, r10, rs2, rs10, sh = [], [], [], [], [], [], []
    for L, per in allr["res"][name].items():
        a = np.asarray(per)[sel].sum(0)
        hh.append(a[0] / a[1]); cu.append(a[2] / max(a[3], 1)); r2.append(a[4] / max(a[5], 1e-9))
        r10.append(a[6] / max(a[7], 1e-9)); rs2.append(a[5] / a[1]); rs10.append(a[7] / a[1])
        if len(a) > 8:
            sh.append(a[8] / a[9])
    return dict(hot=float(np.mean(hh)), churn=float(np.mean(cu)), ret256=float(np.mean(r2)),
                ret1024=float(np.mean(r10)), ret256_share=float(np.mean(rs2)), ret1024_share=float(np.mean(rs10)),
                **({"sal_hot": float(np.mean(sh))} if sh else {}))


def oof(allr, a, b):
    """out-of-fold combination: model a (trained on fold A) on fold-B chains, b on fold-A chains."""
    ra, rb = allr["res"][a], allr["res"][b]
    out = {}
    for L in ra:
        out[L] = [rb[L][i] if c in FOLD["A"] else ra[L][i] for i, c in enumerate(allr["chains"])]
    return out


def report(stream):
    allr = json.load(open(f"{PD}/eval_{stream}.json"))
    names = list(allr["res"])
    for n in list(names):                                  # sA_x / sB_x -> s_x (out of fold)
        if n.startswith("sA_") and "sB_" + n[3:] in allr["res"]:
            allr["res"]["oof_" + n[3:]] = oof(allr, n, "sB_" + n[3:])
            names.append("oof_" + n[3:])
    summ = {n: summarise(allr, n) for n in names}
    per_chain = {}
    if stream.startswith("sm120"):
        for n in names:
            per_chain[n] = {c: round(summarise(allr, n, [c])["hot"] * 100, 2) for c in allr["chains"]
                            if not c.startswith("probe-")}
    json.dump(dict(summary=summ, per_chain=per_chain), open(f"{T.OUT}/sm120_summary_{stream}.json", "w"), indent=1)
    print(f"[{stream}] hits-hot % sync (all slots), churn, return-hit coverage (gap>=256 / >=1024 tokens)")
    for n in names:
        s = summ[n]
        print(f"  {n:14s} hot {s['hot'] * 100:6.2f}  churn {s['churn']:5.2f}  ret256 {s['ret256'] * 100:6.2f}  "
              f"ret1024 {s['ret1024'] * 100:6.2f}  " + (f"SAL-hot {s['sal_hot'] * 100:6.2f}  " if "sal_hot" in s else "") +
              f" (ret shares {s['ret256_share'] * 100:.1f}/{s['ret1024_share'] * 100:.1f}%)",
              flush=True)


# ------------------------------------------------------------------------------------------------ (b) recapture
TFC = f"{PD}/../corpora/sm120tf.map.npz"
TFT = f"{PD}/../trace_sm120"


def prep_tf():
    """sm120tf (FP8 teacher-forced ref routing, w, |x|^2 on the decode rows of the captured windows) + sm120tfx (the
    SAME rows with the SM120 serve's routing ex, count-only) + faithfulness (ref vs ex top-8 overlap)."""
    t0 = time.time()
    mp = np.load(TFC)
    names = [str(x) for x in mp["names"]]
    rows_of, seg_of_, ex_of = {}, {}, {}
    for ti, n in enumerate(names):
        z = np.load(f"{SRC}/{n}.npz")
        wr = mp["row0"][mp["task"] == ti]
        r = (wr[:, None] + np.arange(T.SEQ)[None]).ravel()
        dm = z["dec"][r]
        rd = r[dm]
        req = z["req"]
        rst = np.r_[True, req[rd][1:] != req[rd][:-1]]
        rows_of[n] = dm                                   # per captured row of this task: decode?
        seg_of_[n] = seg_tokens(z["tok"][rd], rst)
        ex_of[n] = z["ex"][rd]
    order = np.argsort(mp["task"], kind="stable")          # capture windows are in task order already
    assert (order == np.arange(len(order))).all()
    starts, n0 = [], 0
    for n in names:
        starts.append(n0); n0 += len(seg_of_[n])
    starts = np.asarray(starts + [n0], np.int64)
    seg = np.concatenate([seg_of_[n] for n in names])
    ex = np.concatenate([ex_of[n] for n in names])
    dmask = np.concatenate([rows_of[n] for n in names])   # over all captured rows (window order)
    pos = np.tile(np.arange(T.SEQ), len(mp["task"]))
    cb = blocks_of(starts)
    blkrow = np.full(len(seg), -1, np.int64); chain_of_row = np.zeros(len(seg), np.int32)
    bstart, nbt = [], 0
    for ci, (r0, nb) in enumerate(cb):
        blkrow[r0:r0 + nb * G] = nbt + np.arange(nb * G) // G
        chain_of_row[r0:starts[ci + 1]] = ci
        bstart.append(nbt); nbt += nb
    keep = blkrow >= 0
    segk = seg[keep]
    nans = segk.reshape(-1, G).sum(1).astype(np.uint8); segl = segk.reshape(-1, G)[:, -1].astype(np.uint8)
    for st in ("sm120tf", "sm120tfx"):
        os.makedirs(f"{BLK}/{st}", exist_ok=True)
        json.dump(dict(chains=names, bstart=bstart + [nbt], rows=int(keep.sum())), open(f"{BLK}/{st}/meta.json", "w"))
    global _S
    _S = (ex, keep, segk, blkrow, chain_of_row, nans, segl, dmask, pos)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        fa = dict(p.imap_unordered(_prep_tf_layer, range(75)))
    json.dump({str(k): v for k, v in sorted(fa.items())}, open(f"{PD}/tf_faithfulness.json", "w"), indent=1)
    agg = {k: float(np.mean([fa[L][k] for L in fa])) for k in fa[T.LAYERS[0]]}
    json.dump(agg, open(f"{T.OUT}/sm120_tf_faithfulness.json", "w"), indent=1)   # aggregate only
    print("tf blocks", nbt, "decode rows", int(dmask.sum()), {k: round(v, 4) for k, v in agg.items()},
          f"{time.time() - t0:.0f}s", flush=True)


def _prep_tf_layer(i):
    ex, keep, segk, blkrow, chain_of_row, nans, segl, dmask, pos = _S
    L = T.LAYERS[i]
    ids_all, w_all, xn_all = T.load_layer(L, "sm120tf", trace=TFT)
    ids, w, xn = ids_all[dmask], w_all[dmask], xn_all[dmask]
    exL = ex[:, i, :]
    A = np.zeros((len(ids), NE), bool); np.put_along_axis(A, ids.astype(np.int64), True, 1)
    ov = np.take_along_axis(A, exL.astype(np.int64), 1).sum(1) / 8.0       # |ref top8 & serve top8| / 8
    pd_ = pos[dmask]
    fa = dict(overlap=float(ov.mean()), top1_in_ref=float(np.take_along_axis(A, exL[:, :1].astype(np.int64), 1).mean()),
              exact_set=float((ov == 1).mean()))
    for a, b in ((0, 256), (256, 1024), (1024, 2048)):
        m = (pd_ >= a) & (pd_ < b)
        fa[f"overlap_pos{a}"] = float(ov[m].mean()) if m.any() else float("nan")
    nb = int(keep.sum()) // G
    for st, I in (("sm120tf", ids), ("sm120tfx", exL)):
        b = np.repeat(blkrow[keep], 8); e = I[keep].astype(np.int64).ravel()
        bc = np.bincount(b * NE + e, minlength=nb * NE).reshape(nb, NE)
        a = np.repeat(segk.astype(bool), 8)
        bca = np.bincount(b[a] * NE + e[a], minlength=nb * NE).reshape(nb, NE)
        r256, r1024 = ret_events(I, chain_of_row, blkrow)
        extra = {}
        if st == "sm120tf":
            v = (w[keep].astype(np.float64) ** 2 * xn[keep].astype(np.float64)[:, None]).ravel()
            extra["bsal"] = np.bincount(b * NE + e, weights=v, minlength=nb * NE).reshape(nb, NE).astype(np.float32)
        np.savez(f"{BLK}/{st}/L{L}.npz", bcnt=bc.astype(np.uint8), bcnta=bca.astype(np.uint8), nans=nans, segl=segl,
                 ret256=r256[:nb].astype(np.uint8), ret1024=r1024[:nb].astype(np.uint8), **extra)
    return L, fa


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        for s in sys.argv[2:] or ["sm120dec", "sm120all", "probemix", "calib-fit", "glm52-heldout"]:
            if os.path.exists(f"{BLK}/{s}/meta.json"):
                print(s, "exists"); continue
            (prep_calib if s in ("calib-fit", "glm52-heldout") else prep_sm)(s)
    elif cmd == "preptf":
        prep_tf()
    elif cmd == "rows":
        make_rows(sys.argv[2], int(sys.argv[3]))
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "")
    elif cmd == "eval":
        evaluate(sys.argv[2], dict(x.split("=", 1) for x in sys.argv[3:]))
    elif cmd == "report":
        report(sys.argv[2])
