#!/usr/bin/env python3
"""T32 M1 (hidden-state topic) + M2 (lm_head uncertainty) features on band-all rows -> rows_m1/CORPUS/L.npz
F [nb, 256(cand), 6]  (PRIVATE; inputs = private/trace3 PCA-32 projections of the RMS-normalised residual after
layers 10/40/70 + per-token reference lm_head entropy / top-1 prob, nq_e2e NQ_TRACE_HID / NQ_TRACE_HEAD).
Causal at the end of block b (sync: every layer has seen the block), chain reset.
  topic u(b) = [token-EMA16, token-EMA64 of the PCA-32 vectors at L10, L40, L70]   (192 dims)
  per layer L, ridge maps fit on calib-fit (out-of-fold by chain parity for calib-fit rows), target = next-64
  normalised salience of all 256 experts (= the v2 target):
    m1_topic  <- u(b)                                  (topic -> expert affinity)
    m1_joint  <- [u(b), sema32(b), sema128(b)]         (topic + the layer's full 256-way salience state)
    m1_state  <- [sema32(b), sema128(b)]               (linear joint 256-way model, no topic: idea-3 lite)
  M2 (per block, same value for every expert): m2_ent16 m2_p1_16 = token-EMA16 of entropy / top-1 prob,
    m2_ent_last = entropy at the block's last token (NaN where the capture has no logits: last position of a window)
  build_m1.py CORPUS [NPROC]"""
import glob
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TR = f"{T.OUT}/private/trace3"
out = f"{T.OUT}/rows_m1/{corpus}"
CT = T.CHAIN * T.SEQ
NBC = CT // T.G
HL = (10, 40, 70)
FEATS_M1 = ("m1_topic", "m1_joint", "m1_state", "m2_ent16", "m2_p1_16", "m2_ent_last")
LAM = float(os.environ.get("M1_LAM", "1.0"))
mL = json.load(open("/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"]


def load_ranked(c, fname, key, per_window=False):
    """per-rank arrays in capture window order -> corpus window order [nwin*SEQ, ...] (or [nwin, ...])."""
    parts = {}
    for f in sorted(glob.glob(f"{TR}/windows.r*of*.json")):
        j = json.load(open(f))
        r, W = j["rank"], j["world"]
        a = np.load(f"{TR}/{fname}.r{r}of{W}.npz")[key]
        off = 0
        for name, wins in j["windows"]:
            for k, wi in enumerate(wins):
                if name == c:
                    parts[wi] = a[off + k] if per_window else a[(off + k) * T.SEQ:(off + k + 1) * T.SEQ]
            off += len(wins)
    ks = sorted(parts)
    assert ks == list(range(len(ks))), (c, fname, ks[:4])
    return np.stack([parts[k] for k in ks]) if per_window else np.concatenate([parts[k] for k in ks])


def tema(X, h):
    a = 0.5 ** (1.0 / h)
    E = np.empty(X.shape, np.float64)
    for c0 in range(0, X.shape[0], CT):
        E[c0:c0 + CT] = lfilter([1.0 - a], [1.0, -a], X[c0:c0 + CT].astype(np.float64), axis=0)
    return E


_cache = {}


def topic(c):
    if c not in _cache:
        zs = [load_ranked(c, f"hid_L{L}", "z").astype(np.float32) for L in HL]
        nb = zs[0].shape[0] // T.G
        ends = np.arange(nb) * T.G + T.G - 1
        u = np.hstack([tema(z, h)[ends] for z in zs for h in (16, 64)])
        ent = load_ranked(c, "head", "ent", per_window=True)            # [nwin, SEQ-1]
        p1 = load_ranked(c, "head", "p1", per_window=True)
        pad = np.full((ent.shape[0], 1), np.nan, np.float32)
        ent = np.hstack([ent, pad]).ravel(); p1 = np.hstack([p1, pad]).ravel()
        fill = lambda x: np.where(np.isnan(x), np.nanmean(x), x)        # noqa: E731  (1 position per window)
        m2 = np.stack([tema(fill(ent)[:, None], 16)[ends, 0], tema(fill(p1)[:, None], 16)[ends, 0], ent[ends]], -1)
        _cache[c] = (u.astype(np.float32), m2.astype(np.float32))
    return _cache[c]


def state(d):
    bc, bs = d["bcnt"].astype(np.float64), d["bsal"].astype(np.float64)
    def ema(M, h):
        a = 0.5 ** (T.G / h)
        E = np.empty(M.shape)
        for c0 in range(0, M.shape[0], NBC):
            E[c0:c0 + NBC] = lfilter([1.0], [1.0, -a], M[c0:c0 + NBC], axis=0)
        return E * ((1 - a) / T.G)
    nrm = ema(bs, 256).sum(1) / np.maximum(ema(bc, 256).sum(1), 1e-30)
    nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
    return np.hstack([ema(bs, 32) / nrm, ema(bs, 128) / nrm])


def target(d, L):
    y = d["ysal"].astype(np.float64) / mL[str(L)]
    ye = np.empty_like(y); np.put_along_axis(ye, d["cand"].astype(np.int64), y, 1)
    ye[~d["valid"]] = np.nan
    return ye


def ridge(Xtr, Ytr, Xte):
    mx, sx, my = Xtr.mean(0), Xtr.std(0) + 1e-9, Ytr.mean(0)
    Z = (Xtr - mx) / sx
    W = np.linalg.solve(Z.T @ Z + LAM * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ (Ytr - my))
    return ((Xte - mx) / sx) @ W + my


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    dc = np.load(f"{T.OUT}/rows_bandall/calib-fit/L{L}.npz")
    uc, _ = topic("calib-fit")
    sc, yc = state(dc), target(dc, L)
    ok = np.isfinite(yc).all(1)
    Xs_c = {"m1_topic": uc, "m1_joint": np.hstack([uc, sc]), "m1_state": sc}
    if corpus == "calib-fit":
        d, m2 = dc, topic(corpus)[1]
        par = (np.arange(len(yc)) // NBC) % 2
        P = {}
        for n, X in Xs_c.items():
            P[n] = np.empty(yc.shape)
            for f in (0, 1):
                tr = ok & (par != f)
                P[n][par == f] = ridge(X[tr], yc[tr], X[par == f])
    else:
        d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
        u, m2 = topic(corpus)
        s = state(d)
        Xs = {"m1_topic": u, "m1_joint": np.hstack([u, s]), "m1_state": s}
        P = {n: ridge(Xs_c[n][ok], yc[ok], Xs[n]) for n in Xs}
    cand = d["cand"].astype(np.int64)
    nb = cand.shape[0]
    assert m2.shape[0] == nb and P["m1_topic"].shape[0] == nb
    cols = [np.take_along_axis(P[n], cand, 1) for n in FEATS_M1[:3]]
    cols += [np.repeat(m2[:, i:i + 1], cand.shape[1], 1) for i in range(3)]
    Fa = np.stack(cols, -1).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/L{L}.part.npz", F=Fa)
    os.replace(f"{out}/L{L}.part.npz", f"{out}/L{L}.npz")
    return L, Fa.shape


if __name__ == "__main__":
    topic("calib-fit")
    if corpus != "calib-fit":
        topic(corpus)
    with Pool(nproc) as p:                                   # fork after the topic cache is filled
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
