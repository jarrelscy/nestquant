"""T32: salience-target retrain of the serve's GBDT floating-set predictor (streaming/gbdt_predictor.py).

Features are produced by the SERVE's own code (GBDTPredictor._close_block / _features, block-fed exactly as
step() would feed them token by token), so a model trained here is a drop-in for gbdt_p64_s5.txt.

Stream semantics (= threads/18-e2e-eval quantisers.Adapt._core_gbdt, chain=1, NQ_SHARD=contig):
  a sequence ("chain") = CHAIN consecutive 2048-token windows of one corpus (8192 = the KLD harness's 4 windows/rank/
  corpus); predictor state resets per chain; think/answer segment from the emitted token ids[t+1] (serve step()
  semantics; new_request at chain start).
Rows: one per (block b, candidate = EMA256 rank 20..120 of the non-fixed experts) at the end of block b (features
through token 16(b+1)-1).  Targets over the next H=64 tokens [16(b+1), 16(b+1)+64) (rows whose horizon leaves the
chain are dropped):  cnt = routed hits;  sal = sum_t w_te^2 |x_t|^2 / m_L  (m_L = train mean w^2|x|^2 per routed slot
of layer L: a per-layer constant, rank-neutral, keeps the target in hit units so forced top-20 scores (1e3+) stay on
top).  PRIVATE inputs (traces) stay in /tmp/nestquant/32-gbdt-sal; only tree files may leave the box."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/streaming")
from gbdt_predictor import GBDTPredictor, THINK_ID, ETHINK_ID, G  # noqa: E402

OUT = "/tmp/nestquant/32-gbdt-sal"
TRACE = f"{OUT}/trace"
CORP = "/tmp/nestquant/18-e2e/corpora"
MANIFEST = "/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json"
SEQ, CHAIN, H, NE = 2048, 4, 64, 256
LAYERS = list(range(3, 78))


def serve_sets():
    m = json.load(open(MANIFEST))
    fixed = {int(L): sorted(map(int, v)) for L, v in m["default_allocation"].items()}
    fdef = {int(L): [int(e) for e in v] for L, v in m["floating_default"].items()}
    return fixed, fdef


def load_layer(L, corpus, trace=TRACE):
    """-> ids [T,8] uint8, w [T,8] f32, xn [T] f32 in corpus window order (T = nwin*SEQ)."""
    parts = {}
    for f in sorted(glob.glob(f"{trace}/windows.r*of*.json")):
        j = json.load(open(f))
        r, W = j["rank"], j["world"]
        d = np.load(f"{trace}/L{L}.r{r}of{W}.npz")
        off = 0
        for name, wins in j["windows"]:
            n = len(wins) * SEQ
            if name == corpus:
                for k, wi in enumerate(wins):
                    s = slice(off + k * SEQ, off + (k + 1) * SEQ)
                    parts[wi] = (d["ids"][s], d["w"][s], d["xn"][s])
            off += n
    ks = sorted(parts)
    assert ks == list(range(len(ks))), (corpus, L, ks[:5], len(ks))
    return tuple(np.concatenate([parts[k][i] for k in ks]) for i in range(3))


def tokens(corpus, nwin):
    return np.load(f"{CORP}/{corpus}.npy")[: nwin * SEQ].astype(np.int64)


def seg_of(tok):
    """per position t: segment (0 think, 1 answer) that the serve assigns to step t's counts: state after the
    emitted token ids[t+1] (new request at chain start -> think)."""
    T = len(tok)
    s = np.zeros(T, np.int8)
    for c0 in range(0, T, CHAIN * SEQ):
        cur = 0
        c1 = min(T, c0 + CHAIN * SEQ)
        for t in range(c0, c1):
            nt = tok[t + 1] if t + 1 < T else -1
            if nt == THINK_ID:
                cur = 0
            elif nt == ETHINK_ID:
                cur = 1
            s[t] = cur
    return s


def block_mats(ids, w, xn, seg):
    """per 16-token block: cnt [nb,NE], cnt_answer [nb,NE], n_answer [nb], sal [nb,NE] (raw w^2|x|^2), seg_last."""
    T = ids.shape[0]
    nb = T // G
    b = (np.arange(T) // G)[:, None].repeat(8, 1)
    idx = (b * NE + ids.astype(np.int64)).ravel()
    cnt = np.bincount(idx, minlength=nb * NE).reshape(nb, NE).astype(np.float32)
    a = np.repeat(seg[:, None], 8, 1).ravel().astype(bool)
    cnta = np.bincount(idx[a], minlength=nb * NE).reshape(nb, NE).astype(np.float32)
    v = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]).ravel()
    sal = np.bincount(idx, weights=v, minlength=nb * NE).reshape(nb, NE)
    nans = seg.reshape(nb, G).sum(1)
    seg_last = seg.reshape(nb, G)[:, -1]
    return cnt, cnta, nans, sal, seg_last


def chain_features(L, fixed, cnt, cnta, nans, seg_last, rlo=20, rhi=121):
    """drive GBDTPredictor's own block code over one chain -> X [nb, 101, 5], cand [nb,101], top [nb,20], e256top."""
    P = GBDTPredictor([L], {L: fixed[L]}, mode="sync", num_threads=1, rlo=rlo, rhi=rhi)
    nb = cnt.shape[0]
    Xs, Cs, Ts, Es = [], [], [], []
    for k in range(nb):
        P.bc = cnt[k][None].copy(); P.bca = cnta[k][None].copy(); P.bans = int(nans[k]); P.btok = G
        P.seg = int(seg_last[k])
        P._close_block()
        X, cand, top, e256 = P._features()
        Xs.append(X); Cs.append(cand[0]); Ts.append(top[0]); Es.append(e256[0, top[0]] if top.shape[1] else e256[0, :0])
    P.close()
    return (np.stack(Xs).reshape(nb, -1, 5), np.stack(Cs).astype(np.uint8), np.stack(Ts).astype(np.uint8),
            np.stack(Es).astype(np.float32))


def future_sum(M, nb_chain):
    """M [nb, NE] per block -> F[b] = sum of blocks b+1 .. b+H/G (valid while b + H/G < nb_chain)."""
    k = H // G
    cs = np.zeros((M.shape[0] + 1, M.shape[1]), np.float64)
    cs[1:] = np.cumsum(M, 0)
    nb = M.shape[0]
    F = np.full(M.shape, np.nan)
    b = np.arange(nb - k)
    F[b] = cs[b + 1 + k] - cs[b + 1]
    return F


def build_layer(L, corpus, fixed, out_dir, rlo=20, rhi=121):
    ids, w, xn = load_layer(L, corpus)
    T = ids.shape[0]
    tok = tokens(corpus, T // SEQ)
    seg = seg_of(tok)
    cnt, cnta, nans, sal, seg_last = block_mats(ids, w, xn, seg)
    nbc = CHAIN * SEQ // G
    res = {k: [] for k in ("X", "cand", "top", "e256", "ycnt", "ysal", "valid", "bcnt", "bsal")}
    for c0 in range(0, cnt.shape[0], nbc):
        s = slice(c0, min(c0 + nbc, cnt.shape[0]))
        X, cand, top, e = chain_features(L, fixed, cnt[s], cnta[s], nans[s], seg_last[s], rlo, rhi)
        Fc, Fs = future_sum(cnt[s], cnt[s].shape[0]), future_sum(sal[s], sal[s].shape[0])
        res["X"].append(X); res["cand"].append(cand); res["top"].append(top); res["e256"].append(e)
        res["ycnt"].append(np.take_along_axis(Fc, cand.astype(np.int64), 1).astype(np.float32))
        res["ysal"].append(np.take_along_axis(Fs, cand.astype(np.int64), 1).astype(np.float32))
        v = np.zeros(X.shape[0], bool); v[: X.shape[0] - H // G] = True
        res["valid"].append(v)
        res["bcnt"].append(cnt[s].astype(np.uint8)); res["bsal"].append(sal[s].astype(np.float32))
    res = {k: np.concatenate(v) for k, v in res.items()}
    res["slot_sal_sum"] = np.float64(sal.sum()); res["slots"] = np.int64(T * 8)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/L{L}.npz", **res)
    return L, res["X"].shape


# --------------------------------------------------------------------------------------------- offline sim
def sim_layer(S_blocks, fixed_L, fdef_L, nf=51, hm=0.5, nbc=CHAIN * SEQ // G, lag=1):
    """exact replay of Adapt._core_gbdt (next_refresh, hysteresis) from precomputed per-block score matrices.
    S_blocks [nb, NE] float32 = GBDTPredictor._score output at the end of block b.  -> serve [nb, NE] bool (floating
    set serving block k; fixed excluded)."""
    nb = S_blocks.shape[0]
    fixed = np.zeros(NE, bool); fixed[fixed_L] = True
    fd = np.zeros(NE, bool); fd[[e for e in fdef_L if e not in set(fixed_L)][:nf]] = True
    serve = np.zeros((nb, NE), bool)
    for c0 in range(0, nb, nbc):
        want = fd.copy()
        for k in range(c0, min(c0 + nbc, nb)):
            serve[k] = want
            if k - c0 >= lag:                           # block k closes: apply S of block k-1 (lag 0: sync, S of k)
                S = S_blocks[k - lag]
                v = np.where(fixed, -np.inf, S).astype(np.float32)
                r = want & ~fixed
                v = np.where(r, v * np.float32(1 + hm), v)
                tot = np.where(fixed, 0, np.maximum(S, 0)).sum()
                if tot <= 0:
                    nw = r
                else:
                    nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
                want = nw & ~fixed
    return serve


def score_blocks(pred, cand, top, e256):
    """GBDTPredictor._score for all blocks at once: pred [nb*101] -> S [nb, NE] float32."""
    nb = cand.shape[0]
    S = np.zeros((nb, NE), np.float32)
    np.put_along_axis(S, cand.astype(np.int64), pred.reshape(nb, -1).astype(np.float32), 1)
    np.put_along_axis(S, top.astype(np.int64), (1e3 + e256).astype(np.float32), 1)
    return S


# --------------------------------------------------------------------------------------------- v2 salience features
FEATS_V2 = ("sema32", "sema128", "sal16", "mps128")


def v2_features(bcnt, bsal, cand, nbc=CHAIN * SEQ // G):
    """causal salience features at each block end (same block cadence / decays as the serve's ema32/ema128), reset
    per chain.  Scale-free per layer: salience is divided by the layer's causal salience per routed slot
    (sum_e EMA256 sal / sum_e EMA256 hits), so values are in hit-equivalents like the count features.
      sema32, sema128  per-token EMA rates of normalised w^2|x|^2 (half-life 32 / 128 tokens)
      sal16            normalised salience in the last 16-token block (hits16 analogue)
      mps128           EMA128 salience per hit / layer salience per hit (1.0 when the expert has no recent hits)
    -> [nb, ncand, 4] float32"""
    nb = bcnt.shape[0]
    ag = [0.5 ** (G / h) for h in (32, 128, 256)]
    out = np.zeros((nb, cand.shape[1], 4), np.float32)
    c = bcnt.astype(np.float64); s = bsal.astype(np.float64)
    for c0 in range(0, nb, nbc):
        Es = [np.zeros(NE) for _ in ag]; Ec = [np.zeros(NE) for _ in ag]
        for k in range(c0, min(c0 + nbc, nb)):
            for j, a in enumerate(ag):
                Es[j] = Es[j] * a + s[k]; Ec[j] = Ec[j] * a + c[k]
            norm = Es[2].sum() / max(Ec[2].sum(), 1e-30)
            if norm <= 0:
                norm = 1.0
            ci = cand[k].astype(np.int64)
            out[k, :, 0] = Es[0][ci] * ((1 - ag[0]) / G) / norm
            out[k, :, 1] = Es[1][ci] * ((1 - ag[1]) / G) / norm
            out[k, :, 2] = s[k][ci] / norm
            h = Ec[1][ci]
            out[k, :, 3] = np.where(h > 1e-3, Es[1][ci] / np.maximum(h, 1e-30) / norm, 1.0)
    return out


FEATS5 = ("ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16")
FEATS_V3 = ("pema32", "pema128", "p16", "nmema32", "nmema128", "nm16", "mgema32", "mg16")   # build_v3.py


FEATS_X = ("cx_e32", "cx_e128", "cx_s32", "cx_s128", "co_e32", "co_s32", "ema512", "ema2048", "sema512", "sema2048",
           "tc_word", "tc_num", "tc_code", "tc_punct", "tc_ws", "pos",
           "mtp1_cnt", "mtp1_sal", "mtp2_cnt", "mtp2_sal", "mtp4_cnt", "mtp4_sal")    # build_x.py (band all)


def feature_matrix(names, corpus, L, band="", valid=False, d=None):
    """[rows, len(names)] float32 in the named order from rows / rows_v2 / rows_v3 (all [nb, ncand, k])."""
    sfx = "_band" + band if band else ""
    d = d if d is not None else np.load(f"{OUT}/rows{sfx}/{corpus}/L{L}.npz")
    src = {}
    for i, n in enumerate(FEATS5):
        src[n] = ("X", i)
    for i, n in enumerate(FEATS_V2):
        src[n] = ("X2", i)
    for i, n in enumerate(FEATS_V3):
        src[n] = ("X3", i)
    for i, n in enumerate(FEATS_X):
        src[n] = ("F", i)
    need = {src[n][0] for n in names}
    arr = {"X": d["X"]}
    if "X2" in need:
        arr["X2"] = np.load(f"{OUT}/rows_v2{sfx}/{corpus}/L{L}.npz")["X2"]
    if "X3" in need:
        arr["X3"] = np.load(f"{OUT}/rows_v3{sfx}/{corpus}/L{L}.npz")["X3"]
    if "F" in need:
        assert band == "all", "rows_x only on band all"
        arr["F"] = np.load(f"{OUT}/rows_x/{corpus}/L{L}.npz")["F"]
    cols = [arr[src[n][0]][..., src[n][1]] for n in names]
    M = np.stack(cols, -1)
    if valid:
        M = M[d["valid"]]
    return M.reshape(-1, len(names)).astype(np.float32)
