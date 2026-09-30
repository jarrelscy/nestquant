#!/usr/bin/env python3
"""T33l: oracle / split-half ceilings at fixed-set size k (default k=0: 77 floating slots over all 256 experts).
Same serve sim as T33k alloc.sim_seg (start = f26_ranked[k:] + floating_default, sync lag 0, per-chain reset) plus the
additive relative margin dm (x 77th-best score, tie-break to incumbents) and an optional new-expert cap.
Score = all-slot sal-hot (fixed k always hot); churn reported incl. chain resets (= ksweep) and within chains (= seval).
  kzero.py CORPUS [K]     CORPUS in glm52-heldout | calib-fit | sm120tf   -> $OUTC/kzero_k{K}_{CORPUS}.json"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import numba
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/alloc")
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
import clib as C
import alib as A
T = C.T
corpus = sys.argv[1]
KF = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NFL = 77 - KF
DMS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 6.0]
HMS = [0.4, 0.5, 0.6, 0.7, 0.85]
TR_SM = "/tmp/nestquant/32-gbdt-sal/private/trace_sm120"
LITE = corpus == "sm120tf"          # 16x more blocks: coarser grid, essential arms only
if LITE:
    DMS = [0.0, 0.3, 0.8, 1.5, 3.0, 6.0]
    HMS = [0.5, 0.6, 0.7]


@numba.njit(cache=True)
def sim(S, Y, fxm, fd, sgs, sge, nf, hm, dm, cap):
    """-> hot-sal numerator, churn sum incl resets, n transitions incl, churn sum within, n within."""
    NE = S.shape[1]
    want = np.zeros(NE, np.bool_)
    num = 0.0; ci = 0.0; ni = 0; cw = 0.0; nw_ = 0
    prev = np.zeros(NE, np.bool_)
    first = True
    for c in range(sgs.shape[0]):
        want[:] = fd
        for k in range(sgs[c], sge[c]):
            # serve want on block k
            h = 0.0
            for e in range(NE):
                if want[e] or fxm[e]:
                    h += Y[k, e]
            num += h
            if not first:
                n = 0
                for e in range(NE):
                    if want[e] and not prev[e]:
                        n += 1
                ci += n; ni += 1
                if k > sgs[c]:
                    cw += n; nw_ += 1
            first = False
            prev[:] = want
            # decide at end of block k
            v = np.empty(NE)
            pos = 0.0; vmax = 0.0
            for e in range(NE):
                if fxm[e]:
                    v[e] = -np.inf
                else:
                    x = S[k, e]
                    if x > 0:
                        pos += x
                    if x > vmax:
                        vmax = x
                    v[e] = x * (1.0 + hm) if want[e] else x
            if pos <= 0:
                continue                                      # keep incumbents (alib rule)
            srt = np.sort(v)
            v51 = srt[NE - nf]
            sc = max(v51, 1e-6 * vmax, 1e-30)
            for e in range(NE):
                if want[e]:
                    v[e] += dm * sc + 1e-9 * vmax
            o = np.argsort(-v)
            nw = np.zeros(NE, np.bool_)
            for i in range(nf):
                if np.isfinite(v[o[i]]):
                    nw[o[i]] = True
            if cap >= 0:
                nnew = 0
                for e in range(NE):
                    if nw[e] and not want[e]:
                        nnew += 1
                if nnew > cap:
                    nw[:] = False
                    kn = 0
                    for i in range(NE):                     # top-cap new
                        e = o[i]
                        if kn < cap and not want[e] and np.isfinite(v[e]):
                            nw[e] = True; kn += 1
                    ki = 0
                    for i in range(NE):                     # best incumbents fill the rest
                        e = o[i]
                        if ki < nf - cap and want[e]:
                            nw[e] = True; ki += 1
            for e in range(NE):
                want[e] = nw[e] and not fxm[e]
    return num, ci, ni, cw, nw_


TOKBLK = f"{C.OUTC}/cache/sm120tf_tokblk.npy"


def sm120_tokblk():
    """trace row (all captured rows, window order) -> sm120tf block index or -1 (= T32 sm120.prep_tf's dmask/keep/blkrow)."""
    if os.path.exists(TOKBLK):
        return np.load(TOKBLK)
    import sm120 as S1
    mp = np.load(S1.TFC)
    names = [str(x) for x in mp["names"]]
    dms, nseg = [], []
    for ti, n in enumerate(names):
        z = np.load(f"{S1.SRC}/{n}.npz")
        wr = mp["row0"][mp["task"] == ti]
        r = (wr[:, None] + np.arange(T.SEQ)[None]).ravel()
        dm = z["dec"][r]
        dms.append(dm); nseg.append(int(dm.sum()))
    starts = np.r_[0, np.cumsum(nseg)].astype(np.int64)
    dmask = np.concatenate(dms)
    blkrow = np.full(starts[-1], -1, np.int64); nbt = 0
    for r0, nb in S1.blocks_of(starts):
        blkrow[r0:r0 + nb * T.G] = nbt + np.arange(nb * T.G) // T.G; nbt += nb
    tb = np.full(len(dmask), -1, np.int64)
    tb[np.nonzero(dmask)[0]] = blkrow
    os.makedirs(os.path.dirname(TOKBLK), exist_ok=True)
    np.save(TOKBLK, tb)
    return tb


def load_trace_light(L, name, trace):
    """= t32lib.load_layer but each rank's arrays are decompressed once (T.load_layer re-reads per window)."""
    import glob as _g
    parts = {}
    for f in sorted(_g.glob(f"{trace}/windows.r*of*.json")):
        j = json.load(open(f))
        d = np.load(f"{trace}/L{L}.r{j['rank']}of{j['world']}.npz")
        a_ids, a_w, a_xn = d["ids"], d["w"], d["xn"]
        off = 0
        for nm, wins in j["windows"]:
            if nm == name:
                for k, wi in enumerate(wins):
                    s = slice(off + k * T.SEQ, off + (k + 1) * T.SEQ)
                    parts[wi] = (a_ids[s], a_w[s], a_xn[s])
            off += len(wins) * T.SEQ
        del a_ids, a_w, a_xn
    ks = sorted(parts)
    assert ks == list(range(len(ks)))
    return tuple(np.concatenate([parts[k][i] for k in ks]) for i in range(3))


def blocks(ids, v, mask, nb, blk=None):
    b = ((np.arange(ids.shape[0]) // T.G) if blk is None else blk)[:, None].repeat(8, 1)
    idx = b * T.NE + ids.astype(np.int64)
    m = np.repeat(mask[:, None], 8, 1)
    return np.bincount(idx[m], weights=v[m], minlength=nb * T.NE).reshape(nb, T.NE)


def fut(M, n, sgs, sge):
    out = np.zeros(M.shape)
    for s, e in zip(sgs, sge):
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        k = np.arange(e - s)
        out[s:e] = cs[np.minimum(k + 1 + n, e - s)] - cs[np.minimum(k + 1, e - s)]
    return out


def job(L):
    if corpus == "sm120tf":
        import lightgbm as lgb
        import scalelib as SL
        D = SL.load(corpus, L)
        F, _ = SL.feats(D)
        nb = F.shape[0]
        S = lgb.Booster(model_file=C.V2).predict(F.reshape(-1, 9), num_threads=1).reshape(nb, 256)
        del F
        sg = D["sg"]
        ids, w, xn = load_trace_light(L, corpus, TR_SM)
        tb = np.load(TOKBLK)
        keep = tb >= 0
        ids, w, xn, blk = ids[keep], w[keep], xn[keep], tb[keep]
        bs_ref = D["bs"]
    else:
        d = np.load(f"{A.CACHE}/{corpus}/L{L}.npz")
        S, bs_ref = d["S"].astype(np.float64), d["bsal"].astype(np.float64)
        nb = S.shape[0]
        sg = [(c0, min(c0 + A.NBC, nb)) for c0 in range(0, nb, A.NBC)]
        ids, w, xn = T.load_layer(L, corpus)
    if corpus != "sm120tf":
        ids, w, xn = ids[:nb * T.G], w[:nb * T.G], xn[:nb * T.G]
        blk = None
    sgs = np.array([s for s, e in sg], np.int64); sge = np.array([e for s, e in sg], np.int64)
    v = w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]
    Tn = ids.shape[0]
    full = blocks(ids, v, np.ones(Tn, bool), nb, blk)
    err = float(np.abs(full - bs_ref).sum() / bs_ref.sum())
    f26 = A.f26_ranked(L)
    fxm = np.zeros(256, bool); fxm[f26[:KF]] = True
    fd = np.zeros(256, bool); fd[[e for e in f26[KF:] + list(A.fdef[L]) if not fxm[e]][:NFL]] = True
    res = {"_err": err}

    def run(name, Sc, Y, grid="dm", cap=-1):
        tot = Y.sum()
        pts = []
        for x in (DMS if grid == "dm" else HMS):
            hm, dm = (0.0, x) if grid == "dm" else (x, 0.0)
            num, ci, ni, cw, nw = sim(Sc, Y, fxm, fd, sgs, sge, NFL, hm, dm, cap)
            pts.append((num / tot, ci / max(ni, 1), cw / max(nw, 1)))
        res[name] = pts
    run("full/v2_hm", S, full, grid="hm")
    run("full/v2", S, full)
    for n in ((4, 8) if LITE else (1, 2, 4, 8)):
        run(f"full/orc{16 * n}", fut(full, n, sgs, sge), full)
    run("full/orc64_cap3", fut(full, 4, sgs, sge), full, cap=3)
    if not LITE:
        run("full/orc64_cap4", fut(full, 4, sgs, sge), full, cap=4)
    rng = np.random.default_rng(1000 + L)
    for split in ("par", "rnd"):
        mA = (np.arange(Tn) % 2 == 0) if split == "par" else (rng.random(Tn) < 0.5)
        Ab, Bb = blocks(ids, v, mA, nb, blk), blocks(ids, v, ~mA, nb, blk)
        run(f"{split}/v2", S, Bb)
        for n in (4, 8):
            run(f"{split}/orc{16 * n}_A", fut(Ab, n, sgs, sge), Bb)
            run(f"{split}/orc{16 * n}_B", fut(Bb, n, sgs, sge), Bb)
        if not LITE:
            run(f"{split}/orc64_full", fut(full, 4, sgs, sge), Bb)
    return L, res


if __name__ == "__main__":
    if corpus == "sm120tf":
        sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
        print("tokblk", int((sm120_tokblk() >= 0).sum()), flush=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        R = dict(p.map(job, T.LAYERS))
    ci_key = 2 if corpus == "sm120tf" else 1          # churn convention: within-chain for sm120tf (seval), incl resets otherwise
    print(f"{corpus} k={KF} max |block sal trace - rows|/sal = {max(R[L]['_err'] for L in T.LAYERS):.2e}", flush=True)
    out = {}
    for n in [k for k in R[T.LAYERS[0]] if not k.startswith("_")]:
        a = np.array([R[L][n] for L in T.LAYERS]).mean(0)          # ngrid, 3
        s32, flag = C.at_churn([(x[0], x[ci_key]) for x in a], 3.2)
        out[n] = dict(pts=a.tolist(), grid=DMS if not n.endswith("_hm") else HMS, at3p2=s32, flag=flag)
        g0 = a[0]
        print(f"{corpus} k{KF} {n:18s} first {100*g0[0]:6.2f}/{g0[1]:5.2f} (within {g0[2]:5.2f})  @churn3.2 {100*s32:6.2f} {flag}"
              + ("  grid " + " ".join(f"{100*x[0]:.2f}/{x[ci_key]:.2f}" for x in a) if n.endswith(("_hm", "cap3", "cap4")) else ""),
              flush=True)
    json.dump(out, open(f"{C.OUTC}/kzero_k{KF}_{corpus}.json", "w"), indent=1)
