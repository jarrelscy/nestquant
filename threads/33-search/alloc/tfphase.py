#!/usr/bin/env python3
"""T33k alloc on sm120tf (decode, teacher-forced ref routing): refresh interval R in {16,4} via phase-shifted v2
blocks (o = 0,4,8,12 rows into each task chain), uniform slots vs per-layer allocation B (alloc_B.json, calib-fit
fit) and an nf grid at R16 for the sm120tf fold refit.  Rows = the kept decode rows of sm120.prep_tf (same mapping,
recomputed here); o=0 blocks are checked equal to the prep_tf block arrays.  Per chain (task) sums are stored so any
task subset can be aggregated.   tfphase.py [LAYER_STRIDE]  -> $A/tfphase/L{L}.json"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/scale")
import alib
from alib import T
import sm120 as SM
import scalelib as SL

G, NE = 16, 256
V2 = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
OUTD = f"{alib.A}/tfphase"
HMS = [float(x) for x in os.environ.get("HMS", "0.5,0.7,1.0").split(",")]
NFG = [int(x) for x in os.environ.get("NFG", "64,77,90,103,116").split(",")]
ALLOC = json.load(open(f"{alib.A}/alloc_B.json"))


def mapping():
    mp = np.load(SM.TFC)
    names = [str(x) for x in mp["names"]]
    dms, segs, n_dec = [], [], []
    for ti, n in enumerate(names):
        z = np.load(f"{SM.SRC}/{n}.npz")
        wr = mp["row0"][mp["task"] == ti]
        r = (wr[:, None] + np.arange(T.SEQ)[None]).ravel()
        dm = z["dec"][r]; rd = r[dm]; req = z["req"]
        rst = np.r_[True, req[rd][1:] != req[rd][:-1]]
        dms.append(dm); segs.append(SM.seg_tokens(z["tok"][rd], rst)); n_dec.append(len(rd))
    starts = np.r_[0, np.cumsum(n_dec)].astype(np.int64)
    return names, np.concatenate(dms), np.concatenate(segs), starts


NAMES, DMASK, SEG, STARTS = mapping()


def blocks(ids, v, seg, rows_by_chain):
    """rows_by_chain: list of (r0, nblk) -> D dict for scalelib.feats + sal per block."""
    bcs, bss, bcas, nans, segl, sg, n0 = [], [], [], [], [], [], 0
    for r0, nb in rows_by_chain:
        rr = slice(r0, r0 + nb * G)
        b = np.repeat(np.arange(nb * G) // G, 8); e = ids[rr].astype(np.int64).ravel()
        bcs.append(np.bincount(b * NE + e, minlength=nb * NE).reshape(nb, NE).astype(np.float64))
        bss.append(np.bincount(b * NE + e, weights=v[rr].ravel(), minlength=nb * NE).reshape(nb, NE)
                   .astype(np.float32).astype(np.float64))
        a = np.repeat(seg[rr].astype(bool), 8)
        bcas.append(np.bincount(b[a] * NE + e[a], minlength=nb * NE).reshape(nb, NE).astype(np.float64))
        sk = seg[rr].reshape(nb, G); nans.append(sk.sum(1).astype(np.float64)); segl.append(sk[:, -1].astype(np.int8))
        sg.append((n0, n0 + nb)); n0 += nb
    cat = np.concatenate
    return dict(bc=cat(bcs), bs=cat(bss), bca=cat(bcas), nans=cat(nans), segl=cat(segl), sg=sg)


def job(L):
    import lightgbm as lgb
    bst = lgb.Booster(model_file=V2)
    ids_all, w_all, xn_all = T.load_layer(L, "sm120tf", trace=SM.TFT)
    ids = ids_all[DMASK]; v = w_all[DMASK].astype(np.float64) ** 2 * xn_all[DMASK].astype(np.float64)[:, None]
    cb = SM.blocks_of(STARTS)
    Sph = {}
    for o in (0, 4, 8, 12):
        rbc = [(r0 + o, max(0, (nb * G - o) // G)) for r0, nb in cb]
        D = blocks(ids, v, SEG, rbc)
        if o == 0:
            z = SL.load("sm120tf", L)
            assert np.array_equal(D["bc"], z["bc"]) and np.allclose(D["bs"], z["bs"], rtol=1e-6), "o=0 != prep_tf"
        F, _ = SL.feats(D)
        nb = F.shape[0]
        pr = np.empty(nb * NE, np.float32)
        for c0 in range(0, nb, 8192):
            pr[c0 * NE:min(nb, c0 + 8192) * NE] = bst.predict(F[c0:c0 + 8192].reshape(-1, 9), num_threads=1)
        Sph[o] = (pr.reshape(nb, NE), D["sg"])
        del F
    st = alib.f26_ranked(L) + list(alib.fdef[L])
    cal = alib.load("calib-fit", L)[1].sum(0)
    st = st + [int(e) for e in np.argsort(-cal, kind="stable") if e not in set(st)]
    out = {}
    for R in (16, 4):
        SR, sR, sg, n0 = [], [], [], 0
        for ci, (r0, nbk) in enumerate(cb):
            ln = nbk * G; ns = ln // R
            tt = np.repeat(np.arange(ln) // R, 8)
            sR.append(np.bincount(tt * NE + ids[r0:r0 + ln].astype(np.int64).ravel(), weights=v[r0:r0 + ln].ravel(),
                                  minlength=ns * NE).reshape(ns, NE))
            S = np.zeros((ns, NE), np.float32)
            for o in range(0, G, R):
                P, psg = Sph[o]; s, e = psg[ci]
                for m in range(e - s):
                    S[(o + G * (m + 1)) // R - 1] = P[s + m]
            SR.append(S); sg.append((n0, n0 + ns)); n0 += ns
        S = np.concatenate(SR); s = np.concatenate(sR)
        arms = [(f"u{n}", n) for n in (NFG if R == 16 else [77])] + [("B", ALLOC[str(L)])]
        for nm, nf in arms:
            for hm in HMS:
                sv = alib.sim_seg(S, [], st, nf, hm, sg=sg)
                rec = []
                for (a, b) in sg:
                    if b <= a:
                        rec.append([0.0, 0.0, 0, 0]); continue
                    rec.append([float((s[a:b] * sv[a:b]).sum()), float(s[a:b].sum()),
                                int((sv[a + 1:b] & ~sv[a:b - 1]).sum()), max(b - a - 1, 0)])
                out[f"R{R}|{nm}|{hm}"] = rec
    json.dump(dict(L=L, chains=[NAMES[i] for i, (r0, nb) in enumerate(cb)], nf_B=ALLOC[str(L)], res=out),
              open(f"{OUTD}/L{L}.json", "w"))
    return L


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    Ls = [L for L in T.LAYERS[::stride] if not os.path.exists(f"{OUTD}/L{L}.json")]
    with Pool(int(os.environ.get("NPROC", "15"))) as p:
        for L in p.imap_unordered(job, Ls):
            print("done", L, flush=True)
