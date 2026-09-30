#!/usr/bin/env python3
"""T33i jF-R / jF-H context from T33l's fp8dec dumps (hid/ dir, PRIVATE, CPU).  Row set / order = fp8blk.py
(feat/fp8dec/meta.json + L*.npz "dump" flag); only dump rows get context, all arms train/eval on dump rows.
Dump format (T33l 2026-09-30): hid.g{g}.npy [4, n_g*W, 6144] uint16 (bf16 bits) = residual entering layers
10/30/50/70 (pre input_layernorm); rlog.g{g}.npy [75, n_g*W, 256] fp16 = pre-sigmoid router logits (before
e_score_correction_bias; bias.npy [75,256]); row = row0 + n (decode index n = pos - prompt_len), valid from_n <= n < n_dec.
  rlog  -> feat/fp8dec/R{L}.npz  R [nrows, 256, 7] fp16 per expert, from block b's tokens and the 64 before:
           last logit, mean16, max16, mean64, max64, sel-gap last, sel-gap mean16
           (sel-gap = sigmoid(l)+bias - 8th-largest(sigmoid(l)+bias) over the 256 experts at that token)
  hid   -> ctx/hidtab.npy [Ntok, 4, 6144] uint16 (dump tokens of kept tasks, task-contiguous) + ctx/rows.npz
           tend [nrows] (table index of block b's last token; -1 = no dump), tlo [nrows] (task's first table index)
  proj  -> ctx/{rsvd,pca}{k}.npy [Ntok, 4, k] fp16: RMS-normalised residual @ basis (rsvd: router-stack SVD,
           router_svd.py; pca: fit on TRAIN-split tokens only, per source layer)
  ctxlib.py HID_DIR {rlog|hid|proj} [--k 256] [--nproc 8] [--src fp8dec|sm120tfd]   (sm120tfd -> feat/sm120tfd, ctx_sm120tfd;
  proj for sm120tfd reuses the fp8dec PCA basis: fit on fp8dec TRAIN tokens only)"""
import argparse
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                  # noqa: E402

import jlib as J                                    # noqa: E402

FD = f"{J.OUT}/feat/fp8dec"
CD = f"{J.OUT}/ctx"
G = J.G
KR = 7


def rows_meta():
    M = json.load(open(f"{FD}/meta.json"))
    z = np.load(f"{FD}/L{J.LAYERS[0]}.npz")
    return M, z["task"], z["blk"], z["dump"]


def rlog_job(args):
    hid, j, L = args
    f = f"{FD}/R{L}.npz"
    if os.path.exists(f):
        return L
    M, task, blk, dump = rows_meta()
    bias = np.load(f"{hid}/bias.npy")[j].astype(np.float32)
    D = 1 + max(p["hg"] for p in M["tasks"] if p["hg"] is not None)
    RL = [np.load(f"{hid}/rlog.g{g}.npy", mmap_mode="r") for g in range(D)]
    R = np.zeros((len(task), J.NE, KR), np.float16)
    for ti, p in enumerate(M["tasks"]):
        s, e = p["rows"]
        ks = np.nonzero(dump[s:e])[0]
        if p["from_n"] is None or not len(ks):
            continue
        n0, n1 = p["from_n"], p["n_dec"]
        lg = np.asarray(RL[p["hg"]][j, p["row0"] + n0:p["row0"] + n1]).astype(np.float32)     # [n1-n0, 256]
        sc = 1 / (1 + np.exp(-lg)) + bias
        gap = sc - np.partition(sc, -8, axis=1)[:, -8:-7]
        c1 = np.vstack([np.zeros((1, J.NE)), np.cumsum(lg, 0)])
        for r in ks:
            b = blk[s + r]
            hi = b * G + G - 1 - p["prompt_len"] - n0 + 1          # exclusive end (local index)
            lo16, lo64 = hi - G, max(hi - 64, 0)
            R[s + r, :, 0] = lg[hi - 1]; R[s + r, :, 1] = (c1[hi] - c1[lo16]) / G; R[s + r, :, 2] = lg[lo16:hi].max(0)
            R[s + r, :, 3] = (c1[hi] - c1[lo64]) / (hi - lo64); R[s + r, :, 4] = lg[lo64:hi].max(0)
            R[s + r, :, 5] = gap[hi - 1]; R[s + r, :, 6] = gap[lo16:hi].mean(0)
    np.savez(f + ".part.npz", R=R); os.replace(f + ".part.npz", f)
    return L


def build_hid(hid):
    M, task, blk, dump = rows_meta()
    D = 1 + max(p["hg"] for p in M["tasks"] if p["hg"] is not None)
    H = [np.load(f"{hid}/hid.g{g}.npy", mmap_mode="r") for g in range(D)]
    ntok = sum(p["n_dec"] - p["from_n"] for p in M["tasks"] if p["from_n"] is not None and p["n_dec"] > p["from_n"])
    os.makedirs(CD, exist_ok=True)
    T = np.lib.format.open_memmap(f"{CD}/hidtab.npy.part", "w+", np.uint16, (ntok, 4, 6144))
    tend = np.full(len(task), -1, np.int64); tlo = np.full(len(task), -1, np.int64); split = np.zeros(ntok, np.int8)
    o = 0
    for ti, p in enumerate(M["tasks"]):
        if p["from_n"] is None or p["n_dec"] <= p["from_n"]:
            continue
        n0, n1 = p["from_n"], p["n_dec"]
        T[o:o + n1 - n0] = np.asarray(H[p["hg"]][:, p["row0"] + n0:p["row0"] + n1]).transpose(1, 0, 2)
        split[o:o + n1 - n0] = {"train": 0, "val": 1, "test": 2, "tf": 3}[p["split"]]
        s, e = p["rows"]
        ks = np.nonzero(dump[s:e])[0]
        tend[s + ks] = o + blk[s + ks] * G + G - 1 - p["prompt_len"] - n0
        tlo[s + ks] = o
        o += n1 - n0
    T.flush(); del T
    os.replace(f"{CD}/hidtab.npy.part", f"{CD}/hidtab.npy")
    np.savez(f"{CD}/rows.npz", tend=tend, tlo=tlo, split=split)
    print(f"hid table {ntok} tokens, dump rows {(tend >= 0).sum()}", flush=True)


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def build_proj(k):
    T = np.load(f"{CD}/hidtab.npy", mmap_mode="r"); sp = np.load(f"{CD}/rows.npz")["split"]
    V = np.load(f"{J.OUT}/hproj/router_svd.npz")["V"][:, :k]
    tr = np.nonzero(sp == 0)[0]
    pb = f"{J.OUT}/ctx/pca{k}_basis.npz"
    if CD != f"{J.OUT}/ctx":                      # sm120tfd: reuse the fp8dec-train basis (no fold leakage)
        z = np.load(pb); B0 = list(zip(z["mu"], z["U"]))
    # PCA per source layer on train-split tokens (subsample <= 200k), RMS-normalised residual
    rng = np.random.default_rng(0); sub = np.sort(rng.choice(tr, min(len(tr), 200000), replace=False))
    B = []
    for s in (range(4) if CD == f"{J.OUT}/ctx" else []):
        X = bf16(np.asarray(T[sub, s])); X /= np.sqrt((X ** 2).mean(1, keepdims=True)) + 1e-6
        mu = X.mean(0); C = np.cov((X - mu).T)
        ev, U = np.linalg.eigh(C); U = U[:, ::-1][:, :k]
        B.append((mu, U))
    if B:
        np.savez(pb, mu=np.stack([b[0] for b in B]), U=np.stack([b[1] for b in B]))
    else:
        B = B0
    for nm in ("rsvd", "pca"):
        O = np.lib.format.open_memmap(f"{CD}/{nm}{k}.npy.part", "w+", np.float16, (T.shape[0], 4, k))
        for a in range(0, T.shape[0], 65536):
            for s in range(4):
                X = bf16(np.asarray(T[a:a + 65536, s])); X /= np.sqrt((X ** 2).mean(1, keepdims=True)) + 1e-6
                O[a:a + 65536, s] = (X @ V) if nm == "rsvd" else ((X - B[s][0]) @ B[s][1])
        O.flush(); del O
        os.replace(f"{CD}/{nm}{k}.npy.part", f"{CD}/{nm}{k}.npy")
    print("proj done", k, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("hid"); ap.add_argument("what", choices=["rlog", "hid", "proj"])
    ap.add_argument("--k", type=int, default=256); ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--src", default="fp8dec", choices=["fp8dec", "sm120tfd"])
    a = ap.parse_args()
    FD = f"{J.OUT}/feat/{a.src}"; CD = f"{J.OUT}/ctx" + ("" if a.src == "fp8dec" else f"_{a.src}")
    if a.what == "rlog":
        sp = json.load(open(f"{a.hid}/meta.json"))["sparse_layers"]
        with Pool(a.nproc) as pool:
            for L in pool.imap_unordered(rlog_job, [(a.hid, j, L) for j, L in enumerate(sp)]):
                print(L, end=" ", flush=True)
        print("done")
    elif a.what == "hid":
        build_hid(a.hid)
    else:
        build_proj(a.k)
