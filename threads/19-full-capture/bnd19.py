"""Thread 19 boundary rows (lead's spec 2026-09-28): positions t with 1 <= b - t <= 32 before a boundary b in the
same (window-local) segment, kinds think (non-empty </think>) and end (first token of "<|im_end|>" text, or
<|endoftext|>); a row near both kinds belongs to the nearer (ties -> end).  Buckets d1, d2_4, d5_16, d17_32.

Per (shard k, layer L) the boundary rows themselves are stored (the per-bucket grams are rebuilt on the GPU from
them by nq19_load.Capture.components_bnd; storing 8 bucket gram sets would be ~360 GB/layer):
    OUT/bnd_rows/s{kk}/L{L}/x.bf16   [Nb, 6144] bf16   normalized MoE input (same rows as the stats)
    OUT/bnd_rows/s{kk}/L{L}/rows.npz  row (shard-local fit row), kind (1 think, 2 end), d (1..32), bucket (0..3),
                                      is_ctx (row in the shard's context set ctx_k), ids [Nb, 8] u8, p [Nb, 8] f32
    OUT/bnd_rows/s{kk}/L{L}/sal.npy   this shard's salience sums [256, 9, 6] f64 (see SAL_*)
Category index (salience): 0 = all routed rows, 1 + 4 (kind - 1) + bucket = boundary bucket.
"""
import json
import os
import shutil

import numpy as np
import torch

import nq19

DMAX = 32
BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 32))
BUCKET_NAMES = ("d1", "d2_4", "d5_16", "d17_32")
KINDS = ("think", "end")
NCAT = 1 + len(KINDS) * len(BUCKETS)
SAL_CATS = ["all"] + [f"{k}/{b}" for k in KINDS for b in BUCKET_NAMES]
SAL_COLS = ["n", "sum_p", "sum_p2", "sum_p4", "sum_p_ynorm", "sum_ynorm"]   # REAP saliency = sum_p_ynorm / n
ROWS_DIR = f"{nq19.OUT}/bnd_rows"
OLD_CORPUS_FLAGS = f"{nq19.OUT}/bnd/flags_15m_v2.npz"                      # from bnd_flags.py


def bucket_of(d):
    d = np.asarray(d)
    b = np.full(d.shape, -1, np.int8)
    for i, (lo, hi) in enumerate(BUCKETS):
        b[(d >= lo) & (d <= hi)] = i
    return b


_FLAGS = {}


def corpus_flags(corpus):
    """(think_d, end_d) int8 [windows, C], exclusive (0 = none)."""
    if corpus not in _FLAGS:
        if os.path.exists(f"{corpus}/bnd_think_d.npy"):
            th = np.load(f"{corpus}/bnd_think_d.npy", mmap_mode="r"); en = np.load(f"{corpus}/bnd_end_d.npy", mmap_mode="r")
            th = np.asarray(th, np.int8).copy(); en = np.asarray(en, np.int8).copy()
            both = (th > 0) & (en > 0)                      # enforce exclusivity (nearer wins, ties -> end)
            th[both & (en <= th)] = 0; en[both & (th > 0) & (th < en)] = 0
        elif os.path.realpath(corpus) == os.path.realpath(nq19.CORPUS):
            f = np.load(OLD_CORPUS_FLAGS); th, en = f["bnd_think_d"], f["bnd_end_d"]
        else:
            raise FileNotFoundError(f"no boundary flags for corpus {corpus}")
        _FLAGS[corpus] = (th, en)
    return _FLAGS[corpus]


def row_flags(corpus, first_window, n_windows):
    """Per fit row (window-major) kind (0/1/2), d, cat (0 = none, else 1 + 4 (kind - 1) + bucket)."""
    th, en = corpus_flags(corpus)
    th = np.asarray(th[first_window:first_window + n_windows]).reshape(-1).astype(np.int16)
    en = np.asarray(en[first_window:first_window + n_windows]).reshape(-1).astype(np.int16)
    kind = np.where(th > 0, 1, np.where(en > 0, 2, 0)).astype(np.int8)
    d = np.where(kind == 1, th, en).astype(np.int8)
    b = bucket_of(d)
    cat = np.where(kind > 0, 1 + 4 * (kind.astype(np.int16) - 1) + b, 0).astype(np.int8)
    return kind, d, cat


def shard_rows(S_, corpus):
    """Attach per-row boundary categories to a capture_stats.load_shard() dict (S_['cat'] int8 torch [T])."""
    m = S_["meta"]
    kind, d, cat = row_flags(corpus, m["fit_start"], m["fit_windows"])
    if len(cat) < m["T"]:
        raise ValueError("flags shorter than shard")
    S_["kind"], S_["d"] = kind[:m["T"]], d[:m["T"]]
    S_["cat"] = torch.from_numpy(cat[:m["T"]].copy())
    return S_


def write_rows(S_, L, root=ROWS_DIR):
    """Store the shard's boundary rows for layer L (atomic directory rename)."""
    m = S_["meta"]
    out = f"{root}/s{m['shard']:02d}/L{L}"
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    rows = np.nonzero(S_["kind"] > 0)[0]
    r = torch.from_numpy(rows)
    is_ctx = np.zeros(m["T"], bool); is_ctx[S_["ctx"].numpy()] = True
    S_["X"][r].view(torch.int16).numpy().tofile(f"{tmp}/x.bf16")
    ids, p = S_["ids"], S_["p"]
    np.savez(f"{tmp}/rows.npz", row=rows.astype(np.int64), kind=S_["kind"][rows], d=S_["d"][rows],
             bucket=bucket_of(S_["d"][rows]), is_ctx=is_ctx[rows], ids=ids[rows].numpy().astype(np.uint8),
             p=p[rows].numpy().astype(np.float32))
    with open(f"{tmp}/meta.json", "w") as f:
        json.dump(dict(shard=m["shard"], layer=L, n_rows=int(len(rows)), T=m["T"], fit_start=m["fit_start"],
                       counts={f"{KINDS[k - 1]}/{BUCKET_NAMES[b]}": int(((S_["kind"] == k) & (bucket_of(S_["d"]) == b)).sum())
                               for k in (1, 2) for b in range(4)}), f, indent=1)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


def sal_add(sal, e, pp, yn, cat):
    """sal: f64 cuda [256, NCAT, 6]; pp, yn fp32 [m]; cat int64 cuda [m] (0 = no boundary)."""
    pp = pp.double(); yn = yn.double()
    v = torch.stack([torch.ones_like(pp), pp, pp ** 2, pp ** 4, pp * yn, yn], 1)
    sal[e, 0] += v.sum(0)
    m = cat > 0
    sal[e].index_add_(0, cat[m], v[m])
