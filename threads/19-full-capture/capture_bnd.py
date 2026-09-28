"""Thread 19: boundary backfill for stats versions merged before boundary support (shard 0, old code).

  capture_bnd.py --root R [--layers auto|3,4,..]      per layer whose current stats version (shard 0 only)
        lacks sal.npy: read the kept shard-0 x.bf16, store the boundary rows (bnd19.write_rows) and compute the
        salience/usage sums with a routed forward identical to capture_stats (Teacher bf16 h, bf16 down GEMM).
        sal.npy is added to the (shard-0) version directory atomically and meta.json is rewritten atomically.
  capture_bnd.py --root R --val                         add bnd_think / bnd_end (int8 distance 1..32, 0 = none)
        to every eval/val/layer_L.pt (atomic rewrite).
Holds the per-layer stats flock while working on a layer, so it can run alongside the stage-2 workers.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch

import bnd19
import capture_stats as cs
import nq19
from nq19 import NEXP, OUT


@torch.no_grad()
def backfill(L, root, src, cache):
    sd = f"{root}/stats/L{L}"
    pd = cs.current(sd)
    if pd is None:
        return None
    m = json.load(open(f"{pd}/meta.json"))
    if os.path.exists(f"{pd}/sal.npy"):
        return "done"
    if [s["shard"] for s in m["shards"]] != [0]:
        raise RuntimeError(f"{pd}: backfill only supports shard-0-only versions")
    t0 = time.time()
    acts = m["shards"][0]["acts"]
    S_ = cs.load_shard(acts)
    if S_["meta"]["T"] != m["shards"][0]["T"]:
        raise RuntimeError("row count mismatch")
    bnd19.shard_rows(S_, S_["meta"]["corpus"])
    out = bnd19.write_rows(S_, L, f"{root}/bnd_rows")
    cache.load(L)
    sal = torch.zeros(NEXP, bnd19.NCAT, len(bnd19.SAL_COLS), dtype=torch.float64, device="cuda")
    q = queue.Queue(maxsize=3); stop = threading.Event()
    rl = [(e, 0, S_["X"], S_["rows_all"][S_["offs"][e]:S_["offs"][e + 1]]) for e in range(NEXP)]
    th = threading.Thread(target=cs.prefetch, args=(rl, q, stop), daemon=True); th.start()
    item = q.get()
    for e in range(NEXP):
        te = cs.Teacher(cache.expert(e, dtype=torch.float32))
        while item is not None and item[0] == e:
            _, _, b, buf = item
            xb = buf.cuda(non_blocking=True)
            pp = S_["p_all"][S_["offs"][e] + b:S_["offs"][e] + b + len(xb)].cuda()
            h, _, _ = te.fwd(xb, accurate=False)
            rr = S_["rows_all"][S_["offs"][e] + b:S_["offs"][e] + b + len(xb)]
            bnd19.sal_add(sal, e, pp, te.ynorm(h), S_["cat"][rr].long().cuda())
            item = q.get()
        del te
    stop.set(); th.join(timeout=5)
    sal = sal.cpu().numpy()
    if not np.array_equal(sal[:, 0, 0], np.asarray(json.load(open(f"{pd}/meta.json"))["n_routed"], np.float64)):
        raise RuntimeError("salience routed counts differ from the stats n_routed")
    np.save(f"{out}/sal.npy", sal)
    tmp = f"{pd}/sal.npy.tmp.npy"
    np.save(tmp, sal); os.replace(tmp, f"{pd}/sal.npy")
    m["shards"][0]["bnd_rows"] = out
    m["shards"][0]["corpus"] = S_["meta"]["corpus"]
    m["files"]["sal"] = f"[256, {bnd19.NCAT}, {len(bnd19.SAL_COLS)}] f64 salience/usage sums"
    m.update(sal_categories=bnd19.SAL_CATS, sal_columns=bnd19.SAL_COLS, sal_backfilled=True)
    cs.write_json(f"{pd}/meta.json", m)
    cs.fsync_dir(pd)
    return round(time.time() - t0, 1)


def val_flags(root):
    for L in range(3, 78):
        path = f"{root}/eval/val/layer_{L}.pt"
        if not os.path.exists(path):
            print(json.dumps(dict(layer=L, val="missing")), flush=True); continue
        d = torch.load(path, weights_only=True, mmap=True)
        if "bnd_think" in d:
            continue
        pr = d["protocol"]
        w = pr["windows"]
        if w != list(range(w[0], w[0] + len(w))):
            raise RuntimeError("non-contiguous val windows")
        th, en = bnd19.corpus_flags(pr["corpus"])
        d = dict(d)
        d["bnd_think"] = torch.from_numpy(np.asarray(th[w[0]:w[0] + len(w)]).reshape(-1).astype(np.int8).copy())
        d["bnd_end"] = torch.from_numpy(np.asarray(en[w[0]:w[0] + len(w)]).reshape(-1).astype(np.int8).copy())
        if len(d["bnd_think"]) != d["x"].shape[0]:
            raise RuntimeError("val row count mismatch")
        d["protocol"] = dict(pr, boundary="bnd_think / bnd_end: distance 1..32 to the next non-empty </think> / "
                                           "answer end in the same segment, 0 = none; exclusive (nearer, ties -> end)")
        tmp = path + ".tmp"
        torch.save(d, tmp); os.replace(tmp, path)
        print(json.dumps(dict(layer=L, val="flagged", think=int((d["bnd_think"] > 0).sum()),
                              end=int((d["bnd_end"] > 0).sum()))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=OUT)
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--val", action="store_true")
    a = ap.parse_args()
    if a.val:
        return val_flags(a.root)
    nq19.gpu_cap()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    src = nq19.Src(); cache = nq19.ExpertCache(src)
    layers = range(3, 78) if a.layers == "auto" else [int(v) for v in a.layers.split(",")]
    for L in layers:
        if not os.path.exists(f"{a.root}/stats/L{L}"):
            continue
        lk = cs.LayerLock(f"{a.root}/stats/L{L}.lock")
        if not lk.try_acquire():
            os.close(lk.fd); continue
        try:
            r = backfill(L, a.root, src, cache)
        finally:
            lk.release()
        if r != "done":
            print(json.dumps(dict(layer=L, backfill_s=r)), flush=True)


if __name__ == "__main__":
    main()
