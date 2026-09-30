#!/usr/bin/env python3
"""T33i: fp8dec (T33l dec.py output) -> joint-model rows for trainh.py --src fp8dec.  PRIVATE, CPU.
Per task: one chain = the whole task sequence (prompt + real decode; state from the request start, t32lib
semantics: 16-token blocks from position 0, think/answer segment from the emitted token), jlib.features over the
chain, target y64 = next-4-block salience / mL.  Stored rows = decode blocks only (block b kept iff its served block
b+1 starts at or after prompt_len), plus ONE lead block (the last prompt block, init row for the replay), per task:
  feat/fp8dec/L{L}.npz   X fp16 [n,256,22] (jlib.INPUTS, v2 last), P f32 (v2), y64 f16, bsal f32, task i32, blk i32
  feat/fp8dec/meta.json  tasks [{id, corpus, split, rows [s,e) into the row arrays, pos0 (token of row s's block)}]
split: fp8dec-heldout / fp8dec-tb21 -> "test" (final only); fp8dec-train -> "val" if sha1(id) % 8 == 0 else "train".
--src sm120tfd (T33l teacher-forced FP8 full-context sm120tf pass, same dec.py layout, coordinator 2026-09-30):
  labels = its fresh routing (NOT the old no-carry trace); split "tf" -> trainh picks per fold (fold's 2 held-out
  sm120tf tasks = test, other 4 = train).  prompt_len 0 there => every block is a row.  Output feat/sm120tfd/.
dump [n] bool: block fully covered by T33l's hid/rlog dump (--hid DIR, meta.json from_n/row0; ctxlib.py).
  fp8blk.py DEC_DIR TASKS_JSON [--hid DEC_DIR/../hid] [--nproc 8]"""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                  # noqa: E402

import jlib as J                                    # noqa: E402
import t32lib as T                                  # noqa: E402  (jlib puts 32-gbdt-sal on sys.path)

OD = f"{J.OUT}/feat/fp8dec"                        # overridden by --src
G = J.G


def seg_task(tok, n):
    """t32lib.seg_of for one request: state after the emitted token tok[t+1]; new request -> think."""
    s = np.zeros(n, np.int8); cur = 0
    for t in range(n):
        nt = tok[t + 1] if t + 1 < len(tok) else -1
        if nt == T.THINK_ID:
            cur = 0
        elif nt == T.ETHINK_ID:
            cur = 1
        s[t] = cur
    return s


def plan(dec_dir, tasks_json, hid_dir=None):
    idx = json.load(open(f"{dec_dir}/index.json"))
    meta = {t["id"]: t for t in json.load(open(tasks_json))} if tasks_json != "-" else {}
    hm = {}
    if hid_dir and os.path.exists(f"{hid_dir}/meta.json"):   # T33l dump: rows valid for from_n <= n < n_dec
        hm = {t["id"]: t for t in json.load(open(f"{hid_dir}/meta.json"))["tasks"]}
    P = []
    for t in idx["tasks"]:
        c = meta[t["id"]].get("corpus", "fp8dec") if meta else "sm120tfd"
        sp = "tf" if c == "sm120tfd" else ("test" if c in ("fp8dec-heldout", "fp8dec-tb21") else
              "val" if int(hashlib.sha1(t["id"].encode()).hexdigest(), 16) % 8 == 0 else "train")
        n = t["prompt_len"] + t["n_dec"]
        nb = n // G
        b0 = max(t["prompt_len"] // G - 1, 0)       # served block b+1 starts >= prompt_len  <=>  b >= prompt_len/G - 1
        b0 = max(b0 - 1, 0)                          # + one lead (init) block
        if nb - 4 <= b0 + 1:                         # need decode rows with a full y64 horizon
            continue
        h = hm.get(t["id"])
        P.append(dict(id=t["id"], corpus=c, split=sp, g=t["g"], off=t["off"], n=n, prompt_len=t["prompt_len"],
                      n_dec=t["n_dec"], b0=b0, nb=nb, from_n=None if h is None else int(h["from_n"]),
                      row0=None if h is None else int(h["row0"]), hg=None if h is None else int(h["g"])))
    return idx["sparse_layers"], P


def dump_rows(p, k):
    """row (block b) has hidden/router-logit dumps for all 16 of its tokens: decode index n = pos - prompt_len of
    block b's tokens in [from_n, n_dec).  Arms are compared on rows with dump == True (identical rows)."""
    if p["from_n"] is None:
        return np.zeros(len(k), bool)
    n_lo = k * G - p["prompt_len"]; n_hi = n_lo + G - 1
    return (n_lo >= p["from_n"]) & (n_hi < p["n_dec"])


def job(args):
    dec_dir, j, L, P = args
    f = f"{OD}/L{L}.npz"
    if os.path.exists(f):
        return L
    D = 1 + max(p["g"] for p in P)
    tok = [np.load(f"{dec_dir}/tok.g{g}.npy", mmap_mode="r") for g in range(D)]
    R = {k: [np.load(f"{dec_dir}/{k}.g{g}.npy", mmap_mode="r") for g in range(D)] for k in ("rid", "rw", "rxn")}
    fx, _ = J.masks(L)
    out = {k: [] for k in ("X", "P", "y64", "bsal", "task", "blk", "dump")}
    for ti, p in enumerate(P):
        g, o, n = p["g"], p["off"], p["nb"] * G
        ids = np.asarray(R["rid"][g][j, o:o + n]); w = np.asarray(R["rw"][g][j, o:o + n]); xn = np.asarray(R["rxn"][g][j, o:o + n])
        tk = np.asarray(tok[g][o:o + n + 1])
        cnt, cnta, nans, sal, segl = T.block_mats(ids, w, xn, seg_task(tk, n))
        d = dict(bcnt=cnt, bcnta=cnta, nans=nans, segl=segl, bsal=sal.astype(np.float32), sg=[(0, p["nb"])])
        F = J.features(d); Pv = J.v2_pred(F, L, fx); X = J.net_inputs(F, Pv)
        y = J.future(sal / J.mL[L], d["sg"], 4)
        k = np.arange(p["b0"], p["nb"] - 4)
        out["X"].append(X[k]); out["P"].append(Pv[k]); out["y64"].append(y[k].astype(np.float16))
        out["bsal"].append(d["bsal"][k]); out["task"].append(np.full(len(k), ti, np.int32)); out["blk"].append(k.astype(np.int32))
        out["dump"].append(dump_rows(p, k))
    np.savez(f + ".part.npz", **{k: np.concatenate(v) for k, v in out.items()})
    os.replace(f + ".part.npz", f)
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dec_dir"); ap.add_argument("tasks")
    ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--hid", default="")
    ap.add_argument("--src", default="fp8dec", choices=["fp8dec", "sm120tfd"], help="sm120tfd: TASKS_JSON = -")
    a = ap.parse_args()
    OD = f"{J.OUT}/feat/{a.src}"
    sp, P = plan(a.dec_dir, a.tasks, a.hid or None)
    os.makedirs(OD, exist_ok=True)
    r0 = 0
    for p in P:
        p["rows"] = [r0, r0 + p["nb"] - 4 - p["b0"]]; r0 = p["rows"][1]; p["pos0"] = p["b0"] * G
    json.dump(dict(src=a.dec_dir, tasks=P, note="PRIVATE fp8dec joint rows; row s of each task = lead (init) block"),
              open(f"{OD}/meta.json", "w"), indent=0)
    print({s: sum(1 for p in P if p["split"] == s) for s in ("train", "val", "test", "tf")}, "rows", r0, flush=True)
    with Pool(a.nproc) as pool:
        for L in pool.imap_unordered(job, [(a.dec_dir, j, L, P) for j, L in enumerate(sp) if L in J.LAYERS]):
            print(L, end=" ", flush=True)
    print("done")
