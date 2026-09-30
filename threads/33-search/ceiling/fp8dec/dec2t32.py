"""T33l fp8dec: dec.py output (index.json + {tok,rid,rw,rxn}.g{g}.npy) -> T32 trace/corpus format.  PRIVATE.

Per corpus NAME (task["corpus"] in the tasks json), each task contributes ONE chain of K x 2048 tokens: the LAST K*2048
positions of its real sequence (prompt + real decode, i.e. up to and excluding the stop token), so the chain ends at
the chain's last real decode token and the decode part is always inside it.  Tasks shorter than K*2048 are dropped
from that corpus (listed in the sidecar).  --world W trims each corpus to a multiple of W tasks (T18 contig sharding).
  trace   OUT/trace/L{li}.r0of1.npz (ids [T,8] uint8, w [T,8] f32, xn [T] f32) + OUT/trace/windows.r0of1.json
  corpora OUT/corpora/NAME.npy (flat int32, n_tasks*K*2048 (+1 look-ahead token: tok[t+1] of the last position))
          OUT/corpora/NAME.dec.npz  task_id, pool_idx, prompt_len, dec_len, stopped, chain_start (abs position of
          the chain's first token in the task sequence), dec_start (offset of the first decode token within the chain;
          may be 0 when the prompt is shorter than the chain's lead), K
Downstream: T32_TRACE=OUT/trace NQ_CORPUS_DIR=OUT/corpora T32_CHAIN=K.
  dec2t32.py DEC_DIR TASKS_JSON OUT [--k NAME=K ...] [--world 1]"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

SEQ = 2048
ap = argparse.ArgumentParser()
ap.add_argument("dec_dir")
ap.add_argument("tasks")
ap.add_argument("out")
ap.add_argument("--k", action="append", default=[], help="NAME=K windows per task (default --k-default)")
ap.add_argument("--k-default", type=int, default=2)
ap.add_argument("--world", type=int, default=1)
a = ap.parse_args()
KS = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.k}
idx = json.load(open(f"{a.dec_dir}/index.json"))
TK = json.load(open(a.tasks))
meta = {t["id"]: t for t in TK}
sp = idx["sparse_layers"]
D = 1 + max(t["g"] for t in idx["tasks"])
tok = [np.load(f"{a.dec_dir}/tok.g{g}.npy", mmap_mode="r") for g in range(D)]
by = defaultdict(list)
dropped = defaultdict(list)
for t in idx["tasks"]:
    m = meta[t["id"]]
    c = m.get("corpus", "fp8dec")
    K = KS.get(c, a.k_default)
    n = t["prompt_len"] + t["n_dec"]
    if n < K * SEQ:
        dropped[c].append(t["id"]); continue
    by[c].append((t, m, K))
os.makedirs(f"{a.out}/trace", exist_ok=True)
os.makedirs(f"{a.out}/corpora", exist_ok=True)
names = sorted(by)
plan = []                     # (corpus, [(g, abs_lo, abs_hi)]) in trace order
wins = []
w0 = 0
for c in names:
    L = by[c][: len(by[c]) // a.world * a.world]
    K = L[0][2]
    segs, side = [], defaultdict(list)
    for t, m, _ in L:
        n = t["prompt_len"] + t["n_dec"]
        lo = t["off"] + n - K * SEQ
        segs.append((t["g"], lo, t["off"] + n))
        side["task_id"].append(t["id"]); side["pool_idx"].append(m.get("pool_idx", -1))
        side["prompt_len"].append(t["prompt_len"]); side["dec_len"].append(t["n_dec"])
        side["stopped"].append(t.get("stopped", False)); side["chain_start"].append(n - K * SEQ)
        side["dec_start"].append(max(0, t["prompt_len"] - (n - K * SEQ)))
    ids = np.concatenate([tok[g][lo:hi] for g, lo, hi in segs] +
                         [tok[segs[-1][0]][segs[-1][2]:segs[-1][2] + 1]]).astype(np.int32)
    np.save(f"{a.out}/corpora/{c}.npy", ids)
    np.savez(f"{a.out}/corpora/{c}.dec.npz", K=K, dropped=np.array(dropped[c]),
             **{k: np.array(v) for k, v in side.items()})
    nw = len(segs) * K
    wins.append([c, list(range(nw))])
    plan.append((c, segs))
    print(f"{c}: {len(segs)} tasks x K={K} = {nw} windows; dropped short {len(dropped[c])}, trimmed "
          f"{len(by[c]) - len(L)}; decode tokens in chains {sum(side['dec_len'])}", flush=True)
json.dump(dict(rank=0, world=1, windows=wins, src=a.dec_dir, note="PRIVATE fp8dec (dec.py, DSA indexer, KV carry)"),
          open(f"{a.out}/trace/windows.r0of1.json", "w"))
for j, li in enumerate(sp):
    arrs = {k: [np.load(f"{a.dec_dir}/{k}.g{g}.npy", mmap_mode="r") for g in range(D)] for k in ("rid", "rw", "rxn")}
    out = {"ids": [], "w": [], "xn": []}
    for c, segs in plan:
        for g, lo, hi in segs:
            out["ids"].append(np.asarray(arrs["rid"][g][j, lo:hi]))
            out["w"].append(np.asarray(arrs["rw"][g][j, lo:hi]))
            out["xn"].append(np.asarray(arrs["rxn"][g][j, lo:hi]))
    np.savez(f"{a.out}/trace/L{li}.r0of1.npz", ids=np.concatenate(out["ids"]).astype(np.uint8),
             w=np.concatenate(out["w"]).astype(np.float32), xn=np.concatenate(out["xn"]).astype(np.float32))
print(f"wrote {len(sp)} layers -> {a.out}/trace", flush=True)
