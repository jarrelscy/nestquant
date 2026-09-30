#!/usr/bin/env python3
"""T32 fp8dec corpus prep for the KLD harness (PRIVATE, CPU): run T33l dec2t32.py on a dec.py output dir with
--world W (T18 contig sharding: whole tasks per rank), then for every corpus write NAME.map.npz for kld_report
--mask-map: dec [nwin, 2048] bool (position is a real decode token: chain position >= dec_start), task [nwin],
chain [nwin] (task index within the corpus), K.  Harness: NQ_CORPUS_DIR=OUT/corpora, Adapt chain=K (one sequence
per task).  Offline: T32_TRACE=OUT/trace T32_CHAIN=K.
  fp8dec_prep.py DEC_DIR TASKS_JSON OUT [--k 2] [--world 8]"""
import argparse
import json
import subprocess
import sys

import numpy as np

SEQ = 2048
ap = argparse.ArgumentParser()
ap.add_argument("dec_dir"); ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--k", type=int, default=2); ap.add_argument("--world", type=int, default=8)
a = ap.parse_args()
D2T = "/home/coder/git/nestquant/threads/33-search/ceiling/fp8dec/dec2t32.py"
subprocess.run([sys.executable, D2T, a.dec_dir, a.tasks, a.out, "--k-default", str(a.k), "--world", str(a.world)],
               check=True)
wins = json.load(open(f"{a.out}/trace/windows.r0of1.json"))["windows"]
for name, _ in wins:
    z = np.load(f"{a.out}/corpora/{name}.dec.npz")
    K = int(z["K"]); nt = len(z["task_id"])
    ids = np.load(f"{a.out}/corpora/{name}.npy")
    assert len(ids) // SEQ == nt * K, (name, len(ids), nt, K)
    pos = np.arange(K * SEQ)
    dec = np.stack([pos >= int(ds) for ds in z["dec_start"]]).reshape(nt * K, SEQ)
    np.savez(f"{a.out}/corpora/{name}.map.npz", dec=dec, task=np.repeat(z["task_id"], K),
             chain=np.repeat(np.arange(nt), K), K=K, names=np.array([name]))
    print(f"{name}: {nt} tasks x K={K} = {nt * K} windows, decode fraction {dec.mean():.3f}, "
          f"tasks/rank {nt / a.world:g}", flush=True)
