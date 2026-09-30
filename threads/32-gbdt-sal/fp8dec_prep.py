#!/usr/bin/env python3
"""T32 fp8dec corpus prep for the KLD harness (PRIVATE, CPU).  Per corpus group (--spec NAME=K:W; corpora not named
use --k/--world), runs T33l dec2t32.py on a filtered view of the dec.py output dir (index.json restricted to the
group's corpora + symlinked arrays) into OUT/g_NAME.../ with --world W (whole tasks per rank when K>1), then
symlinks every corpus into OUT/corpora and writes NAME.map.npz for kld_report --mask-map:
dec [nwin, 2048] bool (chain position >= dec_start = real decode token), task [nwin] (task_id), chain [nwin], K.
Harness: NQ_CORPUS_DIR=OUT/corpora, Adapt chain=map (one sequence per task, any WORLD alignment for K=1).
Offline: T32_TRACE=OUT/<group>/trace  T32_CHAIN=K (printed per corpus).
  fp8dec_prep.py DEC_DIR TASKS_JSON OUT [--k 2] [--world 8] [--spec fp8dec-tb21=1:1 ...]"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

SEQ = 2048
ap = argparse.ArgumentParser()
ap.add_argument("dec_dir"); ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--k", type=int, default=2); ap.add_argument("--world", type=int, default=8)
ap.add_argument("--spec", action="append", default=[], help="NAME=K:W")
a = ap.parse_args()
D2T = "/home/coder/git/nestquant/threads/33-search/ceiling/fp8dec/dec2t32.py"
SP = {s.split("=")[0]: tuple(int(x) for x in s.split("=")[1].split(":")) for s in a.spec}
idx = json.load(open(f"{a.dec_dir}/index.json"))
corp = {t["id"]: t["corpus"] for t in json.load(open(a.tasks))}
groups = defaultdict(list)
for t in idx["tasks"]:
    c = corp[t["id"]]
    groups[SP.get(c, (a.k, a.world))].append(t)
os.makedirs(f"{a.out}/corpora", exist_ok=True)
print(f"src {a.dec_dir} partial={idx.get('partial')} tasks {len(idx['tasks'])}", flush=True)
for (K, W), ts in sorted(groups.items()):
    names = sorted({corp[t["id"]] for t in ts})
    gd = f"{a.out}/g_{'+'.join(names)}"
    view = f"{gd}/src"
    os.makedirs(view, exist_ok=True)
    for f in glob.glob(f"{a.dec_dir}/*.npy"):
        os.symlink(os.path.realpath(f), f"{view}/{os.path.basename(f)}")
    json.dump(dict(idx, tasks=ts), open(f"{view}/index.json", "w"))
    subprocess.run([sys.executable, D2T, view, a.tasks, gd, "--k-default", str(K), "--world", str(W)], check=True)
    for name, _ in json.load(open(f"{gd}/trace/windows.r0of1.json"))["windows"]:
        z = np.load(f"{gd}/corpora/{name}.dec.npz")
        Kc = int(z["K"]); nt = len(z["task_id"])
        n = len(np.load(f"{gd}/corpora/{name}.npy", mmap_mode="r"))
        assert n // SEQ == nt * Kc, (name, n, nt, Kc)
        pos = np.arange(Kc * SEQ)
        dec = np.stack([pos >= int(ds) for ds in z["dec_start"]]).reshape(nt * Kc, SEQ)
        for ext in ("npy", "dec.npz"):
            os.symlink(f"../{os.path.basename(gd)}/corpora/{name}.{ext}", f"{a.out}/corpora/{name}.{ext}")   # relative: OUT may be renamed
        np.savez(f"{a.out}/corpora/{name}.map.npz", dec=dec, task=np.repeat(z["task_id"], Kc),
                 chain=np.repeat(np.arange(nt), Kc), K=Kc, names=np.array([name]))
        print(f"{name}: {nt} tasks x K={Kc} = {nt * Kc} windows (dropped short {len(z['dropped'])}), decode frac "
              f"{dec.mean():.3f}, decode tokens {int(dec.sum())}  offline: T32_TRACE={gd}/trace T32_CHAIN={Kc}",
              flush=True)
