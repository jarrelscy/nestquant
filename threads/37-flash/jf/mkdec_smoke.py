#!/usr/bin/env python3
"""SMOKE ONLY: re-package a (tiny) capture37g trace into the DECODE trace schema that blocks37.py expects from dec37
(seqs.r{r}of{W}.json + tok.r{r}of{W}.npy + L{L}.r{r}of{W}.npz), to exercise the decode code path end to end before
real decode traces exist.  Each 2048-row window becomes one "sequence" with a fake prompt_len; groups pair windows.
NOT training data (teacher-forced calibration routing).
  mkdec_smoke.py SRC_TRACE DST_TRACE [prompt_len=256]"""
import glob
import json
import os
import re
import sys

import numpy as np

import jlib37 as J

src, dst = sys.argv[1], sys.argv[2]
pl = int(sys.argv[3]) if len(sys.argv) > 3 else 256
os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(f"{src}/windows.r*of*.json")):
    r, W = map(int, re.search(r"windows\.r(\d+)of(\d+)\.json$", f).groups())
    j = json.load(open(f))
    toks, seqs = [], []
    for k, (g, w, split) in enumerate(j["wins"]):
        toks.append(np.load(f"{J.GROUP_ROOT[g]}/tokens.npy", mmap_mode="r")[w].astype(np.int32))
        seqs.append(dict(id=f"smoke-{g}-{w}", rows=J.SEQ, prompt_len=pl, group=f"g{(r * 100 + k) // 2}"))
    np.save(f"{dst}/tok.r{r}of{W}.npy", np.concatenate(toks))
    json.dump(dict(N=j["N"], seqs=seqs, note="SMOKE: repackaged capture trace"), open(f"{dst}/seqs.r{r}of{W}.json", "w"))
    for lf in glob.glob(f"{src}/L*.r{r}of{W}.npz"):
        os.symlink(os.path.realpath(lf), f"{dst}/{os.path.basename(lf)}")
print("wrote", dst)
