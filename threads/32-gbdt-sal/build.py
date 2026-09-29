#!/usr/bin/env python3
"""T32 build: per layer L3-77 GBDT rows (serve features) + next-64 cnt / sal targets for a traced corpus.
  build.py CORPUS [NPROC]  -> /tmp/nestquant/32-gbdt-sal/rows/CORPUS/L{L}.npz  (PRIVATE)"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import t32lib as T  # noqa: E402

corpus = sys.argv[1]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 38
fixed, _ = T.serve_sets()
band = os.environ.get("T32_BAND", "")            # "all": score every non-fixed expert (rlo 0: nothing forced),
RLO, RHI = {"": (20, 121), "all": (0, 256), "all20": (20, 256)}[band]   # all20: forced top-20 + rest scored
out = f"{T.OUT}/rows{'_band' + band if band else ''}/{corpus}"


def job(L):
    if os.path.exists(f"{out}/L{L}.npz"):
        return L, "exists"
    return T.build_layer(L, corpus, fixed, out, RLO, RHI)


if __name__ == "__main__":
    with Pool(nproc) as p:
        for r in p.imap_unordered(job, T.LAYERS):
            print(r, flush=True)
