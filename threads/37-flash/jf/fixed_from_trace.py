#!/usr/bin/env python3
"""SMOKE / fallback only: provisional fixed set from the trace's routed counts (NOT the REAP boundary-weighted set of
fixed_set37.py).  fixed_set[L] = top-NFIX experts by train-split routed hits, floating_default[L] = next NF.
  fixed_from_trace.py OUT.json   (needs blocks37.py blk for the train split)"""
import json
import sys

import numpy as np

import jlib37 as J

res = dict(schema="nestquant-19-fixed-set-v2", provisional="routed-count top-K from the trace (smoke)",
           K=J.NFIX, fixed_set={}, floating_default={})
for L in J.LAYERS:
    n = np.load(f"{J.OUT}/blk/train/L{L}.npz")["bcnt"].sum(0, dtype=np.int64)
    o = np.lexsort((np.arange(J.NE), -n))
    res["fixed_set"][str(L)] = sorted(int(e) for e in o[:J.NFIX])
    res["floating_default"][str(L)] = [int(e) for e in o[J.NFIX:J.NFIX + J.NF]]
json.dump(res, open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
