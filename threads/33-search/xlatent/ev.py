#!/usr/bin/env python3
"""ev.py CORPUS SCORE_NAME [hm ...]: sal-hot / churn of D/CORPUS/SCORE_NAME.npy [nb,75,256] (sync, band all)."""
import sys
import numpy as np
import xlib as X
c, n = sys.argv[1], sys.argv[2]
S = X.load(c, n)
for hm in [float(x) for x in sys.argv[3:]] or [0.5]:
    r = X.evaluate(S, c, hm)
    print(f"{c} {n} hm{hm}: sal-hot {r['sal']*100:.2f} churn {r['churn']:.2f}  L3-6 {r['L3_6']*100:.1f} "
          f"L7-40 {r['L7_40']*100:.1f} L41-77 {r['L41_77']*100:.1f}", flush=True)
