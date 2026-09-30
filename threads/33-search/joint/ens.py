"""ens.py OUTDIR w DIR_A DIR_B [STREAM]: geometric blend S = A^w * B^(1-w) per layer."""
import os, sys
import numpy as np
import jlib as J
od, w, A, B = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(od, exist_ok=True)
for L in J.LAYERS:
    a = np.log(np.maximum(np.load(f"{A}/L{L}.npy").astype(np.float64), 1e-30))
    b = np.log(np.maximum(np.load(f"{B}/L{L}.npy").astype(np.float64), 1e-30))
    np.save(f"{od}/L{L}.npy", np.exp(w * a + (1 - w) * b).astype(np.float32))
