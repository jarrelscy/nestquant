"""split rows/NAME/{tr,va}_X.npy into column files rows/NAME/cols/{tr,va}_{col}.npy (fast feature subsets)."""
import os, sys
import numpy as np
import glib as g
R = f"{g.W}/rows/{sys.argv[1]}"
COLS = g.ALLX + ["v2"]
os.makedirs(f"{R}/cols", exist_ok=True)
for k in ("va", "tr"):
    X = np.load(f"{R}/{k}_X.npy", mmap_mode="r")
    n = X.shape[0]
    outs = [np.lib.format.open_memmap(f"{R}/cols/{k}_{c}.npy", "w+", np.float32, (n,)) for c in COLS]
    for s in range(0, n, 4_000_000):
        blk = np.asarray(X[s:s + 4_000_000])
        for j, o in enumerate(outs):
            o[s:s + len(blk)] = blk[:, j]
    for o in outs:
        o.flush()
    print(k, n, flush=True)
