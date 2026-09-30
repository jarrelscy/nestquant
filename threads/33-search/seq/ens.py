"""ens.py CORPUS OUT w1:src1 w2:src2 ...  geometric-mean ensemble (log-score weighted sum, per-layer scale-normalised)."""
import os, sys
import numpy as np
import lite
corpus, out = sys.argv[1], sys.argv[2]
parts = [(float(x.split(":", 1)[0]), x.split(":", 1)[1]) for x in sys.argv[3:]]
os.makedirs(out, exist_ok=True)
D = f"{lite.WD}/data"
def get(src, li, L):
    if src == "v2":
        return np.load(f"{D}/{corpus}.v2.npy", mmap_mode="r")[li]
    if src.endswith(".npy"):
        return np.load(src, mmap_mode="r")[li]
    return np.load(f"{src}/L{L}.npy")
for li, L in enumerate(lite.LAYERS):
    z = 0
    for w, src in parts:
        S = np.asarray(get(src, li, L), np.float64)
        S = S / max(S.mean(), 1e-12)
        z = z + w * np.log(np.maximum(S, 1e-6))
    np.save(f"{out}/L{L}.npy", np.exp(z).astype(np.float32))
