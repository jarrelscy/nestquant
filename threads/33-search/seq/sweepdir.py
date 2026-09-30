"""sweepdir.py CORPUS DIR_OR_V2 [hms]  -> sal-hot/churn per hm (layout via NQ_LAYOUT).  DIR has L{L}.npy or S_*.npy [75,nb,256]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
import lite
corpus, src = sys.argv[1], sys.argv[2]
hms = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.3,0.45,0.6,0.8,1.0").split(",")]
D = f"{lite.WD}/data"
def job(a):
    li, L, hm = a
    cnt = np.load(f"{D}/{corpus}.cnt.npy", mmap_mode="r")[li].astype(np.float32)
    sal = np.load(f"{D}/{corpus}.saln.npy", mmap_mode="r")[li].astype(np.float64)
    if src == "v2":
        S = np.load(f"{D}/{corpus}.v2.npy", mmap_mode="r")[li]
    elif src.endswith(".npy"):
        S = np.load(src, mmap_mode="r")[li]
    else:
        S = np.load(f"{src}/L{L}.npy")
    return lite.metrics(np.asarray(S, np.float32), L, cnt, sal, lite.segs_of(corpus, cnt.shape[0]), hm)
if __name__ == "__main__":
    with Pool(16) as p:
        r = p.map(job, [(li, L, hm) for hm in hms for li, L in enumerate(lite.LAYERS)])
    n = len(lite.LAYERS)
    print(lite.LAYOUT, corpus, os.path.basename(src.rstrip("/")), "  ".join(
        f"hm{hm}: {100*np.mean([x['sal'] for x in r[i*n:(i+1)*n]]):.2f}/{np.mean([x['churn'] for x in r[i*n:(i+1)*n]]):.2f}" for i, hm in enumerate(hms)), flush=True)
