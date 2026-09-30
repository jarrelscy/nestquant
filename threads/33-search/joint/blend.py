"""Stacking proxy: add the decode-trained GBDT's log-correction to the joint score:
  S = j * (rF / v2)^a   (per layer, all blocks).   python blend.py OUT JDIR RFDIR V2DIR a"""
import os, sys
from multiprocessing import Pool
import numpy as np
out, jd, rd, vd, a = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
os.makedirs(out, exist_ok=True)
def job(f):
    if os.path.exists(f"{out}/{f}"): return
    j = np.load(f"{jd}/{f}").astype(np.float64); r = np.load(f"{rd}/{f}").astype(np.float64); v = np.load(f"{vd}/{f}").astype(np.float64)
    S = np.exp(np.log(np.maximum(j, 1e-30)) + a * (np.log(np.maximum(r, 1e-30)) - np.log(np.maximum(v, 1e-30))))
    np.save(f"{out}/{f}.tmp.npy", S.astype(np.float32)); os.rename(f"{out}/{f}.tmp.npy", f"{out}/{f}")
if __name__ == "__main__":
    with Pool(4) as p: p.map(job, sorted(os.listdir(jd)))
