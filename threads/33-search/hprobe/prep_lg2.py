#!/usr/bin/env python3
"""interim: pooled router logits per 16-token block from T32 trace2 p_raw (fp16 sigmoid) -> private/lg2/{corpus}/L.npy"""
import os
from multiprocessing import Pool
import numpy as np
import hplib as HL
import t32lib as T
import glob, json


def load_praw(L, corpus):
    parts = {}
    for f in sorted(glob.glob(f"{T.OUT}/trace2/windows.r*of*.json")):
        j = json.load(open(f)); r, W = j["rank"], j["world"]
        a = np.load(f"{T.OUT}/trace2/L{L}.r{r}of{W}.npz")["p_raw"]
        off = 0
        for name, wins in j["windows"]:
            for k, wi in enumerate(wins):
                if name == corpus:
                    parts[wi] = a[(off + k) * T.SEQ:(off + k + 1) * T.SEQ]
            off += len(wins)
    ks = sorted(parts); assert ks == list(range(len(ks)))
    return np.concatenate([parts[k] for k in ks])


def job(L):
    for c in ("calib-fit", "glm52-heldout"):
        p = load_praw(L, c).astype(np.float64).clip(1e-7, 1 - 1e-4)
        lg = np.log(p) - np.log1p(-p)
        os.makedirs(f"{HL.HP}/private/lg2/{c}", exist_ok=True)
        np.save(f"{HL.HP}/private/lg2/{c}/L{L}.npy", lg.reshape(-1, T.G, 256).mean(1).astype(np.float32))
    return L


if __name__ == "__main__":
    with Pool(12) as p:
        print(len(p.map(job, T.LAYERS)))
