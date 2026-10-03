"""T36 arm E rerun: bandwidth-limited oracle (<= M swaps/layer/block, true block salience), fetched at block start
in place (lead 0) or one block ahead into staging (lead 1).  M_max = floor(0.9 * block_time / per-request time / 75).
Also D with k=0.25.  Appends to results/t36_results.json."""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
import drive

H = 45
D = {}


def work(cfg):
    d = D
    t = time.time()
    if cfg["arm"].startswith("E"):
        M = cfg["M"]
        W = np.stack([sim.oracle_budget(d["ids"][i], d["sal"][i], d["fd"][i], d["bs"], d["be"], H, M)
                      for i in range(len(sim.LAYERS))])
        if cfg["lead"] == 1:
            W1 = W.copy()
            for a, b in zip(d["bs"], d["be"]):
                W1[:, a:b - 1] = W[:, a + 1:b]
            W = W1
    else:
        W = d["jF"]
    r = sim.run(d["ids"], d["sal"], W, d["bs"], d["be"], H, cfg["rate"], cfg["bw"] * 1e9, drive.LAT,
                cfg["perfect"], cfg["lead"], cfg["kup"], cfg["upmode"])
    r.update(cfg); r["secs"] = round(time.time() - t, 1)
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/joint")
    os.environ["LAYOUT"] = "k0"
    import jlib as J
    ids, sal, Wj, bs, be = sim.load_all(H, drive.HM, "jF")
    D.update(ids=ids, sal=sal, jF=Wj, bs=bs, be=be,
             fd=[np.array([int(e) for e in J.FDEF[L]][:H], np.uint8) for L in sim.LAYERS])
    cfgs = []
    for bw in (3.5, 6.6, 11.0):
        for rate in (10.0, 15.0, 20.0):
            mmax = int(0.9 * (1 / rate * 16) / (sim.REC / 2 / (bw * 1e9)) / 75)
            for M in sorted({2, 4, 8, mmax}):
                if M > mmax:
                    continue
                for lead in (0, 1):
                    cfgs.append(dict(arm=f"E_orc_M{M}_lead{lead}", M=M, mmax=mmax, H=H, bw=bw, rate=rate, src="orcM",
                                     perfect=False, lead=lead, kup=0.0, upmode=0))
            cfgs.append(dict(arm="D_k0.25", H=H, bw=bw, rate=rate, src="jF", perfect=False, lead=0, kup=0.25, upmode=1))
    for M in (2, 4, 8, 16):
        cfgs.append(dict(arm=f"E_orc_M{M}_perfect", M=M, H=H, bw=6.6, rate=15.0, src="orcM", perfect=True, lead=0,
                         kup=0.0, upmode=0))
    print(len(cfgs), "configs", flush=True)
    with get_context("fork").Pool(18) as p:
        res = p.map(work, cfgs, chunksize=1)
    R = [r for r in json.load(open("results/t36_results.json")) if r["arm"] not in ("E_orc_blk_lead1", "E_orc_tok_k2")]
    json.dump(R + res, open("results/t36_results.json", "w"), indent=1)
    print("wrote")
