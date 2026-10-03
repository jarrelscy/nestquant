"""T36 driver: arms A-E x SSD {3.5,6.6,11} GB/s x {10,15,20} tok/s at H45 (+ staging-adjusted H for lead arms).
Writes aggregate-only results to results/t36_results.json."""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

LAT = 0.2e-3
HM = 0.7
D = {}


def build(H):
    t = time.time()
    ids, sal, Wj, bs, be = sim.load_all(H, HM, "jF")
    if H != 45:
        return dict(ids=ids, sal=sal, jF=Wj, bs=bs, be=be)
    _, _, Wo, _, _ = sim.load_all(H, HM, "orc")
    Wo1 = Wo.copy()                       # oracle known one block ahead: list k = true top-H of block k+1
    for a, b in zip(bs, be):
        Wo1[:, a:b - 1] = Wo[:, a + 1:b]
    print(f"H{H} built {time.time()-t:.0f}s", flush=True)
    return dict(ids=ids, sal=sal, jF=Wj, orc=Wo, orc1=Wo1, bs=bs, be=be)


def work(cfg):
    d = D[cfg["H"]]
    t = time.time()
    r = sim.run(d["ids"], d["sal"], d[cfg["src"]], d["bs"], d["be"], cfg["H"], cfg["rate"], cfg["bw"] * 1e9, LAT,
                cfg["perfect"], cfg["lead"], cfg["kup"], cfg["upmode"])
    r.update(cfg); r["secs"] = round(time.time() - t, 1)
    print(json.dumps(r), flush=True)
    return r


ARMS = [  # name, src, perfect, lead, kup, upmode
    ("A_perfect", "jF", True, 0, 0.0, 0),
    ("B_real", "jF", False, 0, 0.0, 0),
    ("C_lead1", "jF", False, 1, 0.0, 0),
    ("C_lead2", "jF", False, 2, 0.0, 0),
    ("D_k0.5", "jF", False, 0, 0.5, 1),
    ("D_k1", "jF", False, 0, 1.0, 1),
    ("D_k2", "jF", False, 0, 2.0, 1),
    ("E_orc_blk_perfect", "orc", True, 0, 0.0, 0),
    ("E_orc_blk_lead1", "orc1", False, 1, 0.0, 0),
    ("E_orc_tok_k2", "orc1", False, 1, 0.0, 0),
]

if __name__ == "__main__":
    Hs = [int(x) for x in os.environ.get("HS", "45").split(",")]
    for H in Hs:
        D[H] = build(H)
    cfgs = []
    for H in Hs:
        for bw in (3.5, 6.6, 11.0):
            for rate in (10.0, 15.0, 20.0):
                for name, src, perf, lead, kup, um in ARMS:
                    if name == "E_orc_tok_k2":        # oracle per-token top-up (next 16 tokens) on top of oracle-ahead blocks
                        src, perf, lead, kup, um = "orc", False, 0, 2.0, 2
                    if perf and (bw, rate) != (6.6, 15.0):
                        continue
                    if H != 45 and not name.startswith("C"):
                        continue
                    cfgs.append(dict(arm=name, H=H, bw=bw, rate=rate, src=src, perfect=perf, lead=lead, kup=kup, upmode=um))
    if os.environ.get("ONLY"):
        cfgs = [c for c in cfgs if c["arm"] in os.environ["ONLY"].split(",") and (c["bw"], c["rate"]) == (6.6, 15.0)]
    print(len(cfgs), "configs", flush=True)
    with get_context("fork").Pool(int(os.environ.get("NPROC", "18"))) as p:
        res = p.map(work, cfgs, chunksize=1)
    os.makedirs("results", exist_ok=True)
    out = os.environ.get("OUT", "results/t36_results.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)
