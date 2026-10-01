#!/usr/bin/env python3
"""T34 teacher-panel report: KL(BF16 teacher || arm) per window from merged nq_e2e r*.json (one or more tags).
4-window table (confirmation-0000..0003, = their README), 64-window mean +- window sd per domain, paired dKLD vs
--ref (mean of window differences +- se), resident bpw, swaps/1k (Adapt diag, as kld_report.py).
  report_teacher.py --results DIR/TAG [--results DIR/TAG2] [--ref tr3] [--out J]"""
import argparse
import glob
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", action="append", required=True)
ap.add_argument("--ref", default="tr3")
ap.add_argument("--map", default="/tmp/nestquant/34-tr3/corpora/bf16conf.map.npz")
ap.add_argument("--out")
a = ap.parse_args()
M = np.load(a.map, allow_pickle=True)
dom = [str(x) for x in M["domain"]]
DOMS = sorted(set(dom))
NQ2, NQ4, TR3 = 2.028, 4.140, 3.4557
BPW = {"fp8": 8.0, "tr3": TR3, "nq2": NQ2, "nq4": NQ4, "nqS182": NQ2 + (NQ4 - NQ2) * 182 / 256, "nqS173": NQ2 + (NQ4 - NQ2) * 173 / 256,
       "serve": NQ2 + (NQ4 - NQ2) * 77 / 256, "k26_v2": NQ2 + (NQ4 - NQ2) * 77 / 256,
       "k0_hm10": NQ2 + (NQ4 - NQ2) * 77 / 256, "jF77_hm07": NQ2 + (NQ4 - NQ2) * 77 / 256,
       "jF77_hm10": NQ2 + (NQ4 - NQ2) * 77 / 256}
for n in (26, 51, 128, 173, 182):
    BPW[f"jF{n}"] = NQ2 + (NQ4 - NQ2) * n / 256
PUB = {"their_tr3_fp8kv": {"0000": 0.0141, "0001": 0.0537, "0002": 0.0137, "0003": 0.0148, "mean4": 0.024105}}

W, SW = {}, {}                      # arm -> {global window: KL}, arm -> swaps/1k
for d in a.results:
    parts = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/r*.json"))]
    assert parts and len({p["world"] for p in parts}) == 1 and len(parts) == parts[0]["world"], d
    for p in parts:
        gids = [int(i) for _, mine in p["windows"] for i in mine]
        for arm, r in p["results"].items():
            wk = r["groups"][0]["win_kl"]
            assert len(wk) == len(gids)
            dst = W.setdefault(arm, {})
            for g_, k_ in zip(gids, wk):          # arm in several groups (fp8 ref): must agree bitwise per window
                assert g_ not in dst or abs(dst[g_] - k_) < 1e-6, (arm, g_, dst.get(g_), k_)
                dst[g_] = k_
    for arm in parts[0]["results"]:
        cs = cn = 0.0
        for p in parts:
            for d_ in p["results"][arm].get("extra", {}).get("diag", {}).values():
                cs += d_.get("churn_sum", 0); cn += d_.get("churn_n", 0)
        if cn:
            SW[arm] = cs / cn * 1000 / 16
rep = {"published": PUB, "arms": {}}
print(f"{'arm':10} {'bpw':>6} {'w0000':>7} {'w0001':>7} {'w0002':>7} {'w0003':>7} {'mean4':>8} | {'mean_n':>8} {'sd':>7} "
      + " ".join(f"{d[:12]:>16}" for d in DOMS) + f" | d_vs_{a.ref}+-se   swaps/1k")
for arm, wk in W.items():
    ids = sorted(wk)
    v = np.array([wk[i] for i in ids]); dm = [dom[i] for i in ids]
    R = {"bpw": BPW.get(arm), "windows": ids, "win": v.tolist(), "swaps_per_1k": SW.get(arm)}
    if ids[:4] == [0, 1, 2, 3]:
        R["w4"] = v[:4].tolist(); R["mean4"] = float(v[:4].mean())
    R["mean_all"] = float(v.mean()); R["sd_all"] = float(v.std(ddof=1)) if len(v) > 1 else None
    R["domains"] = {d: {"n": int(sum(x == d for x in dm)), "mean": float(v[[x == d for x in dm]].mean()),
                        "sd": float(v[[x == d for x in dm]].std(ddof=1)) if sum(x == d for x in dm) > 1 else 0.0}
                    for d in DOMS if d in dm}
    if a.ref in W and sorted(W[a.ref]) == ids:
        dd = v - np.array([W[a.ref][i] for i in ids])
        R["d_ref"] = float(dd.mean()); R["d_ref_se"] = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
    rep["arms"][arm] = R
    bp = f"{R['bpw']:.3f}" if R["bpw"] else "  ?  "
    w4 = " ".join(f"{x:7.4f}" for x in R.get("w4", [np.nan] * 4))
    print(f"{arm:10} {bp:>6} {w4} {R.get('mean4', np.nan):8.5f} | {R['mean_all']:8.5f} {R['sd_all'] or 0:7.4f} "
          + " ".join(f"{R['domains'][d]['mean']:8.5f}+-{R['domains'][d]['sd']:.4f}" for d in R["domains"])
          + (f" | {R['d_ref']:+.5f}+-{R['d_ref_se']:.5f}" if "d_ref" in R else "")
          + (f"  {R['swaps_per_1k']:6.1f}" if R["swaps_per_1k"] is not None else ""))
p = PUB["their_tr3_fp8kv"]
print(f"{'pub tr3':10} {TR3:6.3f} {p['0000']:7.4f} {p['0001']:7.4f} {p['0002']:7.4f} {p['0003']:7.4f} {p['mean4']:8.5f}")
if a.out:
    json.dump(rep, open(a.out, "w"), indent=1)
