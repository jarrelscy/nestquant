#!/usr/bin/env python3
"""T32 KLD-pass report: per arm KLD per corpus, paired dKLD vs REF (mean of window differences, se over windows),
harness replay sal-hot / l4 (= routes-hot) / churn from Adapt.diag (salstat=1), pooled over ranks and corpora
(sum over layers; also the per-layer mean), swaps/1k = churn per 16-token refresh x 62.5 (per layer).
Optional decode-only KL (--mask-map map.npz with dec [nwin, 2048]: KL at position t counts if token t+1 is a decode row).
  kld_report.py --results DIR/TAG --ref k26_v2 [--mask-map M | --mask-map NAME=M ...] [--out J]"""
import argparse
import glob
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--ref", default="k26_v2")
ap.add_argument("--mask-map", action="append", default=[],
                help="map.npz (all corpora) or NAME=map.npz (per corpus; repeatable)")
ap.add_argument("--out")
a = ap.parse_args()
parts = sorted((json.load(open(p)) for p in glob.glob(f"{a.results}/r*.json")), key=lambda p: p["rank"])
names, seq = parts[0]["corpora"], parts[0]["seq"] - 1
arms = list(parts[0]["results"])
MM = dict((m.split("=", 1) if "=" in m else ("*", m)) for m in a.mask_map)
DEC = {k: np.load(v)["dec"][:, 1:] for k, v in MM.items()}
rep = {}
for arm in arms:
    R = {}
    tot = {k: 0.0 for k in ("slots", "l4_slots", "sal_tot", "l4_sal", "churn_sum", "churn_n")}
    per = {}
    for p in parts:
        for L, d in p["results"][arm]["extra"]["diag"].items():
            for k in tot:
                tot[k] += d.get(k, 0)
            q = per.setdefault(L, {k: 0.0 for k in tot})
            for k in tot:
                q[k] += d.get(k, 0)
    R["sal_hot"] = 100 * tot["l4_sal"] / tot["sal_tot"]
    R["l4"] = 100 * tot["l4_slots"] / tot["slots"]
    R["sal_hot_layer_mean"] = 100 * float(np.mean([q["l4_sal"] / q["sal_tot"] for q in per.values()]))
    R["churn"] = tot["churn_sum"] / tot["churn_n"]
    R["swaps_per_1k"] = R["churn"] * 1000 / 16
    for g, n in enumerate(names):
        dec = DEC.get(n, DEC.get("*"))
        wd, wdd, kl, kld = [], [], [], []
        for p in parts:
            ka = np.load(f"{a.results}/tokkl_{a.ref}_r{p['rank']}.npy").astype(np.float64).reshape(-1, seq)
            kb = np.load(f"{a.results}/tokkl_{arm}_r{p['rank']}.npy").astype(np.float64).reshape(-1, seq)
            wins = dict(p["windows"])[n] if isinstance(p["windows"], list) else None
            gi = [w for w, gg in enumerate(p["groups_per_window"]) if gg == g]
            for j, w in enumerate(gi):
                wd.append(kb[w].mean() - ka[w].mean()); kl.append(kb[w].mean())
                if dec is not None:
                    m = dec[wins[j]].astype(bool)
                    if m.any():      # windows with no decode positions (all prompt) do not enter the decode-only KL
                        wdd.append(kb[w][m].mean() - ka[w][m].mean()); kld.append(kb[w][m].mean())
        wd = np.array(wd)
        c = dict(kld=float(np.mean(kl)), dkld=float(wd.mean()), se=float(wd.std(ddof=1) / np.sqrt(len(wd))),
                 win_better=float((wd < 0).mean()), nwin=len(wd))
        if dec is not None:
            wdd = np.array(wdd)
            c.update(kld_dec=float(np.mean(kld)), dkld_dec=float(wdd.mean()),
                     se_dec=float(wdd.std(ddof=1) / np.sqrt(len(wdd))))
        R[n] = c
    rep[arm] = R
for arm, R in rep.items():
    s = f"{arm:10s} sal-hot {R['sal_hot']:6.2f} (layer-mean {R['sal_hot_layer_mean']:6.2f})  l4 {R['l4']:5.2f}%  " \
        f"churn {R['churn']:5.2f}  swaps/1k {R['swaps_per_1k']:6.1f}"
    for n in names:
        c = R[n]
        s += f" | {n} KLD {c['kld']:.5f} d {c['dkld']:+.5f}±{c['se']:.5f} ({100 * c['win_better']:.0f}% win)"
        if "kld_dec" in c:
            s += f" dec-only {c['kld_dec']:.5f} d {c['dkld_dec']:+.5f}±{c['se_dec']:.5f}"
    print(s)
if a.out:
    json.dump(rep, open(a.out, "w"), indent=1)
