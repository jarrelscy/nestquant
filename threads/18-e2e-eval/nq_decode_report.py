#!/usr/bin/env python3
"""Decode replay arms of one pass: per-corpus KLD / top-1, paired dKLD vs a base arm (se over windows), 4-bit share,
swaps per 1k tokens per layer.  nq_decode_report.py TAG [BASE=gbdt] -> results/TAG_decode.json"""
import json, glob, sys, numpy as np
tag, base = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "gbdt"
R = "/tmp/nestquant/18-e2e/results"
parts = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/{tag}/r*.json"))]
names, seq = parts[0]["corpora"], parts[0]["seq"] - 1
arms = list(parts[0]["results"].keys())
out = {"tag": tag, "base": base, "windows": sum(len(w) for p in parts for _, w in p["windows"]), "arms": {}}
def tok(a, p):
    return np.load(f"{R}/{tag}/tokkl_{a}_r{p['rank']}.npy").astype(np.float64).reshape(-1, seq)
for a in arms:
    A = {"corpora": {}}
    for g, n in enumerate(names):
        kl = sum(p["results"][a]["groups"][g]["klsum"] for p in parts); nt = sum(p["results"][a]["groups"][g]["ntok"] for p in parts)
        ag = sum(p["results"][a]["groups"][g]["agree"] for p in parts)
        A["corpora"][n] = {"KLD": kl / nt, "top1": ag / nt}
    if a != base:
        wd = {g: [] for g in range(len(names))}
        for p in parts:
            ka, kb = tok(base, p), tok(a, p)
            for w, g in enumerate(p["groups_per_window"]):
                wd[g].append(kb[w].mean() - ka[w].mean())
        for g, n in enumerate(names):
            d = np.array(wd[g]); A["corpora"][n].update(dKLD=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)), win_better=float((d < 0).mean()))
    sl = l4 = ch = 0; nl = set()
    for p in parts:
        for L, d in (p["results"][a].get("extra") or {}).get("diag", {}).items():
            sl += d["slots"]; l4 += d["l4_slots"]; ch += d["churn_sum"]; nl.add(L)
    A["l4_share"] = l4 / sl if sl else float("nan"); A["swaps_per_1k_tok_layer"] = 1000 * ch / (sl / 8) if sl else float("nan")
    out["arms"][a] = A
    print(f"{a:14s} l4 {A['l4_share']*100:5.1f}% swaps/1k {A['swaps_per_1k_tok_layer']:6.1f} | " + " | ".join(
        f"{n} KLD {v['KLD']:.5f} top1 {v['top1']*100:.2f}" + (f" d {v['dKLD']:+.5f}±{v['se']:.5f}" if "dKLD" in v else "")
        for n, v in A["corpora"].items()))
json.dump(out, open(f"{R}/{tag}_decode.json", "w"), indent=1)
print("->", f"{R}/{tag}_decode.json")
