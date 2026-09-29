#!/usr/bin/env python3
"""Paired KLD difference between two candidates of one pass (same windows, same reference): per corpus
mean(KL_b - KL_a) over tokens, se over windows (window-mean differences), and the share of windows where b < a.
    nq_paired.py --tag passO29_same --a nqdef --b nqdef_l29same"""
import argparse, glob, json
import numpy as np

OUT = "/tmp/nestquant/18-e2e/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    parts = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/{a.tag}/r*.json"))]
    names, seq = parts[0]["corpora"], parts[0]["seq"] - 1
    wd = {g: [] for g in range(len(names))}
    for p in parts:
        ka = np.load(f"{OUT}/{a.tag}/tokkl_{a.a}_r{p['rank']}.npy").astype(np.float64).reshape(-1, seq)
        kb = np.load(f"{OUT}/{a.tag}/tokkl_{a.b}_r{p['rank']}.npy").astype(np.float64).reshape(-1, seq)
        for w, g in enumerate(p["groups_per_window"]):
            wd[g].append((kb[w].mean() - ka[w].mean(), ka[w].mean()))
    out = {}
    print(f"{a.tag}: {a.b} - {a.a}")
    for g, n in enumerate(names):
        d = np.array(wd[g])
        m, se = d[:, 0].mean(), d[:, 0].std(ddof=1) / np.sqrt(len(d))
        out[n] = {"dKLD": m, "se": se, "rel": m / d[:, 1].mean(), "win_better": float((d[:, 0] < 0).mean()),
                  "n_windows": len(d)}
        print(f"  {n:10s} dKLD {m:+.5f} ± {se:.5f} ({100 * m / d[:, 1].mean():+.2f}%)  windows improved "
              f"{100 * (d[:, 0] < 0).mean():.0f}% of {len(d)}")
    json.dump(out, open(f"{OUT}/{a.tag}_paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
