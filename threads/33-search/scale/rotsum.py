"""rotating-fold summary: rotsum.py STREAM TAG1,TAG2,.. [ref=v2|0.6] -> per fold + pooled; sal-hot interpolated at the
reference arm's churn (linear in churn over the hm sweep)."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
stream, tags = sys.argv[1], sys.argv[2].split(",")
ref = sys.argv[3] if len(sys.argv) > 3 else "v2|0.6"
def load(tag):
    if "+" in tag:                                   # a+b: merge the key sets of two res dirs (later wins)
        parts = [load(t) for t in tag.split("+")]
        Ls = set.intersection(*[set(p) for p in parts])
        return {L: {k: v for p in parts for k, v in p[L].items()} for L in Ls}
    fs = glob.glob(f"{S.OUT}/private/res/{stream}/{tag}/L*.npz")
    return {int(f.split("/L")[-1][:-4]): dict(np.load(f)) for f in fs}
Rs = {t: load(t) for t in tags}
Ls = sorted(set.intersection(*[set(R) for R in Rs.values()]))
pool = {"pooled": {L: {k: np.concatenate([Rs[t][L][k] for t in tags]) for k in Rs[tags[0]][L]} for L in Ls}}
for t in tags:
    pool[t] = {L: Rs[t][L] for L in Ls}
for name, R in pool.items():
    keys = list(R[Ls[0]])
    res = {k: S.summarize({L: R[L][k] for L in Ls}) for k in keys}
    c0 = res[ref]["churn"]
    arms = sorted({k.split("|")[0] for k in keys if not k.startswith("orc")}, key=lambda a: keys.index(next(k for k in keys if k.startswith(a + "|"))))
    out = []
    for a in arms:
        pts = sorted([(res[k]["churn"], res[k]["sal"], res[k]["cnt"]) for k in keys if k.split("|")[0] == a])
        c, s, h = map(np.array, zip(*pts))
        def ip(y):
            if c.min() <= c0 <= c.max():
                return np.interp(c0, c, y), ""
            j = [0, 1] if c0 < c.min() else [-2, -1]          # linear extrapolation from the two nearest points (*)
            return y[j[0]] + (y[j[1]] - y[j[0]]) * (c0 - c[j[0]]) / (c[j[1]] - c[j[0]]), "*"
        (si, f), (hi, _) = ip(s), ip(h)
        out.append(f"{a} {si:.2f}/{hi:.2f}{f}")
    orc = [f"{k} {res[k]['sal']:.2f}/{res[k]['churn']:.2f}" for k in keys if k.startswith("orc")]
    print(f"[{name}] {len(Ls)} layers, at churn {c0:.2f} ({ref}) sal-hot/hits: " + "  ".join(out) + "  | " + "  ".join(orc))
    for k in keys:
        if not k.startswith("orc"):
            print(f"    {k:14s} {res[k]['sal']:6.2f} {res[k]['cnt']:6.2f} {res[k]['churn']:5.2f}")
