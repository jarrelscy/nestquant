#!/usr/bin/env python3
"""aggregate tfphase.py per-layer JSONs: arms at matched upload bytes per token, all tasks and fold split; B refit on
the fit fold (greedy on the R16 nf-grid curves at hm 0.7, sum = nL*77) evaluated on the test fold by nf-interpolation."""
import glob, json, heapq
import numpy as np
J = {d["L"]: d for d in (json.load(open(f)) for f in glob.glob("/tmp/nestquant/33-search/alloc/tfphase/L*.json"))}
Ls = sorted(J); ch = J[Ls[0]]["chains"]
FIT = [ch.index(n) for n in ("fin-saccr-rwa", "formal-crypto", "sound-change-cascade")]
TEST = [ch.index(n) for n in ("embedding-drift-monitor", "freight-dispatch-shift", "pretrain-shard-corruption")]
ALL = FIT + TEST
HMS = sorted({float(k.split("|")[2]) for k in J[Ls[0]]["res"]})
NFG = sorted({int(k.split("|")[1][1:]) for k in J[Ls[0]]["res"] if k.startswith("R16|u")})


def lay(L, key, T):
    r = np.array(J[L]["res"][key])[T]
    return r[:, 0].sum() / max(r[:, 1].sum(), 1e-30), r[:, 2].sum() / max(r[:, 3].sum(), 1)


def at(c, s, tc):
    o = np.argsort(c); return float(np.interp(tc, np.array(c)[o], np.array(s)[o], left=np.nan, right=np.nan))


def arm(R, nm, T, targets):
    c = []; s = []
    for hm in HMS:
        v = np.array([lay(L, f"R{R}|{nm}|{hm}", T) for L in Ls]); s.append(100 * v[:, 0].mean()); c.append(v[:, 1].mean())
    return [at(c, s, t * R / 16) for t in targets], list(zip(HMS, np.round(c, 2), np.round(s, 2)))


def interp_arm(alloc, T, targets, R=16):
    c = []; s = []
    for hm in HMS:
        vs = []
        for L in Ls:
            g = np.array([lay(L, f"R{R}|u{n}|{hm}", T) for n in NFG])
            vs.append((np.interp(alloc[L], NFG, g[:, 0]), np.interp(alloc[L], NFG, g[:, 1])))
        vs = np.array(vs); s.append(100 * vs[:, 0].mean()); c.append(vs[:, 1].mean())
    return [at(c, s, t) for t in targets]


def refit(T, hm=0.7):
    alloc = {L: NFG[0] for L in Ls}; left = len(Ls) * 77 - sum(alloc.values())
    cur = {}
    for L in Ls:
        y = np.array([lay(L, f"R16|u{n}|{hm}", T)[0] for n in NFG]); xs = np.arange(NFG[0], NFG[-1] + 1)
        yi = np.interp(xs, NFG, y); hull = [0]
        for j in range(1, len(xs)):
            hull.append(j)
            while len(hull) >= 3:
                a, b, c = hull[-3:]
                if (yi[b] - yi[a]) * (xs[c] - xs[a]) <= (yi[c] - yi[a]) * (xs[b] - xs[a]): hull.pop(-2)
                else: break
        cur[L] = np.interp(xs, xs[hull], yi[hull])
    h = [(-(cur[L][1] - cur[L][0]), L) for L in Ls]; heapq.heapify(h)
    while left > 0:
        _, L = heapq.heappop(h); alloc[L] += 1; left -= 1; j = alloc[L] - NFG[0]
        if j + 1 < len(cur[L]): heapq.heappush(h, (-(cur[L][j + 1] - cur[L][j]), L))
    return alloc


tg = [2.2, 2.8]
print(f"layers {len(Ls)} (stride 3); targets churn/16tok {tg} (R4 churn = /4); cells = sal-hot")
for T, tn in ((ALL, "all 6 tasks"), (TEST, "test fold (3)")):
    print(f"-- {tn}")
    for R, nm in ((16, "u64"), (16, "u77"), (16, "u90"), (16, "u103"), (16, "u116"), (16, "B"), (4, "u77"), (4, "B")):
        v, raw = arm(R, nm, T, tg)
        print(f"  R{R:<2d} {nm:5s} @2.2 {v[0]:6.2f}  @2.8 {v[1]:6.2f}   raw(hm,churn,sal) {raw}")
aB = {L: J[L]["nf_B"] for L in Ls}; aF = refit(FIT); aU = {L: 77 for L in Ls}
print("-- B refit on fit fold, tested on test fold (nf-interpolated R16 curves)")
for nm, a in (("uniform77", aU), ("B_calib", aB), ("B_tffit", aF)):
    print(f"  {nm:10s} test @2.2/2.8 {np.round(interp_arm(a, TEST, tg), 2)}  fit-fold {np.round(interp_arm(a, FIT, tg), 2)}")
v = np.array([aF[L] for L in Ls]); b = np.array([aB[L] for L in Ls])
print("  B_tffit nf per layer", dict(zip(Ls, v.tolist())))
print("  B_calib nf per layer", dict(zip(Ls, b.tolist())))
print("  corr(B_tffit, B_calib) %.2f" % np.corrcoef(v, b)[0, 1])
