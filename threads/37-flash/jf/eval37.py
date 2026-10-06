#!/usr/bin/env python3
"""T37 offline eval (= T33 sweep.py / pertask.py semantics): sync lag-0 serve replay per layer (jlib37.replay: fixed
set always 4-bit, NF floating chosen by top-NF of S with hysteresis (1 + hm) on residents, chain start =
floating_default), sal-hot = share of salience sum w^2 xn served at 4 bit, churn = new floating experts per block
transition (incl. chain-start resets).  Reports POOLED (salience summed over layers, tracks KLD) and layer-mean,
at matched churn = v2's churn at hm --cref-hm on the same split AND kind.  Decode and prefill chains are reported
separately (decode = primary, top-level numbers; prefill at 16-token refresh and with the set frozen per J37_PF_CHUNK
-token prefill chunk = how prefill would be served; "all" = every chain).
  eval37.py NAME [--splits val,test] [--hms ...] -> $OUT/eval/NAME.json   (sources: static, v2, NAME)"""
import argparse
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import jlib37 as J  # noqa: E402

HMS = [0.0, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 7.0]
# report groups: (name, chain kind or None = all, replay period in blocks).  "decode" = PRIMARY (model selection /
# release numbers); "prefill" = kind-prefill chains at 16-token refresh (upper bound); "prefill_cK" = the same chains
# with the set frozen within K-token chunks (J37_PF_CHUNK 1024: upper bound for short (<1K) follow-up turns only; the
# Mac never refreshes inside a >=1K layer-major prefill).  The PRIMARY prefill-related number is the HANDOFF metric
# (jlib37.handoff: seeded vs cold decode sal-hot over the first 1 / 4 / 16 decode blocks); "all" = every
# chain (hot_eval churn convention incl. chain-start resets; only with J37_EVAL_ALL=1).
def groups():
    g = [("decode", "decode", 1), ("prefill", "prefill", 1)]
    if J.PF_CHUNK > J.G:
        g.append((f"prefill_c{J.PF_CHUNK}", "prefill", J.PF_CHUNK // J.G))
    return g + ([("all", None, 1)] if os.environ.get("J37_EVAL_ALL", "0") == "1" else [])


def job(args):
    sp, L, srcs, hms, hhm = args
    d = J.load(sp, L)
    fx, fd = J.masks(L)
    out = {}
    for name, sd in srcs.items():
        if sd == "static":
            S = np.broadcast_to(fd.astype(np.float32) + 1e-3, d["bsal"].shape)   # never swaps: fixed + floating_default
        else:
            S = np.load(f"{sd}/L{L}.npy").astype(np.float32)
            assert S.shape == d["bsal"].shape, (sd, L, S.shape, d["bsal"].shape)
        for gn, kind, per in groups():
            sel = None if kind is None else J.kind_sel(d, kind)
            if sel is not None and not sel:
                continue
            for hm in (hms if sd != "static" else [0.0]):
                out[(gn, name, hm)] = J.metric(S, d, L, hm=hm, sel=sel, period=per)
        if J.handoff_pairs(d):
            for k, v in J.handoff(S, d, L, hm=hhm).items():
                out[("handoff", name) + k] = v
    return sp, L, out


def summ_handoff(R, srcs):
    """-> {name: {variant: {n: dict(pooled, lmean, chains)}}} (decode sal-hot over the first n blocks after the
    prefill->decode handoff; seeded vs cold), + seeded-minus-cold gains."""
    if not any(k[0] == "handoff" for k in R[J.LAYERS[0]]):
        return None
    out = {}
    for name in srcs:
        o = {}
        for v in ("seeded", "cold"):
            o[v] = {}
            for n in J.HO_WINS:
                r = np.array([R[L][("handoff", name, v, n)] for L in J.LAYERS], np.float64)
                if r[:, 2].min() <= 0:
                    continue
                o[v][str(n)] = dict(pooled=float(r[:, 0].sum() / r[:, 1].sum()),
                                    lmean=float(np.mean(r[:, 0] / np.maximum(r[:, 1], 1e-30))), chains=int(r[0, 2]))
        o["gain"] = {n: o["seeded"][n]["pooled"] - o["cold"][n]["pooled"] for n in o["seeded"] if n in o["cold"]}
        out[name] = o
    return out


def summarise(R, gn, name, hms, agg):
    pts = []
    for hm in hms:
        if (gn, name, hm) not in R[J.LAYERS[0]]:
            continue
        r = np.array([R[L][(gn, name, hm)] for L in J.LAYERS])
        s = r[:, 0].sum() / r[:, 1].sum() if agg == "pooled" else float(np.mean(r[:, 0] / r[:, 1]))
        pts.append(dict(hm=hm, churn=float(np.mean(r[:, 2] / np.maximum(r[:, 3], 1))), sal=float(s)))
    return pts


def at(pts, c):
    x = [p["churn"] for p in sorted(pts, key=lambda p: p["churn"])]
    y = [p["sal"] for p in sorted(pts, key=lambda p: p["churn"])]
    return float(np.interp(c, x, y)) if x and x[0] <= c <= x[-1] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--hms", default=",".join(map(str, HMS)))
    ap.add_argument("--cref-hm", type=float, default=0.7)
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    hms = [float(h) for h in a.hms.split(",")]
    if a.cref_hm not in hms:
        hms = sorted(hms + [a.cref_hm])
    res = {}
    for sp in a.splits.split(","):
        srcs = {"static": "static", "v2": f"{J.OUT}/scores/v2_{sp}", a.name: f"{J.OUT}/scores/{a.name}_{sp}"}
        with Pool(a.nproc) as p:
            R = {L: o for _, L, o in p.map(job, [(sp, L, srcs, hms, a.cref_hm) for L in J.LAYERS])}
        d0 = J.load(sp, J.LAYERS[0])
        byg = {}
        for gn, kind, per in groups():
            if (gn, "v2", a.cref_hm) not in R[J.LAYERS[0]]:
                continue
            out = {}
            for agg in ("pooled", "lmean"):
                cur = {n: summarise(R, gn, n, hms, agg) for n in srcs}
                cref = next(q["churn"] for q in cur["v2"] if q["hm"] == a.cref_hm)
                out[agg] = dict(curves=cur, cref=cref, at_cref={n: at(c, cref) for n, c in cur.items() if n != "static"},
                                at_1p5cref={n: at(c, 1.5 * cref) for n, c in cur.items() if n != "static"},
                                static=cur["static"][0]["sal"])
                if gn == "decode":
                    for n, c in cur.items():
                        print(f"[eval37] {sp:5s} {gn:13s} {agg:6s} {n:10s} "
                              + " ".join(f"{q['churn']:.2f}/{q['sal'] * 100:.2f}" for q in c), flush=True)
                g = out[agg]["at_cref"]
                print(f"[eval37] {sp:5s} {gn:13s} {agg:6s} @churn {cref:.2f} (v2 hm {a.cref_hm}): v2 {g['v2'] * 100:.2f} "
                      f"{a.name} {g[a.name] * 100:.2f} (gain {100 * (g[a.name] - g['v2']):+.2f}) | @{1.5 * cref:.2f}: "
                      f"v2 {out[agg]['at_1p5cref']['v2'] * 100:.2f} {a.name} {out[agg]['at_1p5cref'][a.name] * 100:.2f}"
                      f" | static {out[agg]['static'] * 100:.2f}", flush=True)
            out["per_layer_at_cref_hm"] = {str(L): {n: R[L][(gn, n, a.cref_hm)][0] / R[L][(gn, n, a.cref_hm)][1]
                                                    for n in ("v2", a.name)} for L in J.LAYERS}
            sel = None if kind is None else J.kind_sel(d0, kind)
            out["chains"] = len(d0["sg"]) if sel is None else len(sel)
            out["blocks"] = int(sum(d0["sg"][i][1] - d0["sg"][i][0] for i in (sel if sel is not None else range(len(d0["sg"])))))
            out["refresh_period_blocks"] = per
            byg[gn] = out
        # top level = DECODE (primary: model selection + release numbers); every group under by_kind
        prim = byg["decode"]
        ho = summ_handoff(R, srcs)
        if ho:
            for name, o in ho.items():
                print(f"[eval37] {sp:5s} handoff  {name:8s} first n decode blocks, pooled sal-hot seeded / cold "
                      f"(hm 0 seed, then hm {a.cref_hm}): " + " ".join(
                          f"n{n} {o['seeded'][n]['pooled'] * 100:.2f}/{o['cold'][n]['pooled'] * 100:.2f}"
                          f" ({100 * o['gain'][n]:+.2f})" for n in o["gain"])
                      + f"  [{o['seeded'][str(J.HO_WINS[0])]['chains']} seqs]", flush=True)
        res[sp] = dict(pooled=prim["pooled"], lmean=prim["lmean"], per_layer_at_cref_hm=prim["per_layer_at_cref_hm"],
                       blocks=prim["blocks"], primary="decode", by_kind=byg, handoff=ho)
    os.makedirs(f"{J.OUT}/eval", exist_ok=True)
    json.dump(dict(name=a.name, hms=hms, cref_hm=a.cref_hm, nf=J.NF, layers=J.LAYERS, pf_chunk=J.PF_CHUNK,
                   handoff=dict(prompt_tokens=J.HO_PROMPT, decode_blocks=J.HO_DEC, windows=list(J.HO_WINS),
                                seed_hm=0.0, decode_hm=a.cref_hm), results=res),
              open(f"{J.OUT}/eval/{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
