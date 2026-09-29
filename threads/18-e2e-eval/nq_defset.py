#!/usr/bin/env python3
"""Snapshot the campaign's default 4-bit set (manifest default_allocation.level4_experts, L3..L77) into one JSON.

    nq_defset.py [--root /tmp/nestquant/nq-encode-v1] [--out FILE] [--extra-layers 3-6]

Output: {"layers": {L: [experts]}, "per_layer_sha256": {L: default_allocation.sha256}, "rules": {rule: [L...]},
         "n_total", "sha256" (of the canonical layers dict), "source", "time"}.
--extra-layers adds every expert of those layers (used for the "L3-6 all at level 4" predecode subset).
"""
import argparse
import hashlib
import json
import os
import time


def topk(a):
    fs = json.load(open(a.score_json))
    layers, shas = {}, {}
    for L in range(a.first, a.last + 1):
        sc = fs["score"][str(L)]
        rank = sorted(range(len(sc)), key=lambda e: (-sc[e], e))
        da = json.load(open(f"{a.root}/L{L}/manifest.json")).get("default_allocation") or {}
        cur = sorted(int(e) for e in da.get("level4_experts") or [])
        assert sorted(rank[:len(cur)]) == cur, (L, "manifest set is not the top of the score ranking")
        layers[str(L)] = sorted(rank[:a.topk])
        shas[str(L)] = da.get("sha256")
    canon = json.dumps(layers, sort_keys=True).encode()
    out = {"layers": layers, "per_layer_sha256": shas, "rules": {f"top {a.topk} per layer by fixed_set.json score "
           f"(blended boundary-weighted REAP, ties -> lower id); nested over the manifest top-{len(cur)}": list(layers)},
           "missing_layers": [], "n_total": sum(len(v) for v in layers.values()), "extra_layers": "",
           "sha256": hashlib.sha256(canon).hexdigest(), "source": a.score_json,
           "score_json_sha256": hashlib.sha256(open(a.score_json, "rb").read()).hexdigest(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "derived_from": None}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.out}: top-{a.topk}, {len(layers)} layers, {out['n_total']} level-4 experts, sha256 {out['sha256'][:16]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/nestquant/nq-encode-v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=3)
    ap.add_argument("--last", type=int, default=77)
    ap.add_argument("--extra-layers", default="")
    ap.add_argument("--from-json", help="start from an earlier snapshot instead of re-reading the manifests")
    ap.add_argument("--topk", type=int, help="extend the ranking instead: top-K per layer by the fixed_set.json 'score' "
                                             "(ties -> lower id); asserts its top-|manifest set| equals the manifest set")
    ap.add_argument("--score-json", default="/tmp/nestquant/nq-encode-v1/_stats/fixed_set.json")
    a = ap.parse_args()
    if a.topk:
        return topk(a)
    layers, shas, rules, missing = {}, {}, {}, []
    if a.from_json:
        d = json.load(open(a.from_json))
        layers, shas, rules, missing = dict(d["layers"]), d["per_layer_sha256"], d["rules"], d["missing_layers"]
        a.first = a.last + 1                       # skip the manifest loop
        out_base = d["sha256"]
    for L in range(a.first, a.last + 1):
        p = f"{a.root}/L{L}/manifest.json"
        if not os.path.exists(p):
            missing.append(L)
            continue
        da = json.load(open(p)).get("default_allocation") or {}
        ex = sorted(int(e) for e in da.get("level4_experts") or [])
        assert da.get("n") in (None, len(ex)), (L, da.get("n"), len(ex))
        layers[str(L)] = ex
        shas[str(L)] = da.get("sha256")
        rules.setdefault(str(da.get("rule"))[:160], []).append(L)
    if a.extra_layers:
        lo, _, hi = a.extra_layers.partition("-")
        for L in range(int(lo), int(hi or lo) + 1):
            layers[str(L)] = list(range(256))
    canon = json.dumps(layers, sort_keys=True).encode()
    out = {"layers": layers, "per_layer_sha256": shas, "rules": rules, "missing_layers": missing,
           "n_total": sum(len(v) for v in layers.values()), "extra_layers": a.extra_layers,
           "sha256": hashlib.sha256(canon).hexdigest(), "source": a.root,
           "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "derived_from": (a.from_json, out_base) if a.from_json else None}
    json.dump(out, open(a.out, "w"), indent=1)
    ns = [len(v) for v in layers.values()]
    print(f"{a.out}: {len(layers)} layers, {out['n_total']} level-4 experts (per layer {min(ns)}-{max(ns)}), "
          f"missing {missing}, sha256 {out['sha256'][:16]}")
    for r, Ls in rules.items():
        print(f"  rule [{len(Ls)} layers]: {r}")


if __name__ == "__main__":
    main()
