"""Thread 19: boundary-weighted REAP fixed set (the always-4-bit streaming set), per layer, from sal.npy.

    fixed_set19.py --root /tmp/nestquant/19-capture-glmfmt [--stats stats1] [--require stats1|chunk0|all] [--K 26]
                   [--weights d1=50,d2_4=20,d5_16=5,d17_32=2] [--out ROOT/fixed_set.json]

S_e = sum_p_ynorm[all] + sum_c (w_c - 1) * sum_p_ynorm[c], with c running over the 8 disjoint boundary categories
(think/end x d1, d2_4, d5_16, d17_32; the same weights for both kinds) and sum_p_ynorm = sal.npy column 4 = sum over
routed rows of p * ||bf16 down(h)||.  This is token-weighted REAP (w_t = w_c for the 32 positions before a boundary,
1 elsewhere), un-normalized so that routing frequency counts.  fixed_set[L] = top-K experts by S_e (ties -> lower id).
Also exported: S_e, the unweighted sum (reap_sum = S_e at w = 1), REAP mean (sum_p_ynorm / n), n_routed, and the
coverage (share of routes landing in the set) on all / think / end / think_d1 / end_d1 rows.
Writes ROOT/fixed_set.json atomically and records it (weights, sha256, stats versions) in ROOT/MANIFEST.json.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

import bnd19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stats", default="stats1", help="stats1 (chunk 0 + traces snapshot), stats0 (chunk 0) or stats")
    ap.add_argument("--require", default="stats1",
                    help="all: every planned shard; chunk0 / stats0: the chunk-0 shards; any plan.json snapshot name "
                         "(e.g. stats1 = chunk 0 + traces): exactly its shards")
    ap.add_argument("--K", type=int, default=26)
    ap.add_argument("--weights", default="d1=50,d2_4=20,d5_16=5,d17_32=2")
    ap.add_argument("--out")
    a = ap.parse_args()
    out_p = a.out or f"{a.root}/fixed_set.json"
    w = {k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(","))}
    assert set(w) == set(bnd19.BUCKET_NAMES), w
    plan = json.load(open(f"{a.root}/plan.json"))
    if a.require == "all":
        need = {int(k) for k in plan["shards"]}
    elif a.require in ("chunk0", "stats0"):
        need = set(plan["chunk0"])
    else:
        need = set(plan["snapshots"][a.require])
    res = dict(fixed_set={}, S_e={}, reap_sum={}, reap_mean={}, n_routed={}, coverage={}, changed_vs_unweighted={},
               stats_version={})
    shards = None
    for L in range(3, 78):
        vd = os.path.realpath(f"{a.root}/{a.stats}/L{L}")
        m = json.load(open(f"{vd}/meta.json"))
        have = {s["shard"] for s in m["shards"]}
        if have != need:
            raise SystemExit(f"L{L}: {os.path.basename(vd)} has shards {sorted(have)}, need {sorted(need)}")
        shards = sorted(have)
        sal = np.load(f"{vd}/sal.npy")
        if not np.array_equal(sal[:, 0, 0], np.asarray(m["n_routed"], np.float64)):
            raise SystemExit(f"L{L}: sal n != n_routed")
        S = sal[:, 0, 4].copy()
        for ki, kind in enumerate(bnd19.KINDS):
            for bi, b in enumerate(bnd19.BUCKET_NAMES):
                S += (w[b] - 1) * sal[:, 1 + 4 * ki + bi, 4]
        order = np.lexsort((np.arange(256), -S))
        top = order[:a.K]
        top0 = np.lexsort((np.arange(256), -sal[:, 0, 4]))[:a.K]
        cov = lambda cats: float(sal[top][:, cats, 0].sum() / max(sal[:, cats, 0].sum(), 1))
        res["fixed_set"][L] = sorted(int(e) for e in top)
        res["S_e"][L] = [float(f"{v:.7g}") for v in S]
        res["reap_sum"][L] = [float(f"{v:.7g}") for v in sal[:, 0, 4]]
        res["reap_mean"][L] = [float(f"{v:.7g}") for v in sal[:, 0, 4] / np.maximum(sal[:, 0, 0], 1)]
        res["n_routed"][L] = sal[:, 0, 0].astype(int).tolist()
        res["coverage"][L] = dict(all=cov([0]), think=cov(list(range(1, 5))), end=cov(list(range(5, 9))),
                                   think_d1=cov([1]), end_d1=cov([5]))
        res["changed_vs_unweighted"][L] = len(set(top.tolist()) - set(top0.tolist()))
        res["stats_version"][L] = os.path.basename(vd)
    doc = dict(schema="nestquant-19-fixed-set-v1", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               root=a.root, stats=a.stats, require=a.require, shards=shards, K=a.K,
               weights={f"{k}/{b}": w[b] for k in bnd19.KINDS for b in bnd19.BUCKET_NAMES}, other_tokens_weight=1.0,
               definition="S_e = sum_p_ynorm[all] + sum_c (w_c - 1) sum_p_ynorm[c] over the 8 disjoint boundary "
                          "categories (think/end x d1,d2_4,d5_16,d17_32; distance = positions before the boundary, "
                          "same segment); sum_p_ynorm = sum over routed rows of p * ||bf16 down(h)|| (sal.npy col 4). "
                          "fixed_set = top-K by S_e (ties -> lower id). reap_sum = unweighted sum, "
                          "reap_mean = sum_p_ynorm / n_routed (classic REAP).",
               **res)
    tmp = out_p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, out_p)
    sha = hashlib.sha256(open(out_p, "rb").read()).hexdigest()
    ch = np.array(list(res["changed_vs_unweighted"].values()))
    cv = {k: float(np.mean([c[k] for c in res["coverage"].values()])) for k in ("all", "think", "end", "think_d1", "end_d1")}
    print(json.dumps(dict(out=out_p, sha256=sha, layers=len(res["fixed_set"]), changed_vs_unweighted_mean=float(ch.mean()),
                          changed_max=int(ch.max()), coverage_mean=cv)))
    if out_p == f"{a.root}/fixed_set.json":
        man_p = f"{a.root}/MANIFEST.json"
        man = json.load(open(man_p)) if os.path.exists(man_p) else {}
        man["fixed_set"] = dict(file="fixed_set.json", sha256=sha, weights=doc["weights"], other_tokens_weight=1.0,
                                K=a.K, stats=a.stats, shards=shards, created_utc=doc["created_utc"],
                                definition=doc["definition"])
        with open(man_p + ".tmp", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(man_p + ".tmp", man_p)


if __name__ == "__main__":
    main()
