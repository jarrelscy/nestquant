"""T37 (GLM-5.3-Flash, 288 experts, L3-44, one shard per layer) copy of T19 fixed_set19.py: K 19, floating 48.
Thread 19: boundary-weighted REAP fixed set (the always-4-bit streaming set), per layer, from sal.npy.

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

import sys; sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
import bnd19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stats", default="stats", help="stats1 (chunk 0 + traces snapshot), stats0 (chunk 0) or stats")
    ap.add_argument("--require", default="stats1",
                    help="all: every planned shard; chunk0 / stats0: the chunk-0 shards; any plan.json snapshot name "
                         "(e.g. stats1 = chunk 0 + traces): exactly its shards")
    ap.add_argument("--K", type=int, default=19)
    ap.add_argument("--weights", default="d1=50,d2_4=20,d5_16=5,d17_32=2")
    ap.add_argument("--out")
    ap.add_argument("--vision-root", help="T26 vision capture root: score = (1-w) S_e/sum S_e + w V/sum V (v3 blend)")
    ap.add_argument("--vision-stats", default="stats")
    ap.add_argument("--vision-w", type=float, default=0.25)
    ap.add_argument("--floating", type=int, default=48, help="floating_default: top-N non-fixed experts by n_routed")
    a = ap.parse_args()
    out_p = a.out or f"{a.root}/fixed_set.json"
    w = {k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(","))}
    assert set(w) == set(bnd19.BUCKET_NAMES), w
    res = dict(fixed_set={}, S_e={}, reap_sum={}, reap_mean={}, n_routed={}, coverage={}, changed_vs_unweighted={},
               stats_version={}, floating_default={})
    if a.vision_root:
        res.update(score={}, vision_sal={}, vision_n_routed={}, vision_stats_version={}, changed_vs_text_only={})
    shards = None
    for L in range(3, 45):
        vd = os.path.realpath(f"{a.root}/{a.stats}/L{L}")
        m = json.load(open(f"{vd}/meta.json"))
        shards = sorted(s["shard"] for s in m["shards"])
        sal = np.load(f"{vd}/sal.npy")
        if not np.array_equal(sal[:, 0, 0], np.asarray(m["n_routed"], np.float64)):
            raise SystemExit(f"L{L}: sal n != n_routed")
        S = sal[:, 0, 4].copy()
        for ki, kind in enumerate(bnd19.KINDS):
            for bi, b in enumerate(bnd19.BUCKET_NAMES):
                S += (w[b] - 1) * sal[:, 1 + 4 * ki + bi, 4]
        score = S
        if a.vision_root:
            vv = os.path.realpath(f"{a.vision_root}/{a.vision_stats}/L{L}")
            vm = json.load(open(f"{vv}/meta.json"))
            vs = np.load(f"{vv}/sal.npy")
            if not np.array_equal(vs[:, 0, 0], np.asarray(vm["n_routed"], np.float64)):
                raise SystemExit(f"L{L}: vision sal n != n_routed")
            V = vs[:, 0, 4]
            score = (1 - a.vision_w) * S / S.sum() + a.vision_w * V / max(V.sum(), 1e-30)
            res["score"][L] = [float(f"{v:.7g}") for v in score]
            res["vision_sal"][L] = [float(f"{v:.7g}") for v in V]
            res["vision_n_routed"][L] = vs[:, 0, 0].astype(int).tolist()
            res["vision_stats_version"][L] = os.path.basename(vv)
            topt = np.lexsort((np.arange(288), -S))[:a.K]
        order = np.lexsort((np.arange(288), -score))
        top = order[:a.K]
        top0 = np.lexsort((np.arange(288), -sal[:, 0, 4]))[:a.K]
        cov = lambda cats: float(sal[top][:, cats, 0].sum() / max(sal[:, cats, 0].sum(), 1))
        res["fixed_set"][L] = sorted(int(e) for e in top)
        res["S_e"][L] = [float(f"{v:.7g}") for v in S]
        res["reap_sum"][L] = [float(f"{v:.7g}") for v in sal[:, 0, 4]]
        res["reap_mean"][L] = [float(f"{v:.7g}") for v in sal[:, 0, 4] / np.maximum(sal[:, 0, 0], 1)]
        res["n_routed"][L] = sal[:, 0, 0].astype(int).tolist()
        res["coverage"][L] = dict(all=cov([0]), think=cov(list(range(1, 5))), end=cov(list(range(5, 9))),
                                   think_d1=cov([1]), end_d1=cov([5]))
        res["changed_vs_unweighted"][L] = len(set(top.tolist()) - set(top0.tolist()))
        if a.vision_root:
            res["changed_vs_text_only"][L] = len(set(top.tolist()) - set(topt.tolist()))
            res["coverage"][L]["vision"] = float(vs[top, 0, 0].sum() / max(vs[:, 0, 0].sum(), 1))
        nr = sal[:, 0, 0].copy()
        nr[top] = -1                                             # non-fixed only
        res["floating_default"][L] = [int(e) for e in np.lexsort((np.arange(288), -nr))[:a.floating]]
        res["stats_version"][L] = os.path.basename(vd)
    doc = dict(schema="nestquant-19-fixed-set-v2", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               root=a.root, stats=a.stats, require=a.require, shards=shards, K=a.K,
               vision=dict(root=a.vision_root, stats=a.vision_stats, w=a.vision_w) if a.vision_root else None,
               weights={f"{k}/{b}": w[b] for k in bnd19.KINDS for b in bnd19.BUCKET_NAMES}, other_tokens_weight=1.0,
               definition="S_e = sum_p_ynorm[all] + sum_c (w_c - 1) sum_p_ynorm[c] over the 8 disjoint boundary "
                          "categories (think/end x d1,d2_4,d5_16,d17_32; distance = positions before the boundary, "
                          "same segment); sum_p_ynorm = sum over routed rows of p * ||bf16 down(h)|| (sal.npy col 4). "
                          "fixed_set = top-K by S_e (ties -> lower id). reap_sum = unweighted sum, "
                          "reap_mean = sum_p_ynorm / n_routed (classic REAP). "
                          + ("With vision: score = (1-w) S_e/sum_e S_e + w V/sum_e V per layer (v3 blend, w = %g), "
                             "V = vision-capture sum_p_ynorm (all rows), fixed_set = top-K by score; changed_vs_text_only "
                             "vs top-K by S_e. " % a.vision_w if a.vision_root else "")
                          + "floating_default = top-%d NON-fixed experts per layer by n_routed on this text capture "
                            "(plain counts, no boundary weighting); seeds the floating 4-bit set on short prompts / "
                            "at boot." % a.floating,
               **res)
    tmp = out_p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, out_p)
    sha = hashlib.sha256(open(out_p, "rb").read()).hexdigest()
    ch = np.array(list(res["changed_vs_unweighted"].values()))
    cv = {k: float(np.mean([c[k] for c in res["coverage"].values()])) for k in ("all", "think", "end", "think_d1", "end_d1", "vision")
          if k in next(iter(res["coverage"].values()))}
    if a.vision_root:
        cv["changed_vs_text_only_mean"] = float(np.mean(list(res["changed_vs_text_only"].values())))
    print(json.dumps(dict(out=out_p, sha256=sha, layers=len(res["fixed_set"]), changed_vs_unweighted_mean=float(ch.mean()),
                          changed_max=int(ch.max()), coverage_mean=cv)))
    if out_p == f"{a.root}/fixed_set.json":
        man_p = f"{a.root}/MANIFEST.json"
        man = json.load(open(man_p)) if os.path.exists(man_p) else {}
        man["fixed_set"] = dict(file="fixed_set.json", sha256=sha, weights=doc["weights"], other_tokens_weight=1.0,
                                K=a.K, stats=a.stats, shards=shards, created_utc=doc["created_utc"], vision=doc["vision"],
                                floating=a.floating,
                                definition=doc["definition"])
        with open(man_p + ".tmp", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(man_p + ".tmp", man_p)


if __name__ == "__main__":
    main()
