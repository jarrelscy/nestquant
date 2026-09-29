#!/usr/bin/env python3
"""nqadapt diagnostics from the per-rank result files: eval-token level-4 share of routed slots (adaptive arms from
Adapt.diag, static arms from the harness's l4_slots counter), floating churn per refresh, KLD by position bucket and
(chained arm) by window index within the rank's chain.
    nq_adapt_report.py --tags passS3,passS4,passS1 [--out FILE.json]"""
import argparse, glob, json
import numpy as np

OUT = "/tmp/nestquant/18-e2e/results"
BUCKETS = ((0, 256), (256, 1024), (1024, 2047))        # position p of the predicting token (p = 0..2046)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = {}
    for tag in a.tags.split(","):
        parts = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/{tag}/r*.json"))]
        names = parts[0]["corpora"]
        for cand in parts[0]["results"]:
            key = f"{cand}@{tag}"
            R = rep[key] = {"tag": tag, "shard": parts[0].get("shard", "stride"), "world": parts[0]["world"]}
            # level-4 share of routed slots (eval tokens, this stream's own routing)
            sl = l4 = fx = f0 = 0
            ch = chn = chf = chfn = 0.0
            chains = []
            for p in parts:
                r = p["results"][cand]
                diag = (r.get("extra") or {}).get("diag")
                if diag:
                    for L, d in diag.items():
                        sl += d["slots"]; l4 += d["l4_slots"]; fx += d["fixed_slots"]; f0 += d["float0_slots"]
                        ch += d["churn_sum"]; chn += d["churn_n"]; chf += d["churn_first_sum"]; chfn += d["churn_first_n"]
                    chains += diag[min(diag, key=int)]["chains"] or []
                else:
                    for L, (h, n) in (r["stats"].get("l4_slots") or {}).items():
                        l4 += h; sl += n
            if sl:
                R["l4_share"] = l4 / sl
            if fx:
                R["fixed_share"], R["float0_share_on_this_routing"] = fx / sl, f0 / sl
                R["churn_per_refresh"] = ch / chn
                R["churn_first_refresh"] = chf / chfn
                R["churn_later_refreshes"] = (ch - chf) / (chn - chfn)
                if chains:
                    R["chain_windows"] = {"n": len(chains), "mean": float(np.mean(chains)), "min": min(chains),
                                          "max": max(chains)}
            # KLD by position bucket, overall and per corpus; chained: by window index within the chain
            seq = parts[0]["seq"] - 1
            acc = {g: [[0.0, 0] for _ in BUCKETS] for g in names + ["all"]}
            byw = {}
            for p in parts:
                kl = np.load(f"{OUT}/{tag}/tokkl_{cand}_r{p['rank']}.npy").astype(np.float64).reshape(-1, seq)
                gpw = p["groups_per_window"]
                assert len(gpw) == kl.shape[0]
                pos_in_group = {}
                for w, g in enumerate(gpw):
                    for b, (lo, hi) in enumerate(BUCKETS):
                        s = kl[w, lo:hi]
                        for G in (names[g], "all"):
                            acc[G][b][0] += s.sum(); acc[G][b][1] += len(s)
                    j = pos_in_group.get(g, 0); pos_in_group[g] = j + 1
                    k = byw.setdefault(min(j, 3), [0.0, 0]); k[0] += kl[w].sum(); k[1] += seq
            R["kld_by_position"] = {G: {f"{lo}-{hi - 1 if hi == 2047 else hi - 1}": v[0] / v[1] for (lo, hi), v in
                                        zip(BUCKETS, acc[G]) if v[1]} for G in acc}
            R["kld_by_window_in_rank_block"] = {("3+" if j == 3 else str(j)): v[0] / v[1] for j, v in sorted(byw.items())}
    for k, R in rep.items():
        print(k, json.dumps({x: (round(y, 4) if isinstance(y, float) else y) for x, y in R.items()
                             if x not in ("kld_by_position",)}))
        print("   pos:", {G: {b: round(v, 4) for b, v in d.items()} for G, d in R["kld_by_position"].items()})
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
