#!/usr/bin/env python3
"""T34 KV-cost table: 4-window KL(teacher||arm) for no-KV (passT34H1+H2), kv-only (passT34K2), kvq serve (passT34K1)."""
import glob, json, sys
import numpy as np
R = "/tmp/nestquant/34-tr3/e2e/results"
def load(tags):
    W = {}
    for t in tags:
        for p in sorted(glob.glob(f"{R}/{t}/r*.json")):
            j = json.load(open(p)); gids = [int(i) for _, m in j["windows"] for i in m]
            for arm, r in j["results"].items():
                for g, k in zip(gids, r["groups"][0]["win_kl"]):
                    W.setdefault(arm, {})[g] = k
    return {a: np.array([w[i] for i in range(4)]) for a, w in W.items() if all(i in w for i in range(4))}
H, KV, KQ = load(["passT34H1", "passT34H2"]), load(["passT34K2"]), load(["passT34K1"])
TR3, TR3N = np.array([.0141, .0537, .0137, .0148]), 0.0395
out = {}
print(f"{'arm':10} {'noKV':>8} {'kv':>8} {'kvq':>8} {'dKV':>8} {'dKVQ':>8} {'kvq rel':>8}  kvq w0..w3                     vs TR3 0.0241")
for a in ["fp8", "serve", "jF77_hm07", "jF128", "jF173", "nq4"]:
    h, k, q = H.get(a), KV.get(a), KQ.get(a)
    f = lambda x: f"{x.mean():8.5f}" if x is not None else "     n/a"
    dd = lambda x: f"{x.mean() - h.mean():+8.5f}" if x is not None and h is not None else "     n/a"
    rel = f"{(q.mean() / h.mean() - 1) * 100:+7.1f}%" if q is not None and h is not None else "     n/a"
    w = " ".join(f"{v:.4f}" for v in q) if q is not None else ""
    vs = f"{q.mean() - TR3.mean():+.5f} ({(q.mean() / TR3.mean() - 1) * 100:+.0f}%)" if q is not None else ""
    print(f"{a:10} {f(h)} {f(k)} {f(q)} {dd(k)} {dd(q)} {rel}  {w}  {vs}")
    out[a] = {k_: (v.tolist() if v is not None else None) for k_, v in (("noKV", h), ("kv", k), ("kvq", q))}
print(f"{'TR3 pub':10} {'':8} {'':8} {TR3.mean():8.5f}  (fp8 KV)  w: " + " ".join(f"{v:.4f}" for v in TR3) + f";  393k nvfp4+rope8 profile {TR3N}")
json.dump(out, open("/tmp/nestquant/34-tr3/report_kvq4.json", "w"), indent=1)
