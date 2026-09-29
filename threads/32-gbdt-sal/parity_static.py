#!/usr/bin/env python3
"""T32: Adapt equivalents of the static arms (n_float=0 -> fixed 26 only; refresh=chain -> fixed + floating_default
never updated) + salstat diag, CPU check on one chain."""
import sys
import numpy as np
import torch
import t32lib as T
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import quantisers as Q  # noqa: E402

corpus, L = sys.argv[1], int(sys.argv[2])
fixed, fdef = T.serve_sets()
n = T.CHAIN * T.SEQ
ids, w, xn = (a[:n] for a in T.load_layer(L, corpus))
fx = np.zeros(256, bool); fx[fixed[L]] = True
fd = fx.copy(); fd[[e for e in fdef[L] if not fx[e]][:51]] = True
base = dict(lo="/tmp/nestquant/18-e2e/farm_H/nq2", hi="/tmp/nestquant/18-e2e/farm_H_all4", manifest=T.MANIFEST,
            chain="1", salstat="1")
v = w.astype(np.float64) ** 2 * xn[:, None]
for name, kw, S in (("static26", dict(n_float="0"), fx), ("static77", dict(refresh=str(n // 2)), fd),
                    ("ema", {}, None), ("gbdt", dict(predictor="gbdt"), None)):
    q = Q.Adapt(**base, **kw)
    it = torch.from_numpy(ids.astype(np.int64))
    q.act_full = (it, torch.from_numpy(w), torch.from_numpy(xn))
    hi, _ = q.schedule(L, it, T.SEQ, groups=[0] * T.CHAIN)
    d = q.diag[L]
    ok = "" if S is None else f" hi==static {bool((hi.numpy() == S[ids]).all())}"
    print(f"{name:9s} l4 {d['l4_slots'] / d['slots']:.4f} l4_sal {d['l4_sal'] / d['sal_tot']:.4f} "
          f"(direct {float((v * hi.numpy()).sum() / v.sum()):.4f}){ok}", flush=True)
