# parity: Adapt._core_gbdt hi mask == scheduler.Scheduler(predictor=GBDTPredictor, unbounded budget) token by token
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval"); sys.path.insert(0, "/home/coder/git/nestquant/streaming")
import quantisers as Q, scheduler as SC
from gbdt_predictor import GBDTPredictor
from safetensors.torch import load_file
MAN = "/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json"
bad = tot = b0 = 0
for L in (10, 50, 70):
    ids = load_file(f"/tmp/nestquant/18-e2e/hdump_L10_50_70/L{L}/ids_nqdef.r0of8.safetensors")[f"ids_nqdef"][:3 * 2048].long()
    A = Q.Adapt(lo="/tmp/nestquant/18-e2e/farm_H/nq2", hi="/tmp/nestquant/18-e2e/farm_H/nq4", manifest=MAN, predictor="gbdt", chain="1")
    for seq in (2048, 3 * 2048):                      # reset per window / chained over 3 windows
        hi, serve, d = A._core(L, ids, seq)
        for n in range(ids.shape[0] // seq):
            P = GBDTPredictor([L], {L: A.fixed[L]}, n_float=51, hm=0.5, mode="next_refresh")
            S = SC.Scheduler([L], {L: A.fixed[L]}, {L: A.fdef[L]}, rec_bytes=1.0, cap_GBps=1e9, predictor=P)
            for t in range(seq):
                lv = S.level()[0]                        # levels serving token t (all earlier upgrades landed)
                row = ids[n * seq + t].numpy()
                ref = lv[row] == 4
                got = hi[n * seq + t].numpy()
                tot += 8; m = int((ref != got).sum()); bad += m if t else 0; b0 += m if not t else 0
                c = np.bincount(row, minlength=256)[None].astype(np.float64)
                ups, downs = S.step(c, 1, None, t == 0)
                for (l, e) in ups: S.landed(l, e)
                for (l, e) in downs: S.released(l, e)
            S.close()
        print(L, seq, "l4 share", d["l4_slots"] / d["slots"], "refreshes", d["churn_n"], "mean churn", d["churn_sum"] / max(1, d["churn_n"]), flush=True)
print(f"GBDT parity: {tot} slots, {bad} mismatches (+ {b0} at token 0: Scheduler test starts floating_default at level 2) -> {'PASS' if bad == 0 else 'FAIL'}")
