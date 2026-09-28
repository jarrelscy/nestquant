import sys, os, torch, json
sys.path.insert(0, "."); import nq_run as R, nq_encode as NE, harness as h
torch.cuda.set_per_process_memory_fraction(12/80); torch.backends.cuda.matmul.allow_tf32 = False
d = h.load_expert(55, 70, source=R.MIMO_SRC, statistics=R.MIMO_STATS.format(L=55, E=70), capture=None)
st = d.stats; cnt = st["metadata"]["training_rows"]
W = d.teacher[2].float().cuda(); H = st["grams"][1].float().cuda()
res = {}
for sig in (0.03,):
    P = NE.prep(W, H, cnt, sig)
    for beta in (0.15, 0.1):
        NE.INNER_BETA = beta
        for inner in (6, 12):
            _, dn, inf, _, enc = NE.encode_projection(None, None, None, None, P=P, lam=0.0, shard_axis="k", inner=inner)
            k = f"s{sig}_b{beta}_in{inner}"
            res[k] = [h.proxy_loss(W, dn[L], H) for L in (2, 4)]
            print(k, [round(x, 5) for x in res[k]], flush=True)
            del dn, enc; torch.cuda.empty_cache()
    NE.free(P)
json.dump(res, open("results/diag_mimo_down_inner2.json", "w"), indent=1)
