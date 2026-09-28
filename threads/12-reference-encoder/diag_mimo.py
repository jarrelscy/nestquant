import sys, json, torch
sys.path.insert(0, "."); import nq_run as R, nq_encode as NE, harness as h
torch.cuda.set_per_process_memory_fraction(12/80); torch.backends.cuda.matmul.allow_tf32 = False
data = h.load_expert(55, 70, source=R.MIMO_SRC, statistics=R.MIMO_STATS.format(L=55, E=70), capture=None)
st = data.stats; cnt = st["metadata"]["training_rows"]
H = [st["grams"][0].float().cuda()] * 2 + [st["grams"][1].float().cuda()]
dg = [st["outputs"][i].float().diagonal().clamp_min(1e-30).pow(0.5) for i in range(2)]
W = [w.float().cuda() for w in data.teacher]
out = {}
for pi, pn in enumerate(R.PROJ):
    r = {}
    for K in (2, 4):
        Wq, _ = h.quantize_exl3_like(W[pi], H[pi], K, count=cnt, sigma_reg=0.03); r[f"exl3_{K}"] = h.proxy_loss(W[pi], Wq, H[pi]); h.free_scratch()
    for G in ([None, "G"] if pi < 2 else [None]):
        for inner in (0, 2):
            P = NE.prep(W[pi], H[pi], cnt, 0.03, G=torch.diag(dg[pi]) if G else None, sigma_out=0.03)
            _, dn, inf, _, enc = NE.encode_projection(None, None, None, None, P=P, lam=0.0, shard_axis=R.AXIS[pn], inner=inner)
            for Lv in (2, 4):
                r[f"nq_{G or '1s'}_in{inner}_L{Lv}"] = h.proxy_loss(W[pi], dn[Lv], H[pi])
            r[f"nq_{G or '1s'}_in{inner}_gs"] = float(P["gs"]); r["skew"] = P["skew"]; r["aos"] = bool(P["aos"])
            NE.free(P); del enc, dn; torch.cuda.empty_cache()
    out[pn] = r; print(pn, {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(out, open("results/diag_mimo_proxy.json", "w"), indent=1)
