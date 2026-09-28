"""Per-expert encode timing on this capture's H (read-only use of thread 12 / thread 05 / orbit code).
./run.sh time_encode.py ROOT L E [stats]  -> JSON with seconds per stage + harness.evaluate on the eval capture."""
import json, os, sys, time
sys.path.insert(0, "/home/coder/git/nestquant/threads/12-reference-encoder")
import torch
import harness as h
import nq19_load as C
import nq_run as R
import nq_encode as NE

root, L, E = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stats = sys.argv[4] if len(sys.argv) > 4 else "stats"
kind = sys.argv[5] if len(sys.argv) > 5 else "matched"
R.ART = "/tmp/nestquant/19-capture/enc_timing"         # keep thread 12's artifact dir untouched
h.gpu_cap()
cap = C.Capture(root, stats)
sync = torch.cuda.synchronize
T = {}; t = time.time()
data = cap.expert_data(L, E, kind); sync(); T["load_teacher_eval"] = time.time() - t
t = time.time(); HG = cap.glm_H(L, E); sync(); T["load_H"] = time.time() - t
Ws = data.teacher
methods, extra = {}, {}
t = time.time()
Ps = {pn: NE.prep(Ws[pi], HG["H"][pi], 1, R.SIG[pn], G=HG["G"][pi]) for pi, pn in enumerate(R.PROJ)}
sync(); T["nq_prep"] = time.time() - t
# encode only (thread 12's post-encode decode self-check D.rotated_levels needs >12 GB in its current revision)
t = time.time(); dense = {2: [], 4: []}; ref = {}; binfo = {}
for pn in R.PROJ:
    planes, dn, inf, enc, sc = NE.encode_projection(Ps[pn], shard_axis=R.AXIS[pn], lam=0.3, base_var=R.BASE_VAR, inner=R.INNER)
    ref[pn] = dict(base=NE.frozen_base(enc), scales=sc[2]); binfo[pn] = inf
    for Lv in (2, 4):
        dense[Lv].append(dn[Lv].cpu())
    del enc, planes; torch.cuda.empty_cache()
sync(); T["nq_ref_encode(L2+L4)"] = time.time() - t
methods["nq/L2"], methods["nq/L4"] = dense[2], dense[4]
ref_bits = R.bits_expert(binfo, 4)
rule = dict(kind="pos", K=2, K_hi=2.5, frac=R.frac_for(ref_bits, 4.125, 2.5))
t = time.time(); d4 = []; rinfo = {}
for pn in R.PROJ:
    planes, dn, inf, enc, sc = NE.encode_projection(Ps[pn], shard_axis=R.AXIS[pn], lam=0.3, base_var=R.BASE_VAR, inner=R.INNER,
                                                    base=ref[pn]["base"], base_scales=ref[pn]["scales"], res_rule=rule)
    d4.append(dn[4].cpu()); rinfo[pn] = inf
    del enc, planes; torch.cuda.empty_cache()
sync(); T["nq_rate_variant_r4.125"] = time.time() - t
methods["nq_r4.125/L4"] = d4
T["nq_bpw"] = {"L2": R.bits_expert(binfo, 2), "L4": ref_bits, "r4.125": R.bits_expert(rinfo, 4)}
for p in Ps:
    NE.free(Ps[p])
del Ps, ref; h.free_scratch(); torch.cuda.empty_cache()
for K in (2, 4):
    t = time.time(); q = []
    for pi, pn in enumerate(R.PROJ):
        Wq, _ = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, count=1, sigma_reg=R.SIG[pn]); q.append(Wq.cpu()); h.free_scratch()
    sync(); T[f"EXL3-{K}"] = time.time() - t; methods[f"EXL3-{K}"] = q
try:
    from orbit_duet.modelopt_nvfp4 import fit_weight
    from orbit_duet.nvfp4_reference import decode
    st = cap.pilot_stats(L, E)
    t = time.time(); q = []
    for i, w in enumerate(Ws):
        _, after, _ = fit_weight(w, st["grams"][int(i == 2)], st["metadata"]["training_rows"]); q.append(decode(after).float().cpu())
    sync(); T["NVFP4"] = time.time() - t; methods["NVFP4"] = q
except Exception as ex:
    T["NVFP4_error"] = repr(ex)[:300]
print(json.dumps(T, default=str), file=sys.stderr, flush=True)
methods = {k: [w.cuda() for w in v] for k, v in methods.items()}
t = time.time(); ev = h.evaluate(data, methods); T["evaluate"] = time.time() - t
out = dict(layer=L, expert=E, stats=stats, eval=kind, meta=HG["meta"], seconds=T,
           l2={k: v["all"]["router_weighted_relative_l2"] if "all" in v else v for k, v in ev.items()},
           peak_cuda_gb=torch.cuda.max_memory_allocated() / 2**30)
print(json.dumps(out, indent=1, default=str))
