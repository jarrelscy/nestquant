import sys, json, time, argparse, torch
from pathlib import Path
import harness as hz
from matgptq import fit_expert

ap = argparse.ArgumentParser()
ap.add_argument("--expert", type=int, default=36); ap.add_argument("--layer", type=int, default=16)
ap.add_argument("--configs", required=True, help="json list of dicts (name + matgptq kwargs)")
ap.add_argument("--out", required=True); ap.add_argument("--refs", action="store_true")
ap.add_argument("--exl3-sigma", type=float, default=None)
a = ap.parse_args()
cfgs = json.loads(Path(a.configs).read_text()) if a.configs.endswith(".json") else json.loads(a.configs)
data = hz.load_expert(a.layer, a.expert)
run = Path(hz.GLM_RUN.format(L=a.layer))
out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
methods = {}
if a.refs:
    for b in [2, 4]:
        methods[f"exl3_{b}"] = hz.load_exl3_bin(run / f"exl3_e{a.expert}/expert_{b}.bin")
    methods["nvfp4"] = hz.load_nvfp4(str(run / f"nvfp4_e{a.expert}/weights.pt"), data)
if a.exl3_sigma is not None:
    for K in [2, 4]:
        q, _ = hz.quantize_expert_exl3_like(data, K, sigma_reg=a.exl3_sigma); hz.free_scratch()
        methods[f"exl3_{K}_s{a.exl3_sigma}"] = q
meta = {}
for c in cfgs:
    c = dict(c); name = c.pop("name"); t0 = time.time()
    q2, q4, info = fit_expert(data, **c)
    methods[name + "@2"] = q2; methods[name + "@4"] = q4
    meta[name] = dict(cfg=c, seconds=round(time.time() - t0, 1), **info)
    print(name, meta[name], flush=True)
ev = hz.evaluate(data, methods)
tab = hz.table(ev)
prox = {m: hz.proxy_losses(data, q) for m, q in methods.items()}
for m in methods:
    base = m.split("@")[0]
    out[m] = dict(table=tab[m], proxy=prox[m], meta=meta.get(base))
    print(m, tab[m], flush=True)
Path(a.out).write_text(json.dumps(out, indent=1))
