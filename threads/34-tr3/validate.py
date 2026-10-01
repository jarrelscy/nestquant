"""T34 decoder validation (CPU): tier map vs trellis widths, exact bits, rel weight error vs the FP8 source.
    validate.py OUT.json [full_layers=3,40,77] [per_layer=8]"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/coder/git/nestquant/threads/34-tr3")
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import tr3_decode as T  # noqa: E402
import nq_io  # noqa: E402

torch.set_num_threads(16)
SRC = "/tmp/nestquant/34-tr3/src"
out = sys.argv[1]
full = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "3,40,77").split(",")]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 8
tb = json.load(open(f"{SRC}/tier_bitmap.json"))
fp8 = nq_io.FP8Model("/tmp/nestquant/src/glm53-fp8")
res = {"layers": {}, "rows": []}
tot_bits = tot_w = 0
for L in range(3, 79):
    R = T.TR3Layer(SRC, L)
    Ks = [R.K(e) for e in range(256)]
    assert Ks == tb[str(L)]["k"], f"L{L}: trellis widths != tier_bitmap"
    bits = [T.bits_of_expert(R, e) for e in range(256)]
    nw = 3 * 2048 * 6144
    res["layers"][L] = dict(n3=Ks.count(3), n4=Ks.count(4), bpw=sum(bits) / (256 * nw))
    if L <= 77:
        tot_bits += sum(bits); tot_w += 256 * nw
    if L == 78:
        continue            # MTP layer: not in the FP8 eval path
    rng = np.random.default_rng(L)
    k3 = [e for e in range(256) if Ks[e] == 3]; k4 = [e for e in range(256) if Ks[e] == 4]
    es = range(256) if L in full else sorted(rng.choice(k3, per // 2, replace=False).tolist() +
                                             rng.choice(k4, per // 2, replace=False).tolist())
    t0 = time.time()
    for e in es:
        raw = R.raw(e)
        W = R.expert(e, raw=raw)
        ref = fp8.expert(L, e, "cpu")
        row = dict(L=L, e=e, K=Ks[e])
        for p in T.PROJ:
            a, b = W[p].float(), ref[p].float()
            row[p] = float((a - b).norm() / b.norm())
        if e == es[0]:          # fp32-Hadamard (exllamav3 get_weight_tensor arithmetic) vs fp64: fp16 ulp flips
            W32 = R.expert(e, raw=raw, had_dtype=torch.float32)
            row["fp32_vs_fp64_flips"] = int(sum(int((W32[p] != W[p]).sum()) for p in T.PROJ))
            row["fp32_vs_fp64_maxrel"] = float(max(((W32[p].float() - W[p].float()).abs().max() /
                                                    W[p].float().abs().max()) for p in T.PROJ))
        res["rows"].append(row)
    print(f"L{L} n3={Ks.count(3)} n4={Ks.count(4)} bpw={res['layers'][L]['bpw']:.4f} "
          f"{len(es)} experts {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"))
res["routed_bpw_L3_77"] = tot_bits / tot_w
rows = res["rows"]
for K in (3, 4):
    for p in T.PROJ:
        v = np.array([r[p] for r in rows if r["K"] == K])
        res[f"K{K}_{p}"] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std()), min=float(v.min()), max=float(v.max()))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "layers")}, indent=1))
