#!/usr/bin/env python3
"""Gate for nq_fastdec: fp32 bit-exact vs nq_decode.decode_expert (from E{E}.pt) on every expert of the given layers at
levels 2 and 4 (low-rank term included), plus an inter_perm check on a few experts with a synthetic permutation.
    nq_fastdec_gate.py --layers 3,30,77 [--part i --nparts n] [--batch 4] [--batch-had 1] --out result.json
"""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_fastdec as F
import nq_decode as D

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/tmp/nestquant/nq-encode-v1")
ap.add_argument("--layers", default="3,30,77")
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--batch", type=int, default=4); ap.add_argument("--batch-had", type=int, default=1)
ap.add_argument("--perm-experts", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda:0"
tot = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("NQ_VRAM_GB", 11)) / tot))
res = {"layers": {}, "batch": a.batch, "batch_had": bool(a.batch_had), "part": a.part, "nparts": a.nparts}
for L in [int(x) for x in a.layers.split(",")]:
    ls = F.LayerShards(a.root, L)
    ex = [e for e in ls.experts if e % a.nparts == a.part]
    bad, n, t_fast, t_ref, lr_seen = [], 0, 0.0, 0.0, 0
    for i in range(0, len(ex), a.batch):
        chunk = ex[i:i + a.batch]
        arts = [ls.art(e) for e in chunk]
        torch.cuda.synchronize(); t = time.time()
        out = F.decode_experts(arts, (2, 4), dev, batch_had=bool(a.batch_had))
        torch.cuda.synchronize(); t_fast += time.time() - t
        for k, e in enumerate(chunk):
            art = torch.load(f"{a.root}/L{L}/experts/E{e}.pt", map_location="cpu", weights_only=False)
            lr_seen += sum(art[p]["base"].get("lr") is not None for p in F.PROJ)
            t = time.time()
            for lv in (2, 4):
                ref = D.decode_expert(art, lv, dev)
                for j, pn in enumerate(F.PROJ):
                    if not torch.equal(ref[j], out[lv][k][j]):
                        bad.append([e, lv, pn, float((ref[j] - out[lv][k][j]).abs().max())])
                    n += 1
            torch.cuda.synchronize(); t_ref += time.time() - t
        del out
    # inter_perm: synthetic permutation of the intermediate channels on the first perm-experts of this part
    pbad = []
    for e in ex[:a.perm_experts]:
        art = torch.load(f"{a.root}/L{L}/experts/E{e}.pt", map_location="cpu", weights_only=False)
        inter = art["gate"]["meta"]["n"]
        perm = torch.randperm(inter, generator=torch.Generator().manual_seed(1000 * L + e)).tolist()
        art.setdefault("meta", {})["inter_perm"] = perm
        out = F.decode_experts([ls.art(e)], (2, 4), dev, batch_had=bool(a.batch_had), perms=[perm])
        for lv in (2, 4):
            ref = D.decode_expert(art, lv, dev)
            pbad += [[e, lv, j] for j in range(3) if not torch.equal(ref[j], out[lv][0][j])]
    res["layers"][L] = dict(experts=len(ex), matrices=n, mismatches=bad, perm_mismatches=pbad,
                            perm_experts=ex[:a.perm_experts], lr_planes=lr_seen,
                            fast_s_per_expert=t_fast / len(ex), ref_s_per_expert=t_ref / len(ex))
    print(f"L{L} part {a.part}: {len(ex)} experts, {n} matrices, mismatches {len(bad)}, perm mismatches {len(pbad)}, "
          f"lr planes {lr_seen}, fast {t_fast / len(ex) * 1e3:.1f} ms/expert (both levels), "
          f"ref {t_ref / len(ex):.2f} s/expert", flush=True)
res["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
json.dump(res, open(a.out, "w"), indent=1)
