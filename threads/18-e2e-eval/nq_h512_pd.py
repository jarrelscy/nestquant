"""Predecode T29's in_had_down=512 refit (L3-6, nq-encode-h512) with T29's decoder (nq29_had.assemble29 +
decode_expert29); the pinned nq_decode / nq_fastdec mis-decode it.  Same layout as predecode-nq:
  {out}/nq{2,4}/layer_LLL.rRofW[.b].safetensors, fp16, keys model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight
phase a = nq2 every expert + nq4 of the default set (enough for nqdef), phase b (.b files) = nq4 of the rest.
  RANK=r WORLD=w python nq_h512_pd.py ROOT OUT DEFSET.json [layers=3-6] [phases=ab]
  python nq_h512_pd.py --gate OUT ROOT [n_per_layer]   stored tensors == decode_expert29(...).half(), torch.equal"""
import os, sys, json, time, threading
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/29-outlier-gap")
import nq29_had as NH
from safetensors.torch import save_file
PROJ = ("gate_proj", "up_proj", "down_proj")
RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD", 1))
if os.environ.get("NQ_VRAM_GB"):
    tot = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["NQ_VRAM_GB"]) / tot))


def log(*a):
    print(f"[r{RANK} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def dec(root, L, E, levels):
    art = NH.assemble29(root, L, E)
    assert NH.width_of(art) == 512, (L, E)
    out = {}
    for lv in levels:
        W = NH.decode_expert29(art, lv)
        t = {}
        for pn, w in zip(PROJ, W):
            assert torch.isfinite(w).all() and float(w.abs().max()) < 6e4, (L, E, pn)
            t[f"model.layers.{L}.mlp.experts.{E}.{pn}.weight"] = w.half().contiguous().cpu()
        out[lv] = t
    return out


def main(root, out, defset, layers="3-6", phases="ab"):
    lo, _, hi = layers.partition("-")
    Ls = range(int(lo), int(hi or lo) + 1)
    d4 = {int(k): set(v) for k, v in json.load(open(defset))["layers"].items()}
    for lv in (2, 4):
        os.makedirs(f"{out}/nq{lv}", exist_ok=True)
    wr = []
    t0 = time.time()
    for ph in phases:
        for L in Ls:
            ex = [e for e in range(256) if (L * 256 + e) % WORLD == RANK]
            jobs = {}
            if ph == "a":
                jobs[2] = (f"{out}/nq2/layer_{L:03d}.r{RANK}of{WORLD}.safetensors", ex)
                jobs[4] = (f"{out}/nq4/layer_{L:03d}.r{RANK}of{WORLD}.safetensors", [e for e in ex if e in d4[L]])
            else:
                jobs[4] = (f"{out}/nq4/layer_{L:03d}.r{RANK}of{WORLD}.b.safetensors", [e for e in ex if e not in d4[L]])
            jobs = {lv: j for lv, j in jobs.items() if j[1] and not os.path.exists(j[0])}
            if not jobs:
                continue
            tens = {lv: {} for lv in jobs}
            for e in sorted(set().union(*[set(j[1]) for j in jobs.values()])):
                r = dec(root, L, e, [lv for lv in jobs if e in jobs[lv][1]])
                for lv, t in r.items():
                    tens[lv].update(t)
            for th in wr:
                th.join()

            def w(tens=tens, jobs=jobs):
                for lv, (f, _) in jobs.items():
                    save_file(tens[lv], f + ".part", metadata={"root": root, "level": str(lv), "in_had_down": "512"})
                    os.rename(f + ".part", f)
            th = threading.Thread(target=w); th.start(); wr = [th]
            log(f"phase {ph} L{L}: " + " ".join(f"nq{lv}:{len(jobs[lv][1])}" for lv in jobs) + f" ({time.time()-t0:.0f}s)")
        for th in wr:
            th.join()
        log(f"PHASE {ph} DONE ({time.time()-t0:.0f}s)")


def gate(out, root, n=16):
    import glob, struct
    from safetensors import safe_open
    bad = tot = 0
    t0 = time.time()
    for f in sorted(glob.glob(f"{out}/nq*/layer_*.safetensors")):
        lv = int(f.split("/nq")[1][0])
        with safe_open(f, "pt") as fh:
            ks = sorted(fh.keys())
            es = sorted({(int(k.split(".")[2]), int(k.split(".")[5])) for k in ks})
            for L, E in es[:: max(1, len(es) // max(1, n // 8))][: max(1, n // 8)]:
                ref = dec(root, L, E, [lv])[lv]
                for k, v in ref.items():
                    tot += 1
                    bad += not torch.equal(fh.get_tensor(k), v)
    print(f"GATE h512 predecode vs decode_expert29: {tot} tensors, {bad} mismatches -> {'PASS' if tot and not bad else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--gate":
        gate(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
    else:
        main(*sys.argv[1:])
