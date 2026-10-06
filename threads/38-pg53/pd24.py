"""T38: predecode the released 2-4 bit GLM-5.3 (HF jarrelscy/GLM-5.3-NestQuant-2-4bit layers/L{L}/tp*.safetensors) to
fp16 in the nq_e2e predecode-nq layout {out}/nq{lv}/layer_LLL.rRofW.safetensors (use as adapt:lo={out}/nq2,hi={out}/nq4).
Decode = nq_decode.decode_matrix per projection (as predecode-nq slow path), down under the T29 Had512 k-side scope for
layers whose manifest config has in_had_down (L3-6 refit), rounded once to fp16.  Waits for each layer's tp files to
reach their manifest byte sizes (runs alongside the download).
  RANK=r WORLD=w NQ_DEV=cuda:0 python pd24.py ROOT/layers OUT [LAYERS=3-77]"""
import json, os, sys, time
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/12-reference-encoder")
sys.path.insert(0, "/home/coder/git/nestquant/threads/29-outlier-gap")
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import nq_decode as D           # noqa: E402
import nq29_had as NH           # noqa: E402
from nq_fastdec import LayerShards   # noqa: E402
from safetensors.torch import save_file   # noqa: E402

root, out = sys.argv[1], sys.argv[2]
lo, _, hi = (sys.argv[3] if len(sys.argv) > 3 else "3-77").partition("-")
RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD", 1))
dev = os.environ.get("NQ_DEV", "cuda:0")
NE, PROJ = 256, ("gate_proj", "up_proj", "down_proj")


def ready(L):
    d = f"{root}/L{L}"
    try:
        m = json.load(open(f"{d}/manifest.json"))
        return all(os.path.getsize(f"{d}/{f}") == v["bytes"] for f, v in m["files"].items())
    except (OSError, ValueError):
        return False


for lv in (2, 4):
    os.makedirs(f"{out}/nq{lv}", exist_ok=True)
t_all = time.time()
for L in range(int(lo), int(hi or lo) + 1):
    files = {lv: f"{out}/nq{lv}/layer_{L:03d}.r{RANK}of{WORLD}.safetensors" for lv in (2, 4)}
    if all(os.path.exists(f) for f in files.values()):
        continue
    while not ready(L):
        time.sleep(30)
    t0 = time.time()
    ls = LayerShards(root, L)
    wd = int(ls.man["config"].get(NH.FIELD, 128))
    tens = {2: {}, 4: {}}
    for e in [e for e in range(NE) if (L * NE + e) % WORLD == RANK]:
        art = ls.art(e)
        rot = {p: D.rotated_levels(art[p], dev) for p in ("gate", "up", "down")}
        for lv in (2, 4):
            W = [D.decode_matrix(art[p], lv, dev, rot=rot[p]) for p in ("gate", "up")]
            if wd != 128:
                with NH.k_had(art["down"]["meta"]["k"], wd):
                    W.append(D.decode_matrix(art["down"], lv, dev, rot=rot["down"]))
            else:
                W.append(D.decode_matrix(art["down"], lv, dev, rot=rot["down"]))
            for pn, w in zip(PROJ, W):
                assert torch.isfinite(w).all() and float(w.abs().max()) < 6e4, (L, e, pn)
                tens[lv][f"model.layers.{L}.mlp.experts.{e}.{pn}.weight"] = w.half().contiguous().cpu()
        del rot, art
    for lv in (2, 4):
        save_file(tens[lv], files[lv] + ".part", metadata={"root": root, "level": str(lv), "had_down": str(wd)})
        os.rename(files[lv] + ".part", files[lv])
    print(f"[r{RANK}/{WORLD}] L{L}: had_down {wd} in {time.time() - t0:.1f}s", flush=True)
    del tens
print(f"[r{RANK}/{WORLD}] done in {time.time() - t_all:.1f}s", flush=True)
