"""T35: predecode the b175 campaign artifacts (pattern-rate base) to fp16, in the layout of nq_e2e predecode-nq:
{out}/nq{lv}/layer_LLL.rRofW.safetensors, keys model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight.
Decode = real_adapt.AdaptReal._real exactly (nq15 base-K decoder over nq_decode, T29 Had512 down scope for layers
3-6, inter_perm undone), rounded once to fp16 like predecode-nq.  Use the output as adapt:lo={out}/nq2,hi={out}/nq4.
  RANK=r WORLD=w NQ_DEV=cuda:0 python predec175.py ROOT OUT [LAYERS=3-77] [LEVELS=2,4]
Shards experts by (L*256+E) % WORLD == RANK (any sharding works for the harness's SafeIndex).  Resumable per file."""
import os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/coder/git/nestquant/threads/12-reference-encoder")
import nq15                                   # noqa: F401,E402  base-K decoder over nq_decode
import nq_decode as D                         # noqa: E402
import nq35_t29 as T29                        # noqa: E402
from safetensors.torch import save_file       # noqa: E402

root, out = sys.argv[1], sys.argv[2]
lo, _, hi = (sys.argv[3] if len(sys.argv) > 3 else "3-77").partition("-")
levels = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,4").split(",")]
RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD", 1))
dev = os.environ.get("NQ_DEV", "cuda:0")
NE = int(os.environ.get("NQ_NE", 256))
PROJ = ("gate_proj", "up_proj", "down_proj")


def decode(art, level, rot):
    W = [D.decode_matrix(art[p], level, dev, rot=rot[p]) for p in ("gate", "up")]
    with T29.down_scope(art):
        W.append(D.decode_matrix(art["down"], level, dev, rot=rot["down"]))
    perm = art.get("meta", {}).get("inter_perm")
    if perm is not None:
        inv = torch.argsort(torch.as_tensor(perm, device=dev))
        W = [W[0][inv], W[1][inv], W[2][:, inv]]
    return W


for lv in levels:
    os.makedirs(f"{out}/nq{lv}", exist_ok=True)
t_all = time.time()
for L in range(int(lo), int(hi or lo) + 1):
    files = {lv: f"{out}/nq{lv}/layer_{L:03d}.r{RANK}of{WORLD}.safetensors" for lv in levels}
    want = [lv for lv in levels if not os.path.exists(files[lv])]
    if not want:
        continue
    ex = [e for e in range(NE) if (L * NE + e) % WORLD == RANK]
    t0 = time.time()
    tens = {lv: {} for lv in want}
    for e in ex:
        f = f"{root}/L{L}/experts/E{e}.pt"
        art = torch.load(f, map_location="cpu", weights_only=False)
        rot = {p: D.rotated_levels(art[p], dev) for p in ("gate", "up", "down")}
        for lv in want:
            for pn, w in zip(PROJ, decode(art, lv, rot)):
                assert torch.isfinite(w).all() and float(w.abs().max()) < 6e4, (L, e, pn)
                tens[lv][f"model.layers.{L}.mlp.experts.{e}.{pn}.weight"] = w.half().contiguous().cpu()
        del rot, art
    for lv in want:
        save_file(tens[lv], files[lv] + ".part", metadata={"root": root, "level": str(lv), "decoder": "nq15+t29"})
        os.rename(files[lv] + ".part", files[lv])
    print(f"[r{RANK}/{WORLD}] L{L}: {len(ex)} experts x levels {want} in {time.time() - t0:.1f}s", flush=True)
    del tens
print(f"[r{RANK}/{WORLD}] done in {time.time() - t_all:.1f}s", flush=True)
