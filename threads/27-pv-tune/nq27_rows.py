"""T27: gather the routed fit rows of chosen experts from T19's kept chunk-0 activations (shards s00-s05, 1,048,576
tokens; FORMAT.md 'Stage-1 activations').  Per (L, E) -> /tmp/nestquant/27-pv-tune/rows/L{L}_E{E}.pt:
  x [N, 6144] bf16 (normalized MoE input), p [N] f32 (router weight incl. scaling), blk [N] int32 (global
  2048-token block id = held-out split unit: window-aligned for both c512 and c2048 shards).
  python nq27_rows.py L E1,E2,...
"""
import os, sys, json
import numpy as np
import torch

ROOT = "/tmp/nestquant/19-capture-glmfmt/shards"
OUT = "/tmp/nestquant/27-pv-tune/rows"
SHARDS = ["s00", "s01", "s02", "s03", "s04", "s05"]
Dm = 6144


def gather(L, experts):
    want = {E: dict(x=[], p=[], blk=[]) for E in experts}
    base = 0
    for s in SHARDS:
        d = f"{ROOT}/{s}/acts/L{L}"
        n = json.load(open(f"{d}/done.json"))["rows"]
        ids = np.fromfile(f"{d}/ids.u8", np.uint8).reshape(n, 8)
        p = np.fromfile(f"{d}/p.f32", np.float32).reshape(n, 8)
        x = np.memmap(f"{d}/x.bf16", np.uint16, "r", shape=(n, Dm))
        for E in experts:
            r, sl = np.nonzero(ids == E)
            o = np.argsort(r); r, sl = r[o], sl[o]
            want[E]["x"].append(torch.from_numpy(np.ascontiguousarray(x[r])).view(torch.bfloat16))
            want[E]["p"].append(torch.from_numpy(p[r, sl].copy()))
            want[E]["blk"].append(torch.from_numpy(((base + r) // 2048).astype(np.int32)))
        base += n
        del x
    for E in experts:
        w = want[E]
        out = dict(layer=L, expert=E, x=torch.cat(w["x"]), p=torch.cat(w["p"]), blk=torch.cat(w["blk"]),
                   source=[f"{ROOT}/{s}/acts/L{L}" for s in SHARDS])
        tmp = f"{OUT}/L{L}_E{E}.pt.tmp"
        torch.save(out, tmp); os.replace(tmp, f"{OUT}/L{L}_E{E}.pt")
        print(f"L{L} E{E} rows {len(out['p'])}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    L = int(sys.argv[1]); ex = [int(e) for e in sys.argv[2].split(",")]
    ex = [E for E in ex if not os.path.exists(f"{OUT}/L{L}_E{E}.pt")]
    if ex:
        gather(L, ex)
