"""GPUJointPredictor vs JointPredictor (numpy reference, itself exact vs offline) on heldout chains; run on CPU
(DEV=cpu, no lock) for arithmetic parity, on CUDA under gpu.lock for timing.
  LAYOUT=k0 DEV=cpu python parity_gpu.py NAME nblocks chains"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as TR                                            # noqa: E402
import jlib as J                                              # noqa: E402
from joint_predictor import JointPredictor                    # noqa: E402
from gpu_predictor import GPUJointPredictor                   # noqa: E402
from gbdt_predictor import THINK_ID, ETHINK_ID                # noqa: E402
name, nbk = sys.argv[1], int(sys.argv[2]); chains = [int(c) for c in sys.argv[3].split(",")]
dev = os.environ.get("DEV", "cpu"); ref = os.environ.get("NOREF") is None
L_ = TR.LAYERS; D = {L: J.load("glm52-heldout", L) for L in L_}; fixed = {L: list(J.FIXED[L]) for L in L_}
mp = f"{J.OUT}/models/{name}.pt"
wx = wp = ws = 0.0; nset = nsame = 0; tt = []; ta = []
for c in chains:
    s0 = D[L_[0]]["sg"][c][0]
    g = GPUJointPredictor(L_, fixed, mp, n_float=J.NF, hm=0.6, device=dev, graph=os.environ.get("GRAPH") == "1",
                          bf16=os.environ.get("BF16", "1") == "1")
    p = JointPredictor(L_, fixed, mp, n_float=J.NF, hm=0.6, device="cpu", mode="sync", num_threads=8) if ref else None
    rg = rp = np.stack([J.masks(L)[1] for L in L_])
    for k in range(s0, s0 + nbk):
        bc = np.stack([D[L]["bcnt"][k] for L in L_]).astype(np.float32); ba = np.stack([D[L]["bcnta"][k] for L in L_]).astype(np.float32)
        na = int(D[L_[0]]["nans"][k]); sl = int(D[L_[0]]["segl"][k]); bs = np.stack([D[L]["bsal"][k] for L in L_]).astype(np.float64)
        th = (bc - ba, 16 - na, [THINK_ID]); an = (ba, na, [ETHINK_ID]); parts = [th, an] if sl == 1 else [an, th]
        z = np.zeros_like(bs)
        cg = [torch.from_numpy(parts[i][0]).to(dev) for i in (0, 1)]; sg = [torch.from_numpy(bs).to(dev), torch.from_numpy(z).to(dev)]
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time(); g.step(cg[0], parts[0][1], parts[0][2], sal=sg[0])
        if dev == "cuda": torch.cuda.synchronize()
        ta.append(time.time() - t0)
        t1 = time.time(); g.step(cg[1], parts[1][1], parts[1][2], sal=sg[1]); tt.append(time.time() - t1)
        if ref:
            for i in (0, 1):
                p.step(parts[i][0], parts[i][1], parts[i][2], sal=bs if i == 0 else z)
            Xg, Pg = g.features(); Xr, Pr = p._features()
            wx = max(wx, float((Xg.float().cpu() - torch.from_numpy(Xr.astype(np.float32))).abs().max()))
            wp = max(wp, float(np.abs(np.log(Pg.cpu().numpy()) - np.log(Pr)).max()))
            ws = max(ws, float(np.abs(np.log(g.S) - np.log(p.S)).max()))
            a_, b_ = g.target(rg), p.target(rp); nset += len(L_); nsame += int((a_ == b_).all(1).sum()); rg, rp = a_, b_
print(f"{name} {dev}: chains {chains} x {nbk} blocks | max|dX| {wx:.3g} max|dlog P_v2| {wp:.3g} max|dlog S| {ws:.3g} "
      f"| identical top-{J.NF} sets {nsame}/{nset}")
import torch as _t
if dev == "cuda": print(f"peak VRAM {_t.cuda.max_memory_allocated() / 2**20:.0f} MiB")
print(f"refresh step (close block + features + 60 trees + net + D2H), {dev}: median {np.median(tt[5:]) * 1e3:.2f} ms  "
      f"p90 {np.percentile(tt[5:], 90) * 1e3:.2f} ms | non-refresh accumulate step median {np.median(ta[5:]) * 1e3:.3f} ms"
      f" | graph={os.environ.get('GRAPH')} bf16={os.environ.get('BF16', '1')}")
