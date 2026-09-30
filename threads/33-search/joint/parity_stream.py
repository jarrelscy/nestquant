"""Offline-vs-streaming parity for JointPredictor (CPU, fp32): replays heldout chains block by block through
JointPredictor.step (think part / answer part as two steps, salience via sal=), compares the 22 inputs, the scores
and the target() sets against the offline pipeline (feat/ X + v2 scores + the same net), and times a refresh.
  LAYOUT=k0 python parity_stream.py NAME [nblocks_per_chain] [chains]"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as TR                                  # noqa: E402  (before jlib: it prepends 32-gbdt-sal/)
import jlib as J                                    # noqa: E402
from joint_predictor import JointPredictor
from gbdt_predictor import THINK_ID, ETHINK_ID

name = sys.argv[1]
nbk = int(sys.argv[2]) if len(sys.argv) > 2 else 64
chains = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 3]
S = "glm52-heldout"
L_ = TR.LAYERS
D = {L: J.load(S, L) for L in L_}
FX = {L: np.load(f"{J.OUT}/feat/{S}/L{L}.npz")["X"] for L in L_}
PV = {L: np.load(f"{J.OUT}/scores/v2_{S}/L{L}.npy") for L in L_}
fixed = {L: list(J.FIXED[L]) for L in L_}
ck = torch.load(f"{J.OUT}/models/{name}.pt", map_location="cpu")
a = ck["args"]
net = TR.Net(ck["K"], a["arch"], a["d"], a["nl"]).eval(); net.load_state_dict(ck["state"])
fxm = torch.from_numpy(np.stack([J.masks(L)[0] for L in L_]))
li = torch.arange(len(L_))
worst = dict(x=0.0, p=0.0, s=0.0); nset = nsame = 0; tt = []
for c in chains:
    s0 = D[L_[0]]["sg"][c][0]
    p = JointPredictor(L_, fixed, f"{J.OUT}/models/{name}.pt", n_float=J.NF, hm=0.6, device="cpu", mode="sync",
                       num_threads=8)
    res_s = np.stack([J.masks(L)[1] for L in L_])            # floating_default at chain start
    res_o = res_s.copy()
    for k in range(s0, s0 + nbk):
        bc = np.stack([D[L]["bcnt"][k] for L in L_]).astype(np.float32)
        ba = np.stack([D[L]["bcnta"][k] for L in L_]).astype(np.float32)
        na = int(D[L_[0]]["nans"][k]); sl = int(D[L_[0]]["segl"][k])
        bs = np.stack([D[L]["bsal"][k] for L in L_]).astype(np.float64)
        th = (bc - ba, G_ := 16 - na, [THINK_ID]); an = (ba, na, [ETHINK_ID])
        parts = [th, an] if sl == 1 else [an, th]
        p.step(parts[0][0], parts[0][1], parts[0][2], sal=bs)
        t1 = time.time()
        ok = p.step(parts[1][0], parts[1][1], parts[1][2], sal=np.zeros_like(bs))
        tt.append(time.time() - t1)
        assert ok and p.nblk == k - s0 + 1
        X, P = p._features()
        Xo = np.stack([FX[L][k] for L in L_]); Po = np.stack([PV[L][k] for L in L_])
        worst["x"] = max(worst["x"], float(np.abs(X.astype(np.float32) - Xo.astype(np.float32)).max()))
        worst["p"] = max(worst["p"], float(np.abs(P - Po).max()))
        with torch.no_grad():
            lp = torch.from_numpy(np.log(np.maximum(Po, 1e-30)).astype(np.float32))
            So = torch.exp((lp + net(torch.from_numpy(Xo), lp, li, fxm)).clamp(max=30)).numpy()
        worst["s"] = max(worst["s"], float(np.abs(np.log(p.S) - np.log(So)).max()))
        # serve decisions on both score matrices, each with its own hysteresis state
        ws = p.target(res_s)
        v = np.where(res_o, So * np.float32(1.6), So); wo = np.zeros_like(res_o)
        np.put_along_axis(wo, np.argsort(-v, 1, kind="stable")[:, :J.NF], True, 1)
        nset += len(L_); nsame += int((ws == wo).all(1).sum()); res_s, res_o = ws, wo
    p.close()
print(f"{name}: chains {chains} x {nbk} blocks x 75 layers | max|dX| {worst['x']:.3g} (fp16 inputs) "
      f"max|dP_v2| {worst['p']:.3g} max|dlog S| {worst['s']:.3g} | identical top-{J.NF} sets {nsame}/{nset}")
print(f"refresh (features + v2 all-256 + net, CPU {torch.get_num_threads()} thr): median {np.median(tt) * 1e3:.1f} ms")
