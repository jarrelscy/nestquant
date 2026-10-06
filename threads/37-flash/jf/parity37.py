#!/usr/bin/env python3
"""T37 offline-vs-streaming parity (port of 33-search/joint/parity_stream.py; CPU fp32).  Replays test-split chains
block by block through JointPredictor.step (think part / answer part as two steps, salience via sal=), compares the
22 inputs, v2 scores, log S and the target() floating sets against the offline pipeline (feat/test X + v2 scores +
the same net), and times a refresh.  Exit 1 if the inputs / scores disagree beyond fp16 / fp32 tolerance.
  parity37.py NAME [nblocks_per_chain=64] [nchains=2] [--model-dir DIR (release dir: uses its jF.pt + v2 txt)]"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train37 as TR  # noqa: E402
import jlib37 as J  # noqa: E402
from joint_predictor37 import JointPredictor, load_net  # noqa: E402
from joint_predictor37 import THINK_ID, ETHINK_ID  # noqa: E402
from gpu_predictor37 import GPUJointPredictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("nbk", nargs="?", type=int, default=64)
ap.add_argument("nch", nargs="?", type=int, default=2); ap.add_argument("--split", default="test")
ap.add_argument("--hm", type=float, default=0.7); ap.add_argument("--model-dir", default="")
a = ap.parse_args()
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
L_ = J.LAYERS
S = a.split
D = {L: J.load(S, L) for L in L_}
FX = {L: np.load(f"{J.OUT}/feat/{S}/L{L}.npz")["X"] for L in L_}
PV = {L: np.load(f"{J.OUT}/scores/v2_{S}/L{L}.npy") for L in L_}
fixed, _ = J.serve_sets()
fixed = {L: fixed[L] for L in L_}
ptf = f"{a.model_dir}/jF.pt" if a.model_dir else f"{J.OUT}/models/{a.name}.pt"
v2f = f"{a.model_dir}/v2_sal_tweedie1.5.txt" if a.model_dir else J.V2
net, ck = load_net(ptf, "cpu")
fxn = np.stack([J.masks(L)[0] for L in L_]); fxm = torch.from_numpy(fxn)
li = torch.tensor([ck["layers"].index(L) for L in L_])
sg = D[L_[0]]["sg"]
rng = np.random.default_rng(0)
kd = D[L_[0]]["ckind"]
cands = [c for c in range(len(sg)) if sg[c][1] - sg[c][0] >= 8 and kd[c] == 0]            # decode chains first
cands = cands or [c for c in range(len(sg)) if sg[c][1] - sg[c][0] >= 8]
chains = sorted(rng.choice(cands, min(a.nch, len(cands)), replace=False).tolist())
worst = dict(x=0.0, p=0.0, s=0.0, g=0.0); tg = []; nset = nsame = 0; tt = []; nb_tot = 0
for c in chains:
    s0, e0 = sg[c]
    nbk = min(a.nbk, e0 - s0)
    p = JointPredictor(L_, fixed, ptf, hm=a.hm, device="cpu", mode="sync", num_threads=8, v2_model=v2f)
    assert p.nf == J.NF and p.NE == J.NE
    q = GPUJointPredictor(L_, fixed, ptf, hm=a.hm, device='cpu', v2_model=v2f)   # torch serve path on CPU
    res_s = np.stack([J.masks(L)[1] for L in L_])            # floating_default at chain start
    res_o = res_s.copy()
    for k in range(s0, s0 + nbk):
        bc = np.stack([D[L]["bcnt"][k] for L in L_]).astype(np.float32)
        ba = np.stack([D[L]["bcnta"][k] for L in L_]).astype(np.float32)
        na = int(D[L_[0]]["nans"][k]); sl = int(D[L_[0]]["segl"][k])
        bs = np.stack([D[L]["bsal"][k] for L in L_]).astype(np.float64)
        th = (bc - ba, J.G - na, [THINK_ID]); an = (ba, na, [ETHINK_ID])
        parts = [th, an] if sl == 1 else [an, th]
        p.step(parts[0][0], parts[0][1], parts[0][2], sal=bs); q.step(parts[0][0], parts[0][1], parts[0][2], sal=bs)
        t1 = time.time()
        ok = p.step(parts[1][0], parts[1][1], parts[1][2], sal=np.zeros_like(bs))
        tt.append(time.time() - t1)
        t1 = time.time(); assert q.step(parts[1][0], parts[1][1], parts[1][2], sal=np.zeros_like(bs))
        tg.append(time.time() - t1)
        worst['g'] = max(worst['g'], float(np.abs(np.log(q.S) - np.log(p.S)).max()))
        assert ok and p.nblk == k - s0 + 1, (ok, p.nblk, k - s0 + 1)
        X, P = p._features()
        Xo = np.stack([FX[L][k] for L in L_]); Po = np.stack([PV[L][k] for L in L_])
        worst["x"] = max(worst["x"], float(np.abs(X.astype(np.float32) - Xo.astype(np.float32)).max()))
        worst["p"] = max(worst["p"], float(np.abs(np.log(np.maximum(P, 1e-30)) - np.log(np.maximum(Po, 1e-30))).max()))
        with torch.no_grad():
            lp = torch.from_numpy(np.log(np.maximum(Po, 1e-30)).astype(np.float32))
            So = torch.exp((lp + net(torch.from_numpy(Xo), lp, li, fxm)).clamp(max=30)).numpy()
        worst["s"] = max(worst["s"], float(np.abs(np.log(p.S) - np.log(So)).max()))
        ws = p.target(res_s)                                  # serve decisions, each with its own hysteresis state
        v = np.where(fxn, -np.inf, So); v = np.where(res_o & ~fxn, v * np.float32(1 + a.hm), v)
        wo = np.zeros_like(res_o); np.put_along_axis(wo, np.argsort(-v, 1, kind="stable")[:, :J.NF], True, 1)
        assert not (ws & fxn).any() and (ws.sum(1) == J.NF).all()
        nset += len(L_); nsame += int((ws == wo).all(1).sum()); res_s, res_o = ws, wo
        nb_tot += 1
    p.close()
# handoff seeder: step_chunk over a handoff chain's prompt tail == offline scores at the handoff, seed set == offline
hw = dict(s=0.0, g=0.0, x=0.0); hsame = hset = 0; hp = J.handoff_pairs(D[L_[0]])[:max(1, a.nch)]
for i, _, h in hp:
    s0, _e = sg[i]
    blk = lambda key, dt: np.stack([np.stack([D[L][key][k] for L in L_]) for k in range(s0, s0 + h)]).astype(dt)  # noqa
    args = (blk("bcnt", np.float32), blk("bsal", np.float64))
    kw = dict(counts_ans=blk("bcnta", np.float32), n_ans=D[L_[0]]["nans"][s0:s0 + h], seg_last=D[L_[0]]["segl"][s0:s0 + h])
    p = JointPredictor(L_, fixed, ptf, hm=a.hm, device="cpu", mode="sync", num_threads=8, v2_model=v2f)
    q = GPUJointPredictor(L_, fixed, ptf, hm=a.hm, device='cpu', v2_model=v2f)
    assert p.step_chunk(*args, **kw) and q.step_chunk(*args, **kw) and p.nblk == h
    k = s0 + h - 1
    Xo = np.stack([FX[L][k] for L in L_]); Po = np.stack([PV[L][k] for L in L_])
    with torch.no_grad():
        lp = torch.from_numpy(np.log(np.maximum(Po, 1e-30)).astype(np.float32))
        So = torch.exp((lp + net(torch.from_numpy(Xo), lp, li, fxm)).clamp(max=30)).numpy()
    X, _P = p._features()
    hw["x"] = max(hw["x"], float(np.abs(X.astype(np.float32) - Xo.astype(np.float32)).max()))
    hw["s"] = max(hw["s"], float(np.abs(np.log(p.S) - np.log(So)).max()))
    hw["g"] = max(hw["g"], float(np.abs(np.log(q.S) - np.log(p.S)).max()))
    ws = p.target_seed()
    wo = np.stack([J.seed_set(So[j], *J.masks(L)) for j, L in enumerate(L_)])
    hset += len(L_); hsame += int((ws == wo).all(1).sum())
if hp:
    print(f"[parity37] handoff seeder: {len(hp)} prompt tails ({[h for _, _, h in hp]} blocks) step_chunk max|dX| "
          f"{hw['x']:.3g} max|dlog S| {hw['s']:.3g} torch {hw['g']:.3g} | identical seed sets (hm 0) {hsame}/{hset}")
    worst["s"] = max(worst["s"], hw["s"]); worst["g"] = max(worst["g"], hw["g"])
print(f"[parity37] {a.name}: {S} chains {chains} ({nb_tot} blocks) x {len(L_)} layers | max|dX| {worst['x']:.3g} "
      f"(fp16 inputs) max|dlog P_v2| {worst['p']:.3g} max|dlog S| {worst['s']:.3g} | identical top-{J.NF} sets "
      f"{nsame}/{nset} | torch serve path max|dlog S| {worst['g']:.3g}")
print(f"[parity37] refresh (features + v2 all-{J.NE} + net, CPU {torch.get_num_threads()} thr, {len(L_)} layers): "
      f"median {np.median(tt) * 1e3:.1f} ms (numpy+lightgbm) / {np.median(tg) * 1e3:.1f} ms (torch trees)")
sys.exit(0 if worst["s"] < 0.05 and worst["p"] < 1e-3 and worst["g"] < 0.05 else 1)
