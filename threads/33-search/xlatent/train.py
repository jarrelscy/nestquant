#!/usr/bin/env python3
"""train.py NAME [--dz 32 --latent gru|ema --noctx --nov2 --zero --epochs 30 ...]: train XLatent on calib-fit train
chains, early-stop on calib-fit val chains (sal-hot sync hm0.5), write scores for val / heldout (/ sm120tf with --tf)
to D/<corpus>/S_<NAME>.npy.  PRIVATE data stays under /tmp."""
import argparse
import json
import os
import time

import numpy as np
import torch

import xlib as X
import xnet as N

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--dz", type=int, default=32)
ap.add_argument("--r", type=int, default=8)
ap.add_argument("--latent", default="gru")
ap.add_argument("--noctx", action="store_true")
ap.add_argument("--nov2", action="store_true")
ap.add_argument("--zero", action="store_true", help="control: z forced to 0 (no shared latent)")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tb", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--tf", action="store_true")
ap.add_argument("--dev", default="cuda")
ap.add_argument("--hm", type=float, default=0.5)
a = ap.parse_args()
torch.manual_seed(0)
dev = torch.device(a.dev)
OUTD = "/tmp/nestquant/33-search/xlatent/runs"
os.makedirs(OUTD, exist_ok=True)
FX, FD = X.masks()
fxt = torch.from_numpy(FX).to(dev)


def feats(corpus):
    p = f"{X.D}/{corpus}/feat7.npy"
    if not os.path.exists(p):
        t0 = time.time()
        f = N.own_features(X.load(corpus, "bsal"), X.load(corpus, "bcnt"), X.load(corpus, "S_v2"), X.chains(corpus), dev)
        np.save(p, f.numpy())
        print(f"features {corpus} {time.time() - t0:.0f}s", flush=True)
    return np.load(p, mmap_mode="r")


# m_L (calib-fit mean salience per routed slot)
bs_c = X.load("calib-fit", "bsal", mmap=False)
mL = bs_c.sum((0, 2), dtype=np.float64) / (bs_c.shape[0] * 16 * 8)
sg = X.chains("calib-fit")
nbc = 512
VAL = [4, 9, 14, 19, 24, 29]
TR = [i for i in range(len(sg)) if i not in VAL]


def targets(bsal, s, e):
    """next-64 salience / m_L for blocks s..e-1 of a chain -> [n,75,256] f32, valid [n]"""
    b = np.asarray(bsal[s:e], np.float64) / mL[None, :, None]
    cs = np.concatenate([np.zeros((1,) + b.shape[1:]), np.cumsum(b, 0)])
    n = e - s
    k = np.arange(n)
    Y = cs[np.minimum(k + 5, n)] - cs[np.minimum(k + 1, n)]
    return Y.astype(np.float32), (k + 4 < n)


Fc = feats("calib-fit")
Ftr = torch.from_numpy(np.stack([np.asarray(Fc[sg[i][0]:sg[i][1]]) for i in TR])).to(dev)       # [B,512,75,256,7] f16
Ytr, Mtr = zip(*[targets(bs_c, *sg[i]) for i in TR])
Ytr = torch.from_numpy(np.stack(Ytr)).to(dev, torch.float16)
Mtr = torch.from_numpy(np.stack(Mtr)).to(dev)
print("train tensors", tuple(Ftr.shape), flush=True)

model = N.XLatent(dz=a.dz, r=a.r, latent=a.latent, use_ctx=not a.noctx, use_v2=not a.nov2, dz_zero=a.zero).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)


@torch.no_grad()
def predict(F, sgs, chunk=64):
    """F array [nb,75,256,7]; sgs chain list -> S [nb,75,256] f32 numpy (mu; fixed experts irrelevant)"""
    model.eval()
    S = np.zeros(F.shape[:3], np.float32)
    for s, e in sgs:
        st = torch.zeros(1, a.dz, device=dev)
        for c0 in range(s, e, chunk):
            c1 = min(e, c0 + chunk)
            f = torch.from_numpy(np.asarray(F[c0:c1])).to(dev)[None]
            lm, st, _ = model(f, st)
            S[c0:c1] = torch.exp(lm[0]).float().cpu().numpy()
    model.train()
    return S


def val_score():
    vs = [sg[i] for i in VAL]
    S = predict(Fc, vs)
    sv = X.replay(S, FX, FD, vs, a.hm)
    return X.metrics(sv, bs_c, FX, vs, tf=True)


ref = X.metrics(X.replay(X.load("calib-fit", "S_v2"), FX, FD, [sg[i] for i in VAL], a.hm), bs_c, FX,
                [sg[i] for i in VAL], tf=True)
print(f"v2 calib-val sal-hot {ref['sal'] * 100:.2f} churn {ref['churn']:.2f}", flush=True)
mask_e = (~fxt)[None, None].float()
best, hist = -1, []
for ep in range(a.epochs):
    t0 = time.time()
    st = torch.zeros(Ftr.shape[0], a.dz, device=dev)
    tot = 0.0
    for c0 in range(0, nbc, a.tb):
        f = Ftr[:, c0:c0 + a.tb]
        lm, st, _ = model(f, st)
        st = st.detach()
        m = Mtr[:, c0:c0 + a.tb, None, None].float() * mask_e
        loss = N.tweedie_loss(lm, Ytr[:, c0:c0 + a.tb].float(), m)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot += loss.item()
    sched.step()
    v = val_score()
    hist.append(dict(ep=ep, loss=tot, val=v["sal"], churn=v["churn"]))
    print(f"ep {ep} loss {tot / (nbc // a.tb):.4f} val sal-hot {v['sal'] * 100:.2f} churn {v['churn']:.2f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if v["sal"] > best:
        best = v["sal"]; torch.save(model.state_dict(), f"{OUTD}/{a.name}.pt")
model.load_state_dict(torch.load(f"{OUTD}/{a.name}.pt"))
res = dict(args=vars(a), v2_val=ref["sal"], v2_val_churn=ref["churn"], best_val=best, hist=hist)
Sv = predict(Fc, [sg[i] for i in VAL])
np.save(f"{X.D}/calib-fit/S_{a.name}_val.npy", Sv)
for corpus in ["glm52-heldout"] + (["sm120tf"] if a.tf else []):
    Fh = feats(corpus)
    S = predict(Fh, X.chains(corpus))
    np.save(f"{X.D}/{corpus}/S_{a.name}.npy", S)
    for hm in (0.3, 0.5, 0.7):
        r = X.evaluate(S, corpus, hm)
        res[f"{corpus}_hm{hm}"] = {k: v for k, v in r.items() if k != "per_layer"}
        print(f"{corpus} {a.name} hm{hm}: sal-hot {r['sal'] * 100:.2f} churn {r['churn']:.2f} L3-6 {r['L3_6'] * 100:.1f} "
              f"L7-40 {r['L7_40'] * 100:.1f} L41-77 {r['L41_77'] * 100:.1f}", flush=True)
json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
