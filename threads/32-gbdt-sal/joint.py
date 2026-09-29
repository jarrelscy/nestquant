#!/usr/bin/env python3
"""T32 idea 3: joint 256-way per-layer model (set transformer over the 256 experts of a layer at a block end).
Why: top-8 routing is zero-sum per token; per-expert GBDT scores cannot see "A rising => B fading" (the rank 30-150
misordering in decomp.py).  Shared across layers + learned (layer, expert) embedding = joint / cross-layer learning.
Per (layer, block b) input = 256 experts x k causal features (the v2 serve features + long EMAs + log v2 GBDT
prediction), target = next-64 salience / m_L (= v2's target), tweedie 1.5 loss on non-fixed experts of valid blocks.
  base  : score = exp(log p_v2 + head)   (residual on the v2 GBDT)
  solo  : score = exp(head), no GBDT input
  joint.py prep CORPUS            -> $OUT/joint/CORPUS/L.npz   (PRIVATE, derived per token)
  joint.py train NAME base|solo [EPOCHS] [NLAYERS] [SUB]   -> $OUT/models/joint/NAME.pt + heldout sim json"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import t32lib as T  # noqa: E402

V2 = "/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
FE = ("ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128",
      "ema512", "ema2048", "sema512", "sema2048")
JD = f"{T.OUT}/joint"
MD = f"{T.OUT}/models/joint"
NBC = T.CHAIN * T.SEQ // T.G
NF = 51
mL = json.load(open("/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"]


def prep_layer(args):
    corpus, L = args
    import lightgbm as lgb
    f = f"{JD}/{corpus}/L{L}.npz"
    if os.path.exists(f):
        return L
    d = np.load(f"{T.OUT}/rows_bandall/{corpus}/L{L}.npz")
    cand = d["cand"].astype(np.int64)
    nb, nc = cand.shape
    assert nc == T.NE
    b = lgb.Booster(model_file=V2)
    p = b.predict(T.feature_matrix(b.feature_name(), corpus, L, band="all", d=d), num_threads=1).reshape(nb, nc)
    X = T.feature_matrix(list(FE), corpus, L, band="all", d=d).reshape(nb, nc, len(FE))
    X = np.concatenate([X, p[..., None]], -1)
    Xe = np.empty_like(X); np.put_along_axis(Xe, cand[..., None], X, 1)
    y = d["ysal"].astype(np.float64) / mL[str(L)]
    ye = np.empty_like(y); np.put_along_axis(ye, cand, y, 1)
    ye[~d["valid"]] = np.nan
    os.makedirs(f"{JD}/{corpus}", exist_ok=True)
    np.savez(f + ".part.npz", X=Xe.astype(np.float32), y=ye.astype(np.float32))
    os.replace(f + ".part.npz", f)
    return L


def transform(X):
    """[.., 14] raw -> network input (log-compressed rates, tok_since_hit, log p)."""
    import torch
    X = torch.nan_to_num(X, nan=0.0)
    rate = [0, 1, 4, 5, 6, 7, 9, 10, 11, 12]
    out = [torch.log1p(X[..., rate].clamp(min=0) * 16.0), X[..., 2:3],
           torch.log1p(X[..., 3:4].clamp(min=0)) / 5.0, torch.log(X[..., 8:9].clamp(min=1e-3)),
           torch.log(X[..., 13:14].clamp(min=1e-4))]
    return torch.cat(out, -1)


def replay(S, fx, fd, hm=0.5):
    nb = S.shape[0]
    serve = np.zeros((nb, T.NE), bool)
    for c0 in range(0, nb, NBC):
        want = fd.copy()
        for k in range(c0, min(c0 + NBC, nb)):
            serve[k] = want
            v = np.where(fx, -np.inf, S[k]).astype(np.float64)
            v = np.where(want, v * (1 + hm), v)
            nw = np.zeros(T.NE, bool); nw[np.argsort(-v, kind="stable")[:NF]] = True
            want = nw & ~fx
    return serve


def build_model(mode, nlayers, d=64):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(14, d)
            self.emb = nn.Parameter(torch.zeros(len(T.LAYERS), T.NE, d))
            enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, nlayers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)

        def forward(self, x, li):
            z = transform(x)
            if mode == "solo":
                z = torch.cat([z[..., :-1], torch.zeros_like(z[..., -1:])], -1)
            h = self.enc(self.inp(z) + self.emb[li])
            r = self.head(h).squeeze(-1)
            return r + (z[..., -1] if mode == "base" else 0.0)       # log mu
    return Net()


def load(corpus):
    Xs, ys, ls = [], [], []
    for i, L in enumerate(T.LAYERS):
        z = np.load(f"{JD}/{corpus}/L{L}.npz")
        Xs.append(z["X"]); ys.append(z["y"]); ls.append(np.full(z["X"].shape[0], i, np.int16))
    return Xs, ys, ls


def train(name, mode, epochs, nlayers, sub):
    import torch
    torch.set_num_threads(int(os.environ.get("JT", "48")))
    torch.manual_seed(0)
    fixed, fdef = T.serve_sets()
    fxm = np.zeros((len(T.LAYERS), T.NE), bool)
    for i, L in enumerate(T.LAYERS):
        fxm[i, fixed[L]] = True
    Xs, ys, ls = load("calib-fit")
    X = np.concatenate([x[::sub] for x in Xs]); y = np.concatenate([v[::sub] for v in ys])
    li = np.concatenate([v[::sub] for v in ls]); del Xs, ys, ls
    ok = np.isfinite(y).all(1)
    X, y, li = torch.from_numpy(X[ok]), torch.from_numpy(y[ok]), torch.from_numpy(li[ok].astype(np.int64))
    fxt = torch.from_numpy(fxm)
    Xh, yh, lh = load("glm52-heldout")
    net = build_model(mode, nlayers)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = X.shape[0]; bs = 256
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    print(f"{name} mode {mode} train samples {n} steps {steps} params {sum(p.numel() for p in net.parameters())}",
          flush=True)

    def tweedie(logmu, yy, m):
        rho = 1.5
        l = -yy * torch.exp((1 - rho) * logmu) / (1 - rho) + torch.exp((2 - rho) * logmu) / (2 - rho)
        return (l * m).sum() / m.sum()

    def evaluate():
        net.eval()
        hot = []; loss = []
        with torch.no_grad():
            for i, L in enumerate(T.LAYERS):
                x = torch.from_numpy(Xh[i]); lt = torch.full((x.shape[0],), i, dtype=torch.long)
                lm = torch.cat([net(x[j:j + 512], lt[j:j + 512]) for j in range(0, x.shape[0], 512)])
                yy = torch.from_numpy(yh[i]); m = (torch.isfinite(yy) & ~fxt[i][None]).float()
                loss.append(float(tweedie(lm, torch.nan_to_num(yy), m)))
                S = np.exp(lm.double().numpy())
                fx = fxm[i]; fd = np.zeros(T.NE, bool)
                fd[[e for e in fdef[L] if e not in set(fixed[L])][:NF]] = True
                sv = replay(S, fx, fd) | fx
                bs_ = np.load(f"{T.OUT}/rows_bandall/glm52-heldout/L{L}.npz")["bsal"].astype(np.float64)
                hot.append(((bs_ * sv).sum() / bs_.sum(), float((sv[1:] & ~sv[:-1]).sum(1).mean())))
        net.train()
        h = np.array(hot)
        return float(np.mean(loss)), float(h[:, 0].mean()), float(h[:, 1].mean()), h[:, 0].tolist()

    t0 = time.time(); step = 0; hist = []
    ev = evaluate()
    print(f"INIT heldout tweedie {ev[0]:.5f} sync hot sal {ev[1] * 100:.2f} churn {ev[2]:.2f}", flush=True)
    for ep in range(epochs):
        perm = torch.randperm(n)
        run = 0.0
        for j in range(0, n - bs + 1, bs):
            idx = perm[j:j + bs]
            yy = y[idx]; m = (~fxt[li[idx]]).float()
            lm = net(X[idx], li[idx])
            loss = tweedie(lm, yy, m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1; run += float(loss)
            if step % 500 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {run / 500:.5f} {time.time() - t0:.0f}s", flush=True)
                run = 0.0
        ev = evaluate()
        hist.append(dict(epoch=ep, heldout_loss=ev[0], hot_sal=ev[1], churn=ev[2]))
        print(f"EPOCH {ep} heldout tweedie {ev[0]:.5f} sync hot sal {ev[1] * 100:.2f} churn {ev[2]:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
        os.makedirs(MD, exist_ok=True)
        torch.save(net.state_dict(), f"{MD}/{name}.pt")
        json.dump(dict(name=name, mode=mode, nlayers=nlayers, sub=sub, epochs=epochs, hist=hist, hot_L=ev[3]),
                  open(f"{MD}/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        with Pool(int(os.environ.get("NPROC", "16"))) as p:
            for r in p.imap_unordered(prep_layer, [(sys.argv[2], L) for L in T.LAYERS]):
                print(r, end=" ", flush=True)
    else:
        a = sys.argv[2:]
        train(a[0], a[1], int(a[2]) if len(a) > 2 else 3, int(a[3]) if len(a) > 3 else 2, int(a[4]) if len(a) > 4 else 2)
