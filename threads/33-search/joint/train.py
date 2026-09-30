#!/usr/bin/env python3
"""T33i joint 256-expert model: GPU trainer.  Per (layer, block) sample: 256 experts x K causal features (jlib.INPUTS,
last = log v2 prediction), residual on v2 (log mu = log p_v2 + head; head zero-init => starts exactly at v2).
  arch: tf (transformer over experts) | mixer (MLP-mixer) | mlp (per-expert, no interaction: control)
  loss: tw (tweedie 1.5 on y64) + a*soft (soft top-51 coverage, metric-aligned) + b*lnet (salience-weighted ListNet)
Train = calib-fit chains 0..27, val = chains 28..31 (model selection / hm tuning); heldout scored only at the end.
  train.py NAME [--arch tf --d 96 --nl 2 --epochs 8 --soft 0 --lnet 0 --tw 1 --target y64 --bs 512 --lr 1e-3]
PRIVATE data; models + scores under /tmp/nestquant/33-search/joint."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

OUT = "/tmp/nestquant/33-search/joint"
LAYERS = list(range(3, 78))
LAYOUT = os.environ.get("LAYOUT", "k26")
NE, NBC = 256, 512
NF = 77 if LAYOUT == "k0" else 51
TRAIN_CH, VAL_CH = range(0, 28), range(28, 32)
MANIFEST = ("/tmp/nestquant/32-gbdt-sal/k0_manifest.json" if LAYOUT == "k0" else
            "/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json")


def sets():
    m = json.load(open(MANIFEST))
    fx = np.zeros((len(LAYERS), NE), bool); fd = np.zeros((len(LAYERS), NE), bool)
    for i, L in enumerate(LAYERS):
        f = sorted(map(int, m["default_allocation"][str(L)])); fx[i, f] = True
        fd[i, [e for e in map(int, m["floating_default"][str(L)]) if e not in set(f)][:NF]] = True
    return fx, fd


class Net(nn.Module):
    def __init__(self, K, arch, d, nl, heads=4, extra=3):
        super().__init__()
        self.arch = arch
        self.inp = nn.Sequential(nn.Linear(K + extra, d), nn.GELU(), nn.Linear(d, d))
        self.emb = nn.Parameter(torch.zeros(len(LAYERS), NE, d))
        self.lemb = nn.Parameter(torch.zeros(len(LAYERS), 1, d))
        if arch == "tf":
            enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(enc, nl, enable_nested_tensor=False)
        elif arch == "mixer":
            self.tok = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), Tmix(NE, 128)) for _ in range(nl)])
            self.ch = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                     for _ in range(nl)])
        else:
            self.ch = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                     for _ in range(nl)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)

    def forward(self, x, lp, li, fxm):
        x = x.float()
        # layout-agnostic context features (no fixed mask: one model serves k26 and k0): v2 rank + gap to v2's
        # 77/78 boundary over all 256 experts; 3rd extra column kept (zeros) for checkpoint compatibility
        kth = lp.topk(78, -1).values[..., 76:78].mean(-1, keepdim=True)
        rank = lp.argsort(-1, descending=True).argsort(-1).float() / NE
        z = torch.cat([x, (lp - kth).clamp(-8, 8)[..., None], rank[..., None], torch.zeros_like(lp)[..., None]], -1)
        h = self.inp(z) + self.emb[li] + self.lemb[li]
        if self.arch == "tf":
            h = self.body(h)
        elif self.arch == "mixer":
            for t, c in zip(self.tok, self.ch):
                h = h + t(h); h = h + c(h)
        else:
            for c in self.ch:
                h = h + c(h)
        return self.head(h).squeeze(-1)                             # residual head (log mu = lp + head)


class Tmix(nn.Module):
    def __init__(self, n, hdim):
        super().__init__()
        self.a = nn.Linear(n, hdim); self.b = nn.Linear(hdim, n)

    def forward(self, h):                                           # mix across experts
        return self.b(Fn.gelu(self.a(h.transpose(1, 2)))).transpose(1, 2)


def tweedie(lm, y, m, rho=1.5):
    l = -y * torch.exp((1 - rho) * lm) / (1 - rho) + torch.exp((2 - rho) * lm) / (2 - rho)
    return (l * m).sum() / m.sum()


def softk(lm, y, m, T):
    """soft top-51 coverage of next-k salience among non-fixed; threshold = mean of 51st/52nd score (differentiable)."""
    s = lm.masked_fill(~m.bool(), -1e4)
    tv = s.topk(NF + 1, -1).values
    tau = tv[..., NF - 1:NF + 1].mean(-1, keepdim=True)
    p = torch.sigmoid((s - tau) / T) * m
    yo = (y * m).topk(NF, -1).values.sum(-1).clamp(min=1e-6)
    return -((p * y).sum(-1) / yo).mean()


def listnet(lm, y, m):
    s = lm.masked_fill(~m.bool(), -1e4)
    t = (y * m) / (y * m).sum(-1, keepdim=True).clamp(min=1e-6)
    return -(t * torch.log_softmax(s, -1)).sum(-1).mean()


@torch.no_grad()
def gpu_replay(S, fx, fd, bsal, hms):
    """S [NL, nch, nb, NE] positive scores, fx/fd [NL, NE], bsal like S -> per hm: sal-hot per layer [NL], churn [NL]."""
    NL, nch, nb, _ = S.shape
    out = []
    for hm in hms:
        want = fd[:, None].expand(NL, nch, NE).clone()
        num = torch.zeros(NL, device=S.device); ch = torch.zeros(NL, device=S.device)
        fxb = fx[:, None].expand(NL, nch, NE)
        for k in range(nb):
            num += (bsal[:, :, k] * (want | fxb)).sum((1, 2))
            v = S[:, :, k].masked_fill(fxb, -float("inf"))
            v = torch.where(want, v * (1 + hm), v)
            nw = torch.zeros_like(want).scatter_(-1, v.topk(NF, -1).indices, True)
            if k + 1 < nb:
                ch += (nw & ~want).sum((1, 2)).float()
            want = nw
        out.append(((num / bsal.sum((1, 2, 3))).cpu().numpy(), (ch / (nch * (nb - 1))).cpu().numpy()))
    return out


def interp(pts, c):
    pts = sorted(pts); x = [p[0] for p in pts]; y = [p[1] for p in pts]
    return float(np.interp(c, x, y)) if x[0] <= c <= x[-1] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--arch", default="tf"); ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--nl", type=int, default=2); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tw", type=float, default=1.0); ap.add_argument("--soft", type=float, default=0.0)
    ap.add_argument("--lnet", type=float, default=0.0); ap.add_argument("--T", type=float, default=0.3)
    ap.add_argument("--target", default="y64"); ap.add_argument("--drop", default="", help="comma input idx to zero")
    ap.add_argument("--noresid", action="store_true"); ap.add_argument("--noemb", action="store_true")
    ap.add_argument("--score", default="glm52-heldout,calib-fit", help="streams to dump scores for at the end")
    ap.add_argument("--budget", type=float, default=0, help="train wall minutes (cosine lr by time)")
    ap.add_argument("--valmin", type=float, default=5.0); ap.add_argument("--dev", default="cuda"); ap.add_argument("--maxn", type=int, default=0)
    a = ap.parse_args()
    dev = a.dev
    torch.manual_seed(0)
    fx_np, fd_np = sets()
    fx = torch.from_numpy(fx_np).to(dev); fd = torch.from_numpy(fd_np).to(dev)
    t0 = time.time()
    Xt, Yt, Lt, Xv, Bv, Pt, Pv = [], [], [], [], [], [], []
    for i, L in enumerate(LAYERS):
        z = np.load(f"{OUT}/feat/calib-fit/L{L}.npz")
        P = np.log(np.maximum(np.load(f"{OUT}/scores/v2_calib-fit/L{L}.npy"), 1e-30)).astype(np.float32)
        X, y = z["X"], z[a.target]
        nb = X.shape[0]; ch = np.arange(nb) // NBC
        tr = np.isin(ch, TRAIN_CH) & np.isfinite(y).all(1)
        Xt.append(torch.from_numpy(X[tr]).to(dev)); Yt.append(torch.from_numpy(y[tr].astype(np.float16)).to(dev))
        Lt.append(torch.full((int(tr.sum()),), i, dtype=torch.int16))
        va = np.isin(ch, VAL_CH)
        Xv.append(torch.from_numpy(X[va])); Pt.append(torch.from_numpy(P[tr]).to(dev)); Pv.append(torch.from_numpy(P[va]))
        Bv.append(torch.from_numpy(np.load(f"{OUT}/blk/calib-fit/L{L}.npz")["bsal"][va]))
    Xt = torch.cat(Xt).to(dev); Yt = torch.cat(Yt).to(dev); Lt = torch.cat(Lt).to(dev).long()
    Pt = torch.cat(Pt).to(dev); Pv = torch.stack(Pv).to(dev)
    Xv = torch.stack(Xv).to(dev); Bv = torch.stack(Bv).to(dev).float()          # [NL, nbv, NE]
    K = Xt.shape[-1]
    drop = [int(v) for v in a.drop.split(",") if v]
    if a.maxn:
        sel = torch.randperm(Xt.shape[0], device=dev)[:a.maxn]
        Xt, Yt, Lt, Pt = Xt[sel], Yt[sel], Lt[sel], Pt[sel]
    n = Xt.shape[0]
    print(f"{a.name}: train samples {n} K {K} load {time.time() - t0:.0f}s", flush=True)
    net = Net(K, a.arch, a.d, a.nl).to(dev)
    if a.noemb:                                   # no (layer, expert) identity embedding (transfer rule)
        net.emb.requires_grad_(False)
    if a.noresid:
        pass
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (n // a.bs)
    print(f"params {sum(p.numel() for p in net.parameters())} steps {steps}", flush=True)
    hms = [0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0, 25.0]

    def prep_x(x):
        if drop:
            x = x.clone(); x[..., drop] = 0
        return x

    def fwd(x, lp, li):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            r = net(prep_x(x), lp, li, fx[li]).float()
        return (0.0 if a.noresid else lp) + r

    @torch.no_grad()
    def score(X, P):              # X [NL, nb, NE, K] -> positive scores
        net.eval()
        out = []
        for i in range(X.shape[0]):
            li = torch.full((X.shape[1],), i, dtype=torch.long, device=dev)
            out.append(torch.cat([fwd(X[i, j:j + 1024], P[i, j:j + 1024], li[j:j + 1024])
                                  for j in range(0, X.shape[1], 1024)]))
        net.train()
        return torch.exp(torch.stack(out).clamp(max=30))

    def val():
        S = score(Xv, Pv)
        nbv = S.shape[1] // len(VAL_CH)
        r = gpu_replay(S.view(len(LAYERS), len(VAL_CH), nbv, NE), fx, fd, Bv.view(len(LAYERS), len(VAL_CH), nbv, NE), hms)
        pts = [(float(c.mean()), float(s.mean())) for s, c in r]
        return pts, interp(pts, 2.78), interp(pts, 3.2)

    pts, v278, v32 = val()
    print(f"INIT (=v2) val: " + " ".join(f"{c:.2f}/{s * 100:.2f}" for c, s in pts) +
          f"  @2.78 {v278 * 100:.2f} @3.2 {v32 * 100:.2f}", flush=True)
    hist = [dict(epoch=-1, pts=pts, v278=v278, v32=v32)]
    best = (-1, None)
    step = 0; ts = time.time(); tv = ts; budget = a.budget * 60 if a.budget else None
    run = torch.zeros(3, device=dev); cnt = 0; done = False; ep = 0
    while not done:
        perm = torch.randperm(n, device=dev)
        for j in range(0, n - a.bs + 1, a.bs):
            frac = (time.time() - ts) / budget if budget else step / steps
            if frac >= 1:
                done = True; break
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + np.cos(np.pi * frac))
            idx = perm[j:j + a.bs]
            x, y, li = Xt[idx], Yt[idx].float(), Lt[idx]
            m = (~fx[li]).float()
            lm = fwd(x, Pt[idx], li)
            parts = [tweedie(lm, y, m) if a.tw else lm.new_zeros(()),
                     softk(lm, y, m, a.T) if a.soft else lm.new_zeros(()),
                     listnet(lm, y, m) if a.lnet else lm.new_zeros(())]
            loss = a.tw * parts[0] + a.soft * parts[1] + a.lnet * parts[2]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            run += torch.stack([p.detach().float() for p in parts]); cnt += 1
            if time.time() - tv > a.valmin * 60:
                break
        else:
            ep += 1
        tv = time.time()
        rr = run.cpu().numpy() / max(cnt, 1); run.zero_(); cnt = 0
        pts, v278, v32 = val()
        hist.append(dict(step=step, epoch=ep, pts=pts, v278=v278, v32=v32, loss=rr.tolist()))
        print(f"step {step} ep {ep + j / n:.2f} loss tw {rr[0]:.4f} soft {rr[1]:.4f} lnet {rr[2]:.4f} | val " +
              " ".join(f"{c:.2f}/{s * 100:.2f}" for c, s in pts) + f"  @2.78 {v278 * 100:.2f} @3.2 {v32 * 100:.2f}"
              f"  {time.time() - t0:.0f}s", flush=True)
        key = v32 if v32 == v32 else -0.5
        if key > best[0]:
            v32 = key
            best = (v32, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if not budget and step >= steps:
            done = True
    net.load_state_dict(best[1])
    os.makedirs(f"{OUT}/models", exist_ok=True)
    torch.save(dict(state=best[1], args=vars(a), K=K), f"{OUT}/models/{a.name}.pt")
    json.dump(dict(args=vars(a), hist=hist, best_val32=best[0]), open(f"{OUT}/models/{a.name}.json", "w"), indent=1)
    del Xt, Yt
    torch.cuda.empty_cache()
    for s in [v for v in a.score.split(",") if v]:
        od = f"{OUT}/scores/{a.name}_{s}"; os.makedirs(od, exist_ok=True)
        for i, L in enumerate(LAYERS):
            X = torch.from_numpy(np.load(f"{OUT}/feat/{s}/L{L}.npz")["X"]).to(dev)
            P = torch.from_numpy(np.log(np.maximum(np.load(f"{OUT}/scores/v2_{s}/L{L}.npy"), 1e-30)).astype(np.float32)).to(dev)
            np.save(f"{od}/L{L}.npy", score_one(X, P, i, fwd, net))
    # serve cost: one refresh = 75 layers x 256 experts, batch 75, GPU fwd (bf16 autocast)
    xs = Xv[:, 0].contiguous(); ps_ = Pv[:, 0].contiguous(); li = torch.arange(len(LAYERS), device=dev)
    net.eval(); tt = []
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    with torch.no_grad():
        for r_ in range(60):
            sync(); t1 = time.time(); fwd(xs, ps_, li); sync()
            tt.append(time.time() - t1)
    print(f"serve fwd 75 layers: {np.median(tt[10:]) * 1e3:.2f} ms ({dev}, {torch.get_num_threads()} thr, batch 75)", flush=True)
    print(f"done best val@3.2 {best[0] * 100:.2f} {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def score_one(X, P, i, fwd, net):
    net.eval()
    li = torch.full((X.shape[0],), i, dtype=torch.long, device=X.device)
    lm = torch.cat([fwd(X[j:j + 1024], P[j:j + 1024], li[j:j + 1024]) for j in range(0, X.shape[0], 1024)])
    return torch.exp(lm.clamp(max=30)).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    main()
