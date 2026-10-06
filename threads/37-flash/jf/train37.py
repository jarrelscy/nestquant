#!/usr/bin/env python3
"""T37 port of threads/33-search/joint/train.py (the jF trainer).  Per (layer, block) sample: 288 experts x 22 causal
features (jlib37.INPUTS, last = log v2 prediction), residual on v2 (log mu = log p_v2 + head; head zero-init => starts
exactly at v2).  Shipped GLM-5.3 config "jF" = --arch tf --d 96 --nl 2 --tw 1 --noemb --budget 10 (Tweedie 1.5 on y64).
Diffs vs GLM-5.3 (minimal):
  * NE 288, layers 3-44, NF 48 floating + 19 fixed (fixed_set.json); the loss mask / replay exclude the fixed experts.
  * boundary feature: gap to the v2 NF/NF+1 boundary among the NON-fixed experts (GLM-5.3 k0 had no fixed set:
    topk(78)[76:78] over all experts); the 3rd extra input column (zeros on GLM-5.3) carries the fixed mask.
  * data: $OUT/feat/train (every J37_SUB-th block of the train split) / feat/val (whole chains) from feat37.py;
    chains have variable length, so the validation replay is padded + masked.
  * mixed data: train rows = decode + a prefill minority (feat37 / chains.json mix, default 20%); model selection
    on the DECODE val chains only, prefill val chains (--valblk-pf) logged as a secondary metric.
  * model selection: POOLED sal-hot (salience summed over layers; tracks KLD per T32) at matched churn = v2's val
    churn at hm --cref-hm (0.7 = GLM-5.3 jF serve hm); the layer mean is printed too.
  * --dev cpu works (no autocast); the run script defaults to CPU while the GPUs are busy.
  train37.py NAME [--arch tf --d 96 --nl 2 --tw 1 --noemb --budget 10 --dev cuda|cpu --score val,test]
PRIVATE data; models + scores under $OUT (/tmp/nestquant/37-flash/jf)."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import jlib37 as J

OUT = J.OUT
LAYERS = J.LAYERS
NE, NF = J.NE, J.NF


def sets():
    fx = np.zeros((len(LAYERS), NE), bool); fd = np.zeros((len(LAYERS), NE), bool)
    for i, L in enumerate(LAYERS):
        fx[i], fd[i] = J.masks(L)
    return fx, fd


class Net(nn.Module):
    def __init__(self, K, arch, d, nl, heads=4, extra=3, nf=None, ne=None, nlayers=None):
        super().__init__()
        self.arch = arch
        self.nf = nf or NF; self.ne = ne or NE
        nlay = nlayers or len(LAYERS)
        self.inp = nn.Sequential(nn.Linear(K + extra, d), nn.GELU(), nn.Linear(d, d))
        self.emb = nn.Parameter(torch.zeros(nlay, self.ne, d))
        self.lemb = nn.Parameter(torch.zeros(nlay, 1, d))
        if arch == "tf":
            enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(enc, nl, enable_nested_tensor=False)
        elif arch == "mixer":
            self.tok = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), Tmix(self.ne, 128)) for _ in range(nl)])
            self.ch = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                     for _ in range(nl)])
        else:
            self.ch = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                     for _ in range(nl)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)

    def forward(self, x, lp, li, fxm):
        x = x.float()
        # context features: v2 rank over all experts + gap to v2's NF/NF+1 boundary among the non-fixed experts;
        # 3rd extra column = fixed mask (fixed experts are context only: excluded from loss and selection)
        lpm = lp.masked_fill(fxm, -1e4)
        kth = lpm.topk(self.nf + 1, -1).values[..., self.nf - 1:self.nf + 1].mean(-1, keepdim=True)
        rank = lp.argsort(-1, descending=True).argsort(-1).float() / self.ne
        z = torch.cat([x, (lp - kth).clamp(-8, 8)[..., None], rank[..., None], fxm.to(lp.dtype)[..., None]], -1)
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
    """soft top-NF coverage of next-k salience among non-fixed; threshold = mean of NF-th/NF+1-th score."""
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
def replay_pad(S, fx, fd, B, valid, hms):
    """S, B [NL, nch, nbmax, NE] (B zero on pads), valid [nch, nbmax] bool, fx/fd [NL, NE] -> per hm:
    (num [NL], den [NL], churn [NL] per within-chain transition).  Same semantics as T33 gpu_replay / jlib37.replay."""
    NL, nch, nb, _ = S.shape
    den = B.sum((1, 2, 3)).double()
    ntr = (valid[:, 1:] & valid[:, :-1]).sum().clamp(min=1).double()
    fxb = fx[:, None].expand(NL, nch, NE)
    out = []
    for hm in hms:
        want = fd[:, None].expand(NL, nch, NE).clone()
        num = torch.zeros(NL, dtype=torch.float64, device=S.device); ch = torch.zeros_like(num)
        for k in range(nb):
            num += (B[:, :, k] * (want | fxb)).sum((1, 2)).double()
            v = S[:, :, k].masked_fill(fxb, -float("inf"))
            v = torch.where(want, v * (1 + hm), v)
            nw = torch.zeros_like(want).scatter_(-1, v.topk(NF, -1).indices, True)
            if k + 1 < nb:
                ch += ((nw & ~want).sum(-1).double() * (valid[:, k + 1] & valid[:, k])).sum(-1)
            want = nw
        out.append((num.cpu().numpy(), den.cpu().numpy(), (ch / ntr).cpu().numpy()))
    return out


def curve(res, agg):
    pts = []
    for num, den, ch in res:
        s = num.sum() / den.sum() if agg == "pooled" else float(np.mean(num / den))
        pts.append((float(ch.mean()), float(s)))
    return pts


def interp(pts, c):
    pts = sorted(pts); x = [p[0] for p in pts]; y = [p[1] for p in pts]
    return float(np.interp(c, x, y)) if x[0] <= c <= x[-1] else float("nan")


def val_chains(cap, kind=0, seed=0):
    """random whole val chains of one kind (0 decode / 1 prefill) up to `cap` blocks per layer."""
    meta = json.load(open(f"{OUT}/blk/val/meta.json"))
    bs = meta["bstart"]; kd = meta.get("kind", [0] * (len(bs) - 1))
    sg = [(a, b) for a, b, k in zip(bs[:-1], bs[1:], kd) if b > a and k == kind]
    if not sg:
        return []
    order = np.random.default_rng(seed).permutation(len(sg))
    sel, n = [], 0
    for i in order:
        if sel and n + sg[i][1] - sg[i][0] > cap:
            continue
        sel.append(int(i)); n += sg[i][1] - sg[i][0]
    return [sg[i] for i in sorted(sel)]


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
    ap.add_argument("--score", default="val,test", help="splits to dump full scores for at the end")
    ap.add_argument("--budget", type=float, default=0, help="train wall minutes (cosine lr by time)")
    ap.add_argument("--valmin", type=float, default=5.0)
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--maxn", type=int, default=0)
    ap.add_argument("--valblk", type=int, default=int(os.environ.get("J37_VALBLK", "4096")),
                    help="DECODE val blocks per layer used for model selection (whole chains)")
    ap.add_argument("--valblk-pf", type=int, default=int(os.environ.get("J37_VALBLK_PF", "2048")),
                    help="PREFILL val blocks per layer (secondary metric, not used for selection; 0 = off)")
    ap.add_argument("--cref-hm", type=float, default=0.7, help="matched churn = v2 val churn at this hm")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "20")))
    a = ap.parse_args()
    assert a.target == "y64", "feat37 stores y64 only"
    dev = a.dev
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    fx_np, fd_np = sets()
    fx = torch.from_numpy(fx_np).to(dev); fd = torch.from_numpy(fd_np).to(dev)
    # validation sets: DECODE val chains = model selection; PREFILL val chains = secondary (logged only)
    VS, vidx, off = {}, [], 0
    for kn, kd, cap in (("decode", 0, a.valblk), ("prefill", 1, a.valblk_pf)):
        vsg = val_chains(cap, kd) if cap > 0 else []
        if not vsg:
            continue
        nch, nbmax = len(vsg), max(e - s for s, e in vsg)
        pad = np.full((nch, nbmax), -1, np.int64)
        for c, (s, e) in enumerate(vsg):
            pad[c, :e - s] = np.arange(off, off + e - s); off += e - s
            vidx.append(np.arange(s, e))
        VS[kn] = dict(nch=nch, nb=int((pad >= 0).sum()), valid=torch.from_numpy(pad >= 0).to(dev),
                      padi=torch.from_numpy(np.maximum(pad, 0)).to(dev))
    assert "decode" in VS, "no decode val chains: model selection is decode-only"
    vidx = np.concatenate(vidx)
    t0 = time.time()
    Xt, Yt, Lt, Xv, Bv, Pt, Pv, Kt = [], [], [], [], [], [], [], []
    for i, L in enumerate(LAYERS):
        z = np.load(f"{OUT}/feat/train/L{L}.npz")
        y = z[a.target]
        tr = np.isfinite(y.astype(np.float32)).all(1)
        Xt.append(torch.from_numpy(z["X"][tr])); Yt.append(torch.from_numpy(y[tr].astype(np.float16)))
        Pt.append(torch.from_numpy(np.log(np.maximum(z["P"][tr], 1e-30)).astype(np.float32)))
        Lt.append(torch.full((int(tr.sum()),), i, dtype=torch.int16))
        Kt.append(z["kind"][tr] if "kind" in z.files else np.zeros(int(tr.sum()), np.int8))
        zv = np.load(f"{OUT}/feat/val/L{L}.npz")
        assert (zv["keep"] == np.arange(len(zv["keep"]))).all()
        Xv.append(torch.from_numpy(zv["X"][vidx]))
        Pv.append(torch.from_numpy(np.log(np.maximum(zv["P"][vidx], 1e-30)).astype(np.float32)))
        Bv.append(torch.from_numpy(np.load(f"{OUT}/blk/val/L{L}.npz")["bsal"][vidx]))
        del z, zv
    Xt = torch.cat(Xt).to(dev); Yt = torch.cat(Yt).to(dev); Lt = torch.cat(Lt).to(dev).long()
    Pt = torch.cat(Pt).to(dev); Pv = torch.stack(Pv).to(dev)
    Xv = torch.stack(Xv).to(dev); Bv = torch.stack(Bv).to(dev).float()          # [NL, nbv, NE]
    for v in VS.values():
        v["Bp"] = Bv[:, v["padi"]] * v["valid"][None, :, :, None]               # [NL, nch, nbmax, NE]
    del Bv
    Kt = np.concatenate(Kt)
    K = Xt.shape[-1]
    drop = [int(v) for v in a.drop.split(",") if v]
    if a.maxn:
        sel = torch.randperm(Xt.shape[0], device=dev)[:a.maxn]
        Xt, Yt, Lt, Pt = Xt[sel], Yt[sel], Lt[sel], Pt[sel]
    n = Xt.shape[0]
    pf_share = float((Kt == 1).mean()) if len(Kt) and not a.maxn else float("nan")
    print(f"{a.name}: train samples {n} (prefill {100 * pf_share:.1f}%) K {K} val "
          + " ".join(f"{k} {v['nch']} chains {v['nb']} blocks/layer" for k, v in VS.items())
          + f" load {time.time() - t0:.0f}s dev {dev} threads {torch.get_num_threads()}", flush=True)
    net = Net(K, a.arch, a.d, a.nl).to(dev)
    if a.noemb:                                   # no (layer, expert) identity embedding (transfer rule)
        net.emb.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    steps = max(1, a.epochs * (n // a.bs))
    print(f"params {sum(p.numel() for p in net.parameters())} steps {steps}", flush=True)
    hms = [0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0, 25.0]

    def prep_x(x):
        if drop:
            x = x.clone(); x[..., drop] = 0
        return x

    def fwd(x, lp, li):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
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

    crefs = {}

    def val():
        """-> {kind: (pooled curve, lmean curve, pooled@cref, lmean@cref, pooled@1.5cref)}; cref per kind = v2's
        churn at --cref-hm on that kind (fixed at INIT).  Selection key = decode pooled@cref."""
        S = score(Xv, Pv)
        res = {}
        for kn, v in VS.items():
            r = replay_pad(S[:, v["padi"]], fx, fd, v["Bp"], v["valid"], hms)
            pp, pl = curve(r, "pooled"), curve(r, "lmean")
            if kn not in crefs:
                crefs[kn] = pp[hms.index(a.cref_hm)][0]
            c = crefs[kn]
            res[kn] = (pp, pl, interp(pp, c), interp(pl, c), interp(pp, 1.5 * c))
        return res

    def fmt(res):
        out = []
        for kn, (pp, pl, kp, kl, k15) in res.items():
            c = crefs[kn]
            out.append((f"[{kn}] " + " ".join(f"{ch:.2f}/{s * 100:.2f}" for ch, s in pp) if kn == "decode" else f"[{kn}]")
                       + f" pooled@{c:.2f} {kp * 100:.2f} (lmean {kl * 100:.2f}) pooled@{1.5 * c:.2f} {k15 * 100:.2f}")
        return " | ".join(out)

    def hrec(res):
        return {kn: dict(pooled=pp, lmean=pl, key=kp, key_lmean=kl, key15=k15, cref=crefs[kn])
                for kn, (pp, pl, kp, kl, k15) in res.items()}

    vr = val()
    kp = vr["decode"][2]
    print(f"INIT (=v2) val: {fmt(vr)}", flush=True)
    hist = [dict(epoch=-1, key=kp, key_prefill=vr["prefill"][2] if "prefill" in vr else None, val=hrec(vr))]
    best = (kp, {k: v.detach().clone() for k, v in net.state_dict().items()})
    step = 0; ts = time.time(); tv = ts; budget = a.budget * 60 if a.budget else None
    run = torch.zeros(3, device=dev); cnt = 0; done = False; ep = 0; j = 0
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
        vr = val()
        kp = vr["decode"][2]
        hist.append(dict(step=step, epoch=ep, key=kp, key_prefill=vr["prefill"][2] if "prefill" in vr else None,
                         val=hrec(vr), loss=rr.tolist()))
        print(f"step {step} ep {ep + j / max(n, 1):.2f} loss tw {rr[0]:.4f} soft {rr[1]:.4f} lnet {rr[2]:.4f} | val "
              f"{fmt(vr)}  {time.time() - t0:.0f}s", flush=True)
        key = kp if kp == kp else -0.5
        if key > best[0]:
            best = (key, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if not budget and step >= steps:
            done = True
    net.load_state_dict(best[1])
    os.makedirs(f"{OUT}/models", exist_ok=True)
    fsha = hashlib.sha256(open(J.FIXED_JSON, "rb").read()).hexdigest()
    meta = dict(nf=NF, ne=NE, layers=LAYERS, n_fixed=int(fx_np.sum(1).max()), fixed_set_sha256=fsha,
                inputs=J.INPUTS, extra=["gap_to_v2_nonfixed_boundary", "v2_rank", "fixed_mask"])
    torch.save(dict(state=best[1], args=vars(a), K=K, **meta), f"{OUT}/models/{a.name}.pt")
    json.dump(dict(args=vars(a), hist=hist, best_key=best[0], selection="decode val pooled sal-hot @ v2 decode churn",
                   cref=crefs["decode"], crefs=crefs, train_prefill_frac=pf_share, hms=hms, **meta),
              open(f"{OUT}/models/{a.name}.json", "w"), indent=1)
    del Xt, Yt
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    for s in [v for v in a.score.split(",") if v]:
        od = f"{OUT}/scores/{a.name}_{s}"; os.makedirs(od, exist_ok=True)
        for i, L in enumerate(LAYERS):
            z = np.load(f"{OUT}/feat/{s}/L{L}.npz")
            X = torch.from_numpy(z["X"]).to(dev)
            P = torch.from_numpy(np.log(np.maximum(z["P"], 1e-30)).astype(np.float32)).to(dev)
            np.save(f"{od}/L{L}.npy", score_one(X, P, i, fwd, net))
        print(f"scored {s} {time.time() - t0:.0f}s", flush=True)
    # serve cost: one refresh = all layers x 288 experts, batch NL
    xs = Xv[:, 0].contiguous(); ps_ = Pv[:, 0].contiguous(); li = torch.arange(len(LAYERS), device=dev)
    net.eval(); tt = []
    sync = torch.cuda.synchronize if dev.startswith("cuda") else (lambda: None)
    with torch.no_grad():
        for r_ in range(30):
            sync(); t1 = time.time(); fwd(xs, ps_, li); sync()
            tt.append(time.time() - t1)
    print(f"serve fwd {len(LAYERS)} layers: {np.median(tt[5:]) * 1e3:.2f} ms ({dev}, {torch.get_num_threads()} thr)", flush=True)
    print(f"done best DECODE val pooled@{crefs['decode']:.2f} {best[0] * 100:.2f} (init {hist[0]['key'] * 100:.2f}) "
          f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def score_one(X, P, i, fwd, net):
    net.eval()
    li = torch.full((X.shape[0],), i, dtype=torch.long, device=X.device)
    lm = torch.cat([fwd(X[j:j + 1024], P[j:j + 1024], li[j:j + 1024]) for j in range(0, X.shape[0], 1024)])
    return torch.exp(lm.clamp(max=30)).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    main()
