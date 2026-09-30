#!/usr/bin/env python3
"""T33i jF-H / jF-R / jF-noresid trainer (coordinator 2026-09-30): GBDT-free joint 256-expert set model on
model-generated text only (fp8dec decode + sm120tf), task-level holdout, checkpoint pick on held-out VAL tasks
(never calib), final test on untouched tasks.  Arms (flags):
  jF-noresid   routing EMAs only (jlib.INPUTS minus the v2 column), no v2 anywhere, head = log salience directly
  jF-R         + per-expert router-logit history features (fp8dec dump; --rlog)
  jF-H         + hidden-state context tokens (cross-attention; --ctx) and router-row content keys (--rkey)
  jF-H+R       both;  control: any of the above + --resid (log v2 residual + v2 rank/gap columns, = jF recipe)
Data source "sm120tf" (T33j tmp_tfX / feat/sm120tf_s10; learning curve by --train-tasks / --frac) or "fp8dec"
(fp8blk.py output: feat/fp8dec/, ctx/, rlog/).  PRIVATE; models + scores under /tmp/nestquant/33-search/joint.
  LAYOUT=k0 trainh.py NAME --src sm120tf --train-tasks a,b --test-tasks c,d [--val-tasks ...] [--frac 0.25]
         [--resid] [--rkey] [--ctx {hid,rsvd256,pca256}] [--rlog] [--budget MIN] [--dev cpu|cuda]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as TR                                  # noqa: E402  (before jlib: it prepends 32-gbdt-sal/)
import jlib as J                                    # noqa: E402

OUT = J.OUT
LAYERS, NE, NL = J.LAYERS, J.NE, len(J.LAYERS)
KV2 = J.INPUTS.index("v2")                          # log v2 column in jlib net inputs
TD = f"{OUT}/tmp_tfX"
TF_META = "/tmp/nestquant/32-gbdt-sal/private/sm120/blk/sm120tf/meta.json"
RW = f"{OUT}/router_w.npz"


class Block(nn.Module):
    """pre-norm: expert tokens cross-attend to context tokens (optional), then self-attend + FFN."""

    def __init__(self, d, heads, cross):
        super().__init__()
        self.cross = cross
        if cross:
            self.nq, self.nc = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, c=None, cmask=None):
        if self.cross and c is not None:
            q = self.nq(h); kv = self.nc(c)
            h = h + self.ca(q, kv, kv, key_padding_mask=cmask, need_weights=False)[0]
        x = self.n1(h)
        h = h + self.sa(x, x, x, need_weights=False)[0]
        return h + self.ff(self.n2(h))


class NetH(nn.Module):
    def __init__(self, K, d=96, nl=2, heads=4, resid=False, rkey=0, ctx_dim=0, ctx_src=0, ctx_len=0):
        super().__init__()
        self.resid, self.ctx_dim = resid, ctx_dim
        extra = 2 if resid else 0                  # v2 gap-to-boundary + rank (the jF context columns)
        self.inp = nn.Sequential(nn.Linear(K + extra, d), nn.GELU(), nn.Linear(d, d))
        self.lemb = nn.Parameter(torch.zeros(NL, 1, d))
        self.key = None
        if rkey:                                   # router-row content key: W_r[L,e] (fixed basis rkey-d) -> d
            self.key = nn.Sequential(nn.LayerNorm(rkey), nn.Linear(rkey, d))
            self.register_buffer("keys", torch.zeros(NL, NE, rkey), persistent=False)
        if ctx_dim:
            self.cin = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, d))
            self.cpos = nn.Parameter(torch.zeros(ctx_len, d))
            self.csrc = nn.Parameter(torch.zeros(ctx_src, 1, d))
            self.cl = nn.Parameter(torch.zeros(NL, 1, d))      # target layer -> context query bias
        self.body = nn.ModuleList([Block(d, heads, bool(ctx_dim)) for _ in range(nl)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)

    def forward(self, x, li, lp=None, ctx=None, cmask=None):
        """x [B,NE,K]; li [B]; lp [B,NE] log v2 (resid only); ctx [B,nsrc,C,ctx_dim] (+cmask [B,C] True=pad)."""
        x = x.float()
        if self.resid:
            kth = lp.topk(78, -1).values[..., 76:78].mean(-1, keepdim=True)
            rank = lp.argsort(-1, descending=True).argsort(-1).float() / NE
            x = torch.cat([x, (lp - kth).clamp(-8, 8)[..., None], rank[..., None]], -1)
        h = self.inp(x) + self.lemb[li]
        if self.key is not None:
            h = h + self.key(self.keys[li])
        c = m = None
        if self.ctx_dim:
            B, S, C, _ = ctx.shape
            c = (self.cin(ctx.float()) + self.cpos[:C] + self.csrc[:S, None]).reshape(B, S * C, -1) + self.cl[li]
            m = None if cmask is None else cmask.repeat(1, S)
        for b in self.body:
            h = b(h, c, m)
        r = self.head(h).squeeze(-1)
        return (lp + r) if self.resid else r


def router_keys(k):
    """[NL, NE, k] router rows projected on the top-k router-stack SVD basis (router_svd.py), row-normalised."""
    z = np.load(RW); W = z["W"].astype(np.float32) * z["ln"][:, None, :]
    V = np.load(f"{OUT}/hproj/router_svd.npz")["V"][:, :k]
    K = W @ V
    return torch.from_numpy(K / np.linalg.norm(W, axis=-1, keepdims=True))


# --------------------------------------------------------------------------------------------- data: sm120tf
def tf_tasks():
    m = json.load(open(TF_META))
    return [(n, s, e) for n, s, e in zip(m["chains"], m["bstart"][:-1], m["bstart"][1:]) if e > s]


def tf_train_rows(tasks, frac, cols, dev, resid):
    """feat/sm120tf_s10 rows of the given tasks (first frac of each task's blocks) -> X, y, li, lp."""
    TK = tf_tasks(); names = [t[0] for t in TK]
    tid = {names.index(t): TK[names.index(t)] for t in tasks}
    Xs, Ys, Ls, Ps, ntok = [], [], [], [], 0
    for i, L in enumerate(LAYERS):
        z = np.load(f"{OUT}/feat/sm120tf_s10/L{L}.npz")
        keep = np.zeros(len(z["task"]), bool)
        for t, (_, s, e) in tid.items():
            keep |= (z["task"] == t) & (z["blk"] < s + frac * (e - s))
        Xs.append(torch.from_numpy(z["X"][keep][..., cols]).to(dev)); Ys.append(torch.from_numpy(z["y64"][keep]).to(dev))
        Ls.append(torch.full((int(keep.sum()),), i, dtype=torch.int16, device=dev))
        if resid:
            Ps.append(torch.from_numpy(np.log(np.maximum(z["P"][keep], 1e-30)).astype(np.float32)).to(dev))
    ntok = int(sum(frac * (e - s) for _, s, e in tid.values()) * J.G)
    return torch.cat(Xs), torch.cat(Ys), torch.cat(Ls).long(), (torch.cat(Ps) if resid else None), ntok


def tf_eval_cache(tasks, nblk):
    """first nblk blocks of each task (contiguous chain) for every layer -> cache under evalc/ (X all 22 cols, P, bsal)."""
    TK = tf_tasks(); names = [t[0] for t in TK]
    tag = "_".join(sorted(tasks)) + f"_{nblk}"
    d0 = f"{OUT}/evalc/sm120tf_{tag}"
    if not os.path.exists(f"{d0}/done"):
        os.makedirs(d0, exist_ok=True)
        rng = [(TK[names.index(t)][1], min(TK[names.index(t)][2], TK[names.index(t)][1] + nblk)) for t in tasks]
        idx = np.concatenate([np.arange(s, e) for s, e in rng])
        for L in LAYERS:
            f = f"{d0}/L{L}.npz"
            if os.path.exists(f):
                continue
            z = np.load(f"{TD}/L{L}.npz"); d = J.load("sm120tf", L)
            np.savez(f + ".part.npz", X=z["X"][idx], P=z["P"][idx], bsal=d["bsal"][idx].astype(np.float32))
            os.replace(f + ".part.npz", f)
        sg, o = [], 0
        for s, e in rng:
            sg.append((o, o + e - s)); o += e - s
        json.dump(dict(sg=sg, tasks=tasks, nblk=nblk), open(f"{d0}/meta.json", "w"))
        open(f"{d0}/done", "w").close()
    return d0


def replay_pts(S, fx, fd, bsal, sg, hms):
    """S / bsal [NL, nb, NE] torch (one device), sg list of chains -> [(churn mean over layers, sal-hot lmean,
    sal-hot pooled)] per hm (sync lag 0, k0, churn within chains)."""
    out = []
    for hm in hms:
        num = torch.zeros(NL, dtype=torch.float64, device=S.device); den = torch.zeros_like(num)
        cs = torch.zeros_like(num); cn = 0
        for s, e in sg:
            want = fd.clone()
            for k in range(s, e):
                num += (bsal[:, k].double() * (want | fx)).sum(-1); den += bsal[:, k].double().sum(-1)
                v = S[:, k].masked_fill(fx, -float("inf"))
                v = torch.where(want, v * (1 + hm), v)
                nw = torch.zeros_like(want).scatter_(-1, v.topk(J.NF, -1).indices, True)
                if k + 1 < e:
                    cs += (nw & ~want).sum(-1).double()
                want = nw
            cn += e - s - 1
        out.append((float((cs / cn).mean()), float((num / den).mean()), float(num.sum() / den.sum())))
    return out


def at(pts, c, j):
    x, y = zip(*sorted((p[0], p[j]) for p in pts))
    return float(np.interp(c, x, y)) if x[0] <= c <= x[-1] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--src", default="sm120tf", choices=["sm120tf", "fp8dec"])
    ap.add_argument("--train-tasks", default=""); ap.add_argument("--test-tasks", default="")
    ap.add_argument("--val-tasks", default="", help="checkpoint pick (held-out tasks); empty = last checkpoint")
    ap.add_argument("--frac", type=float, default=1.0, help="first frac of each train task's blocks (token curve)")
    ap.add_argument("--eval-blocks", type=int, default=4096, help="contiguous blocks per test/val task")
    ap.add_argument("--d", type=int, default=96); ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--budget", type=float, default=0, help="train wall minutes (cosine by time; overrides steps)")
    ap.add_argument("--max-epochs", type=float, default=0)
    ap.add_argument("--nval", type=int, default=4, help="val evaluations during training (if --val-tasks)")
    ap.add_argument("--resid", action="store_true", help="control: v2 residual + v2 columns (GBDT back in)")
    ap.add_argument("--rkey", type=int, default=0, help="router-row content keys, basis dim (e.g. 256)")
    ap.add_argument("--dev", default="cuda"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--hms", default="0.1,0.2,0.35,0.5,0.7,1.0,1.5,2.5,4.0")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    dev = a.dev
    hms = [float(v) for v in a.hms.split(",")]
    cols = [c for c in range(len(J.INPUTS)) if a.resid or c != KV2]
    assert a.src == "sm120tf", "fp8dec source: fp8blk.py loader (pending T33l dump format)"
    tr_t = [t for t in a.train_tasks.split(",") if t]; te_t = [t for t in a.test_tasks.split(",") if t]
    va_t = [t for t in a.val_tasks.split(",") if t]
    assert not (set(tr_t) & set(te_t)) and not (set(tr_t) & set(va_t)) and not (set(va_t) & set(te_t))
    t0 = time.time()
    Xt, Yt, Lt, Pt, ntok = tf_train_rows(tr_t, a.frac, cols, dev, a.resid)
    n, K = Xt.shape[0], Xt.shape[-1]
    fx_np, fd_np = TR.sets()
    fx = torch.from_numpy(fx_np).to(dev); fd = torch.from_numpy(fd_np).to(dev)
    net = NetH(K, a.d, a.nl, resid=a.resid, rkey=a.rkey).to(dev)
    if a.rkey:
        net.keys = router_keys(a.rkey).to(dev)
    print(f"{a.name}: train {tr_t} frac {a.frac} rows {n} (~{ntok} tokens) K {K} params "
          f"{sum(p.numel() for p in net.parameters())} load {time.time() - t0:.0f}s", flush=True)

    def load_eval(tasks):
        d0 = tf_eval_cache(tasks, a.eval_blocks); meta = json.load(open(f"{d0}/meta.json"))
        X, P, B = [], [], []
        for L in LAYERS:
            z = np.load(f"{d0}/L{L}.npz")
            X.append(torch.from_numpy(z["X"][..., cols])); P.append(torch.from_numpy(z["P"])); B.append(torch.from_numpy(z["bsal"]))
        return torch.stack(X), torch.stack(P), torch.stack(B), [tuple(s) for s in meta["sg"]]

    @torch.no_grad()
    def evaluate(E, with_v2=False):
        X, P, B, sg = E
        net.eval(); S = []
        for i in range(NL):
            li = torch.full((X.shape[1],), i, dtype=torch.long, device=dev); o = []
            for j in range(0, X.shape[1], 1024):
                x = X[i, j:j + 1024].to(dev)
                lp = torch.log(P[i, j:j + 1024].clamp(min=1e-30)).to(dev) if a.resid else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o.append(net(x, li[j:j + 1024], lp).float())
            S.append(torch.exp(torch.cat(o).clamp(max=30)))
        net.train()
        r = dict(arm=replay_pts(torch.stack(S), fx, fd, B.to(dev), sg, hms))
        if with_v2:
            r["v2"] = replay_pts(P.to(dev), fx, fd, B.to(dev), sg, [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5])
        return r

    EV = load_eval(va_t) if va_t else None
    if a.max_epochs:                               # small-data points: cap passes over the rows (no val pick)
        a.steps = min(a.steps, int(np.ceil(a.max_epochs * n / a.bs)))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    budget = a.budget * 60 if a.budget else None
    ts = time.time(); step = 0; best = (-1.0, None); hist = []
    vat = set(int(a.steps * (q + 1) / a.nval) for q in range(a.nval)) if EV else set()
    run = 0.0; cnt = 0
    while True:
        frac = (time.time() - ts) / budget if budget else step / a.steps
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + np.cos(np.pi * frac))
        idx = torch.randint(0, n, (a.bs,), device=dev)
        x, y, li = Xt[idx], Yt[idx].float(), Lt[idx]
        m = (~fx[li]).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            lm = net(x, li, Pt[idx] if a.resid else None).float()
        loss = TR.tweedie(lm, y, m)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
        run += float(loss.detach()); cnt += 1
        if step % 100 == 0:
            print(f"step {step} loss {run / cnt:.4f} {time.time() - ts:.0f}s", flush=True); run = 0.0; cnt = 0
        if step in vat:
            r = evaluate(EV)["arm"]; v = at(r, 2.78, 2)
            hist.append(dict(step=step, val=r)); print(f"  val@{step}: pooled@2.78 {100 * v:.2f}", flush=True)
            if v > best[0]:
                best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    os.makedirs(f"{OUT}/models", exist_ok=True)
    torch.save(dict(state=net.state_dict(), args=vars(a), K=K, cols=cols), f"{OUT}/models/{a.name}.pt")
    res = dict(args=vars(a), rows=n, tokens=ntok, steps=step, train_s=time.time() - ts, hist=hist)
    if te_t:
        del Xt, Yt
        r = evaluate(load_eval(te_t), with_v2=True)
        res["test"] = r
        for arm in ("arm", "v2"):
            print(f"TEST {a.name if arm == 'arm' else 'v2'}: " + " ".join(f"{c:.2f}/{100 * p:.2f}" for c, _, p in r[arm]) +
                  f" | @2.78 lmean {100 * at(r[arm], 2.78, 1):.2f} pooled {100 * at(r[arm], 2.78, 2):.2f}", flush=True)
    json.dump(res, open(f"{OUT}/models/{a.name}.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
