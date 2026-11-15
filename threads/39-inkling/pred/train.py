"""Inkling floating hot-set predictor (T33 jF-style, trained end to end on GPU).

Per (layer, 16-token block b, expert e) features at the END of block b, causal within a chain (chain = document =
request; state resets at chain start):
  hit EMAs (cnt/16) and salience EMAs (sal/m_L) at half-lives 1, 4, 16, 64 blocks; this block's hits and salience;
  log1p(blocks since last hit); log1p(block position); static prior log(384 * routed share on the train split).
Score net: Linear(K->d) + layer embedding -> 2-layer transformer over the 384 experts of the layer (no expert-id
embedding; identity enters only through the static prior) -> scalar log-score s.
Loss: Poisson deviance of exp(s) vs y = sal[b+1] + 0.5 * mean(sal[b+1..b+4]) (normalised by m_L).
Serve semantics (sync, lag 0): the set picked at the end of block b serves block b+1; hysteresis: resident scores
x (1 + hm); the chain starts from the static top-H set.

  python train.py fit   --out /tmp/nq_inkling/pred/net.pt
  python train.py eval  --net /tmp/nq_inkling/pred/net.pt [--H 64 --hm 0.5]
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

BLK = "/tmp/nq_inkling/pred/blk"
NL, NE = 48, 384
HL = (1, 4, 16, 64)
K = 2 * len(HL) + 5


def load_layer(L, dev):
    z = np.load(f"{BLK}/L{L}.npz")
    cnt = torch.from_numpy(z["cnt"]).to(dev).float()
    sal = torch.from_numpy(z["sal"]).to(dev).float()
    chain = torch.from_numpy(z["chain"]).to(dev).long()
    pos = torch.from_numpy(z["pos"].astype(np.int64)).to(dev)
    split = torch.from_numpy(z["split"]).to(dev).long()
    tr = split == 0
    mL = sal[tr].sum(1).mean() / NE                                     # mean per-expert block salience (train)
    nr = cnt[tr].sum(0)
    prior = torch.log(nr / nr.sum() * NE + 1e-3)
    return dict(cnt=cnt, sal=sal / mL, chain=chain, pos=pos, split=split, prior=prior, mL=float(mL))


def features(d):
    """[nb, 384, K] fp16 features at the end of each block + targets y [nb, 384] (nan where no next block)."""
    cnt, sal, pos = d["cnt"] / 16.0, d["sal"], d["pos"]
    nb = cnt.shape[0]; dev = cnt.device
    em = [torch.zeros_like(cnt) for _ in range(2 * len(HL))]
    since = torch.zeros_like(cnt)
    a = [0.5 ** (1.0 / h) for h in HL]
    maxp = int(pos.max()) + 1
    for p in range(maxp):
        ix = torch.nonzero(pos == p)[:, 0]
        for i, h in enumerate(HL):
            prev_c = em[i][ix - 1] if p else 0.0
            prev_s = em[len(HL) + i][ix - 1] if p else 0.0
            em[i][ix] = a[i] * prev_c + (1 - a[i]) * cnt[ix]
            em[len(HL) + i][ix] = a[i] * prev_s + (1 - a[i]) * sal[ix]
        prev = since[ix - 1] if p else torch.full_like(cnt[ix], 64.0)
        since[ix] = torch.where(cnt[ix] > 0, torch.zeros_like(prev), (prev + 1).clamp(max=128))
    X = torch.stack([torch.log1p(e * 8) for e in em] + [cnt, torch.log1p(sal), torch.log1p(since),
                     torch.log1p(pos.float())[:, None].expand(nb, NE), d["prior"][None].expand(nb, NE)], -1)
    # targets: next block and mean of next 4 (same chain only)
    same = torch.zeros(nb, dtype=torch.bool, device=dev); same[:-1] = d["chain"][1:] == d["chain"][:-1]
    y = torch.full_like(sal, float("nan"))
    y[:-1][same[:-1]] = sal[1:][same[:-1]]
    acc = torch.zeros_like(sal); n = torch.zeros(nb, 1, device=dev)
    for k in range(1, 5):
        ok = torch.zeros(nb, dtype=torch.bool, device=dev)
        ok[:-k] = d["chain"][k:] == d["chain"][:-k]
        acc[:-k] += torch.where(ok[:-k, None], sal[k:], torch.zeros_like(sal[k:])); n[:-k] += ok[:-k, None].float()
    y4 = acc / n.clamp(min=1)
    return X.half(), y, y4


class Net(nn.Module):
    def __init__(self, d=96, nl=2, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(K, d), nn.GELU(), nn.Linear(d, d))
        self.lemb = nn.Embedding(NL, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, nl)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, X, L):                     # X [B, 384, K], L [B]
        h = self.inp(X.float()) + self.lemb(L)[:, None]
        return self.out(self.tf(h))[..., 0]


def replay(S, sal, chain, pos, prior, H=64, hm=0.5, return_sets=False):
    """Coverage of block b+1 salience by the set chosen at the end of block b (sync), hysteresis hm.
    S [nb,384] scores at the end of each block. Returns (covered, total, swaps, n_refresh)."""
    nb = S.shape[0]; dev = S.device
    static = torch.zeros(NE, dtype=torch.bool, device=dev); static[torch.topk(prior, H).indices] = True
    R = torch.zeros(nb, NE, dtype=torch.bool, device=dev)              # resident set serving block b
    R[pos == 0] = static
    cov = tot = 0.0; swaps = 0; nref = 0
    maxp = int(pos.max()) + 1
    for p in range(maxp - 1):
        ix = torch.nonzero(pos == p)[:, 0]
        nxt = ix + 1
        ok = nxt < nb
        ix, nxt = ix[ok], nxt[ok]
        ok = chain[nxt] == chain[ix]
        ix, nxt = ix[ok], nxt[ok]
        if not len(ix):
            continue
        sc = S[ix] * torch.where(R[ix], 1.0 + hm, 1.0)
        newR = torch.zeros_like(R[ix]); newR.scatter_(1, torch.topk(sc, H, dim=1).indices, True)
        swaps += int((newR & ~R[ix]).sum()); nref += len(ix)
        R[nxt] = newR
    if return_sets:
        return R
    cov = float((sal * R).sum()); tot = float(sal.sum())
    return cov, tot, swaps, nref


def layer_stats():
    """{L: (mL, prior[384])} from the train split, cached (needed to build features at eval/serve time)."""
    f = "/tmp/nq_inkling/pred/layer_stats.pt"
    if not os.path.exists(f):
        st = {}
        for L in range(NL):
            d = load_layer(L, "cpu"); st[L] = (d["mL"], d["prior"].float())
        torch.save(st, f)
    return torch.load(f)


def sets_for_blocks(net, L, cnt, sal_raw, chain, pos, H=64, hm=0.5, stats=None):
    """Serve-time selection for one layer: block matrices (raw salience) -> resident set serving each block [nb,384]."""
    stats = stats or layer_stats()
    mL, prior = stats[L]
    dev = cnt.device
    d = dict(cnt=cnt.float(), sal=sal_raw.float() / mL, chain=chain, pos=pos, prior=prior.to(dev))
    X, _, _ = features(d)
    S = score_layer(net, X, L, dev)
    return replay(S, d["sal"], chain, pos, d["prior"], H, hm, return_sets=True)


def score_layer(net, X, L, dev, bs=2048):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, X.shape[0], bs):
            x = X[i:i + bs]
            out.append(net(x, torch.full((x.shape[0],), L, device=dev, dtype=torch.long)).float().exp())
    return torch.cat(out)


def evaluate(net, layers, dev, H, hm, splits=(1, 2), baselines=True):
    res = {}
    for L in layers:
        d = load_layer(L, dev)
        X, y, _ = features(d)
        heur = torch.expm1(X[..., len(HL) + 1].float()) / 8 + 1e-4 * torch.exp(d["prior"])[None]   # sal EMA h=4
        S_net = score_layer(net, X, L, dev) if net is not None else None
        for sp in splits:
            m = d["split"] == sp
            idx = torch.nonzero(m)[:, 0]
            sub = lambda t: t[idx]
            r = {}
            for name, S in (("net", S_net), ("ema4", heur), ("static", d["prior"][None].expand_as(heur))):
                if S is None or (name != "net" and not baselines):
                    continue
                cv, tt, sw, nr = replay(sub(S), sub(d["sal"]), sub(d["chain"]), sub(d["pos"]), d["prior"], H, hm)
                r[name] = dict(cov=cv / tt, swaps_per_refresh=sw / max(nr, 1))
            res.setdefault(sp, {})[L] = r
        del X, y, d; torch.cuda.empty_cache()
    return res


def fit(a):
    dev = "cuda"
    torch.manual_seed(0)
    data = []
    t0 = time.time()
    for L in range(NL):
        d = load_layer(L, dev)
        X, y, y4 = features(d)
        keep = (d["split"] == 0) & ~torch.isnan(y[:, 0])
        idx = torch.nonzero(keep)[:, 0][::a.sub]
        tgt = (y[idx] + 0.5 * y4[idx]).half()
        data.append((X[idx].contiguous(), tgt.contiguous()))
        vk = (d["split"] == 1) & ~torch.isnan(y[:, 0])
        vidx = torch.nonzero(vk)[:, 0][::4]
        data[-1] = data[-1] + (X[vidx].contiguous(), (y[vidx] + 0.5 * y4[vidx]).half().contiguous())
        del X, y, y4, d; torch.cuda.empty_cache()
        print(f"L{L} train {len(idx)} val {len(vidx)} ({time.time() - t0:.0f}s)", flush=True)
    net = Net(a.d, a.nl).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    ntr = [x[0].shape[0] for x in data]
    total = a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)

    def loss_fn(s, t):
        t = t.float()
        return (torch.exp(s) - t * s).mean()

    def val():
        net.eval(); tot = n = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for L, x in enumerate(data):
                Xv, Tv = x[2], x[3]
                for i in range(0, Xv.shape[0], 4096):
                    s = net(Xv[i:i + 4096], torch.full((min(4096, Xv.shape[0] - i),), L, device=dev, dtype=torch.long)).float()
                    tot += float(loss_fn(s, Tv[i:i + 4096]) * s.shape[0]); n += s.shape[0]
        net.train(); return tot / n

    best = 1e9
    for step in range(total):
        Ls = torch.randint(0, NL, (a.bs,))
        X = torch.empty(a.bs, NE, K, dtype=torch.float16, device=dev); T = torch.empty(a.bs, NE, dtype=torch.float16, device=dev)
        for L in Ls.unique().tolist():
            j = torch.nonzero(Ls == L)[:, 0]
            r = torch.randint(0, ntr[L], (len(j),), device=dev)
            X[j.to(dev)] = data[L][0][r]; T[j.to(dev)] = data[L][1][r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = net(X, Ls.to(dev)).float()
        loss = loss_fn(s, T)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == total - 1:
            v = val()
            print(f"step {step} train {float(loss):.5f} val {v:.5f} ({time.time() - t0:.0f}s)", flush=True)
            if v < best:
                best = v
                torch.save(dict(state=net.state_dict(), args=vars(a), K=K, HL=HL, step=step, val=v), a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--out", default="/tmp/nq_inkling/pred/net.pt")
    ap.add_argument("--net", default="")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--hm", type=float, default=0.5)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.cmd == "fit":
        return fit(a)
    net = None
    if a.net:
        ck = torch.load(a.net, map_location="cuda")
        net = Net(ck["args"]["d"], ck["args"]["nl"]).cuda().eval(); net.load_state_dict(ck["state"])
    lo, hi = map(int, a.layers.split("-"))
    res = evaluate(net, range(lo, hi + 1), "cuda", a.H, a.hm)
    for sp, rl in res.items():
        names = sorted({n for r in rl.values() for n in r})
        line = {n: (np.mean([r[n]["cov"] for r in rl.values()]), np.mean([r[n]["swaps_per_refresh"] for r in rl.values()])) for n in names}
        print(f"split {sp} H{a.H} hm{a.hm}: " + "  ".join(f"{n} cov {c:.4f} swaps {s:.2f}" for n, (c, s) in line.items()), flush=True)
    if a.json:
        json.dump({str(k): {str(L): v for L, v in rl.items()} for k, rl in res.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
