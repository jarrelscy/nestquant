"""T33a seq: per-(layer,expert) causal TCN over raw per-block history (cnt, salience[, v2 score]), shared across
layers/experts with layer+expert embeddings and a per-block cross-expert mean context; targets next-64/128/256
normalised salience; loss Tweedie(1.5) [+ ListNet over non-fixed experts].  Causal: block t's output uses blocks <= t.
  train.py NAME [--v2 0/1] [--resid 0/1] [--H 48] [--dil 7] [--lw 0.0] [--ep 6] [--lr 2e-3] [--gru 0]"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lite

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--v2", type=int, default=0); ap.add_argument("--resid", type=int, default=0)
ap.add_argument("--H", type=int, default=48); ap.add_argument("--dil", type=int, default=7)
ap.add_argument("--lw", type=float, default=0.0); ap.add_argument("--ep", type=int, default=6)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--aux", type=float, default=0.3); ap.add_argument("--ctx", type=int, default=1)
ap.add_argument("--tau", type=float, default=1.0); ap.add_argument("--full", type=int, default=0)
ap.add_argument("--noval", type=int, default=0); ap.add_argument("--maxb", type=int, default=0); ap.add_argument("--es", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NT", "20"))); torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("dev", dev, flush=True)
D = f"{lite.WD}/data"; OUTD = f"{lite.WD}/runs/{a.name}"; os.makedirs(OUTD, exist_ok=True)
NL = 75
FX = torch.tensor(np.stack([lite.fxmask(L) for L in lite.LAYERS]), device=dev)       # [75,256]
VALCH = [3, 8, 13, 18, 23, 28]
HOR = (4, 8, 16)


def load(corpus):
    cnt = torch.from_numpy(np.load(f"{D}/{corpus}.cnt.npy")).to(dev)
    sal = torch.from_numpy(np.load(f"{D}/{corpus}.saln.npy")).to(dev)
    v2 = torch.from_numpy(np.load(f"{D}/{corpus}.v2.npy")).to(dev) if a.v2 or a.resid else None
    return cnt, sal, v2


def feats(cnt, sal, v2, pos):
    """[..., T, 256] -> [..., T, 256, C]"""
    fl = [torch.log1p(cnt.float()), torch.log1p(sal.float().clamp_min(0))]
    if a.v2:
        fl.append(torch.log1p(v2.float().clamp_min(0)))
    fl.append(pos[..., None].expand_as(fl[0]))
    return torch.stack(fl, -1)


def targets(sal, T):
    """sal [B,T,256] -> y [B,T,256,3], m [B,T,3]  (future sums of blocks t+1..t+h inside the window)"""
    cs = torch.cat([torch.zeros_like(sal[:, :1]), sal.float().cumsum(1)], 1)
    ys, ms = [], []
    for h in HOR:
        idx = torch.arange(T, device=sal.device)
        hi = (idx + 1 + h).clamp(max=T)
        ys.append(cs[:, hi] - cs[:, idx + 1]); ms.append((idx + h < T).float())
    return torch.stack(ys, -1), torch.stack(ms, -1)[None].expand(sal.shape[0], -1, -1)


class TCN(nn.Module):
    def __init__(s, cin, H, ndil):
        super().__init__()
        s.lemb = nn.Embedding(NL, 8); s.eemb = nn.Embedding(NL * 256, 12)
        nn.init.normal_(s.eemb.weight, std=0.1)
        s.inp = nn.Linear(cin + 1, H)
        s.dil = [2 ** i for i in range(ndil)]
        s.conv = nn.ModuleList([nn.Linear(2 * H, 2 * H) for d in s.dil])
        s.proj = nn.ModuleList([nn.Linear(H, H) for _ in s.dil])
        s.ctx = nn.Linear(H, H) if a.ctx else None
        s.emb2h = nn.Linear(20, H)
        s.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, 3))
        s.rs = nn.Parameter(torch.tensor(1.0))
        s.RF = sum(s.dil) + 1

    def forward(s, x, Lidx, fx, lv2=None, eix=None):
        """x [B,T,256,C]; Lidx [B]; fx [B,256] -> log-rate [B,T,256,3]   (channels-last GEMMs, causal shifts on T)"""
        B, T, E, C = x.shape
        z = torch.cat([x, fx.float()[:, None, :, None].expand(B, T, E, 1)], -1)
        h = s.inp(z)                                                     # [B,T,E,H]
        for cv, pj, d in zip(s.conv, s.proj, s.dil):
            hp = F.pad(h, (0, 0, 0, 0, d, 0))[:, :T]                     # h[t-d] (zeros before chain start)
            u = cv(torch.cat([hp, h], -1))
            u = torch.tanh(u[..., :u.shape[-1] // 2]) * torch.sigmoid(u[..., u.shape[-1] // 2:])
            h = h + pj(u)
        if eix is None:
            eix = torch.arange(256, device=x.device)[None].expand(B, -1)
        eid = Lidx[:, None] * 256 + eix
        emb = torch.cat([s.lemb(Lidx)[:, None].expand(-1, E, -1), s.eemb(eid)], -1)   # [B,E,20]
        h = h + s.emb2h(emb)[:, None]
        if s.ctx is not None:
            w = (~fx).float()[:, None, :, None]
            m = (h * w).sum(2, keepdim=True) / w.sum(2, keepdim=True)
            h = h + s.ctx(F.gelu(m))
        o = s.head(F.gelu(h))
        if lv2 is not None:
            o = o + s.rs * lv2[..., None]
        return o


def tweedie(o, y, p=1.5):
    return -y * torch.exp(o * (1 - p)) / (1 - p) + torch.exp(o * (2 - p)) / (2 - p)


def batch_loss(model, cnt, sal, v2, Lidx, t0, T, pos, eix=None):
    fx = FX[Lidx]
    if eix is not None:
        g = lambda M: None if M is None else torch.gather(M, 2, eix[:, None].expand(-1, M.shape[1], -1))
        cnt, sal, v2 = g(cnt), g(sal), g(v2)
        fx = torch.gather(fx, 1, eix)
    x = feats(cnt, sal, v2, pos)
    lv2 = torch.log(v2.float().clamp_min(0) + 1e-2) if a.resid else None
    o = model(x, Lidx, fx, lv2, eix)
    y, m = targets(sal, T)
    wmask = (~fx).float()[:, None, :, None] * m[:, :, None, :]        # [B,T,E,3]
    ym = (y * wmask).sum((0, 1, 2)) / wmask.sum((0, 1, 2))
    tw = tweedie(o - torch.log(ym + 1e-6), y / (ym + 1e-6))
    hw = torch.tensor([1.0, a.aux, a.aux], device=o.device)
    lt = ((tw * wmask).sum((0, 1, 2)) / wmask.sum((0, 1, 2)) * hw).sum()
    ll = torch.zeros((), device=o.device)
    if a.lw > 0:
        o0 = o[..., 0].masked_fill(fx[:, None], -1e9) / a.tau
        y0 = y[..., 0] * (~fx)[:, None]
        pt = y0 / y0.sum(-1, keepdim=True).clamp_min(1e-9)
        ls = -(pt * F.log_softmax(o0, -1)).sum(-1)
        mm = m[..., 0] * (y0.sum(-1) > 0)
        ll = (ls * mm).sum() / mm.sum()
    return lt + a.lw * ll, lt.item(), ll.item()


def chain_pos(T, device):
    return (torch.arange(T, device=device).float().clamp(max=128) / 128)


@torch.no_grad()
def infer(model, corpus, cnt, sal, v2):
    """-> scores [75, nb, 256] float32 numpy (exp of next-64 head), chains from lite.segs_of (full causal)."""
    nb = cnt.shape[1]
    sg = lite.segs_of(corpus, nb)
    out = np.zeros((NL, nb, 256), np.float32)
    model.eval()
    CH = 4096                                           # time chunk with RF lookback
    for li in range(NL):
        Lidx = torch.tensor([li], device=dev)
        for s, e in sg:
            for c0 in range(s, e, CH):
                c1 = min(c0 + CH, e); b0 = max(s, c0 - model.RF)
                pos = chain_pos(c1 - s, dev)[b0 - s:]
                x = feats(cnt[li:li + 1, b0:c1], sal[li:li + 1, b0:c1], None if v2 is None else v2[li:li + 1, b0:c1], pos)
                lv2 = torch.log(v2[li:li + 1, b0:c1].float().clamp_min(0) + 1e-2) if a.resid else None
                o = model(x, Lidx, FX[Lidx], lv2)[0, c0 - b0:, :, 0]
                out[li, c0:c1] = torch.exp(o).float().cpu().numpy()
    model.train()
    return out


def evaluate(S, corpus, cnt, sal, hms=(0.3, 0.5, 0.7, 1.0), sel=None):
    from multiprocessing import Pool
    bc = cnt.cpu().numpy(); bs = sal.float().cpu().numpy()
    jobs = []
    for li, L in enumerate(lite.LAYERS):
        sg = lite.segs_of(corpus, bc.shape[1]) if sel is None else sel
        for hm in hms:
            jobs.append((S[li], L, bc[li], bs[li], sg, hm))
    with Pool(16) as p:
        r = p.starmap(lite.metrics, jobs)
    res = {}
    for j, hm in enumerate(hms):
        rr = r[j::len(hms)]
        res[hm] = dict(sal=100 * np.mean([x["sal"] for x in rr]), churn=np.mean([x["churn"] for x in rr]))
    return res


if __name__ == "__main__":
    t00 = time.time()
    cnt, sal, v2 = load("calib-fit")
    nch = cnt.shape[1] // 512
    trch = [c for c in range(nch) if a.full or c not in VALCH]
    cin = 3 + a.v2
    model = TCN(cin, a.H, a.dil).to(dev)
    print("params", sum(p.numel() for p in model.parameters()), "RF", model.RF, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    items = [(li, c) for li in range(NL) for c in trch]
    nsteps = a.ep * (len(items) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=nsteps, pct_start=0.05)
    pos = chain_pos(512, dev)
    step = 0
    for ep in range(a.ep):
        perm = np.random.permutation(len(items))
        tl = []
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            if a.maxb and i // a.bs >= a.maxb:
                break
            it = [items[j] for j in perm[i:i + a.bs]]
            Li = torch.tensor([x[0] for x in it], device=dev)
            sl = [slice(c * 512, c * 512 + 512) for _, c in it]
            cb = torch.stack([cnt[l, s] for (l, _), s in zip(it, sl)])
            sb = torch.stack([sal[l, s] for (l, _), s in zip(it, sl)])
            vb = torch.stack([v2[l, s] for (l, _), s in zip(it, sl)]) if v2 is not None else None
            eix = None
            if a.es:                                       # expert subsample: half from top-128 recent activity, half uniform (non-fixed)
                act = sb.float().sum(1).masked_fill(FX[Li], -1)
                r = torch.rand_like(act) + (act > 0).float() + (act >= act.topk(128, 1).values[:, -1:]).float() * 2
                r = r.masked_fill(FX[Li], -9)
                eix = r.topk(a.es, 1).indices
                if torch.rand(()) < 0.5:
                    eix = torch.rand_like(act).masked_fill(FX[Li], -9).topk(a.es, 1).indices
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss, lt, ll = batch_loss(model, cb, sb, vb, Li, 0, 512, pos, eix)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tl.append((lt, ll))
        # val loss
        with torch.no_grad():
            vl = []
            for li in range(NL):
                for c in (VALCH if not a.maxb else VALCH[:1]):
                    s = slice(c * 512, c * 512 + 512)
                    _, lt, ll = batch_loss(model, cnt[li, s][None], sal[li, s][None], None if v2 is None else v2[li, s][None],
                                           torch.tensor([li], device=dev), 0, 512, pos)
                    vl.append((lt, ll))
        print(f"ep {ep} train {np.mean(tl, 0).round(4)} val {np.mean(vl, 0).round(4)} {time.time() - t00:.0f}s", flush=True)
    torch.save(dict(sd=model.state_dict(), args=vars(a)), f"{OUTD}/model.pt")
    res = {}
    # calib-val
    idx = np.concatenate([np.arange(c * 512, c * 512 + 512) for c in VALCH])
    sgv = [(i * 512, i * 512 + 512) for i in range(len(VALCH))]
    S = infer(model, "calib-val", cnt[:, idx], sal[:, idx], None if v2 is None else v2[:, idx])
    res["calib-val"] = evaluate(S, "calib-val", cnt[:, idx], sal[:, idx], sel=sgv)
    Sv2 = np.load(f"{D}/calib-fit.v2.npy")[:, idx]
    res["calib-val-v2"] = evaluate(Sv2, "calib-val", cnt[:, idx], sal[:, idx], sel=sgv)
    del cnt, sal, v2, S
    for corpus in ["glm52-heldout"] + ([] if a.noval else ["sm120tf"]):
        c_, s_, v_ = load(corpus)
        S = infer(model, corpus, c_, s_, v_)
        np.save(f"{OUTD}/S_{corpus}.npy", S.astype(np.float16))
        res[corpus] = evaluate(S, corpus, c_, s_)
        del c_, s_, v_, S
        print(corpus, res[corpus], flush=True)
    json.dump(res, open(f"{OUTD}/res.json", "w"), indent=1, default=float)
    for k, v in res.items():
        print(k, " ".join(f"hm{hm}: {r['sal']:.2f}/{r['churn']:.2f}" for hm, r in v.items()), flush=True)
    print(f"done {time.time() - t00:.0f}s", flush=True)
