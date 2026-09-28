"""Optimise continuous parameters of a fixed-code EXL3 expert against the nonlinear teacher output.

usage: tune.py MODEL BITS GROUPS(comma or 'none') [--rank R] [--rows N] [--lr LR] [--tag T]
"""
import argparse, json, time, os, re
import torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('bits', type=int); ap.add_argument('groups')
ap.add_argument('--rank', type=int, default=0)
ap.add_argument('--rows', type=int, default=0, help='subsample training rows (0=all non-held-out)')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--patience', type=int, default=8)
ap.add_argument('--batch', type=int, default=2048)
ap.add_argument('--tag', default='')
ap.add_argument('--no-eval', action='store_true')
ap.add_argument('--track-train', action='store_true')
ap.add_argument('--no-select', action='store_true', help='keep final (overtrained) params instead of best held-out')
a = ap.parse_args()
setup()
t0 = time.time()
groups = [] if a.groups == 'none' else a.groups.split(',')
c = CFG[a.model]
Wt = teacher_weights(a.model)
parts, ref = load_exl3(a.model, a.bits)
m = Tuned(parts, groups, rank=a.rank).cuda()

# data: seeded 10% held-out split of the training sample (early stopping only)
s = torch.load(c['sample'], mmap=True, weights_only=True)
N = len(s['p'])
g = torch.Generator().manual_seed(90909)
perm = torch.randperm(N, generator=g)
nh = N // 10
hold, train = perm[:nh], perm[nh:]
if a.rows: train = train[:a.rows]
def prep(ids):
    ids = ids.sort().values
    x = s['x'][ids].cuda(); p2 = s['p'][ids].float().cuda().square()
    with torch.no_grad():
        y = torch.cat([expert(x[i:i + 2048], Wt).bfloat16() for i in range(0, len(x), 2048)])
    return x, p2, y
Xtr, Ptr, Ytr = prep(train); Xh, Ph, Yh = prep(hold)
den_tr = float((Ytr.float().square().sum(-1) * Ptr).sum()); den_h = float((Yh.float().square().sum(-1) * Ph).sum())

# parameter scaling: bias in units of teacher projection-output rms; low-rank initialised by Hessian-weighted SVD
if 'bias' in groups:
    with torch.no_grad():
        xb = Xtr[:2048].bfloat16()
        gt = F.linear(xb, Wt[0].bfloat16()).float(); ut = F.linear(xb, Wt[1].bfloat16()).float()
        h = (F.silu(gt.bfloat16()) * ut.bfloat16()).float(); dt = F.linear(h.bfloat16(), Wt[2].bfloat16()).float()
        bscale = [float(t.square().mean().sqrt()) for t in (gt, ut, dt)]
else:
    bscale = [1., 1., 1.]
if a.rank:
    with torch.no_grad():
        # input Grams from the training split itself (gate/up: x, down: teacher hidden)
        xx = Xtr.float(); w2 = Ptr
        Hx = (xx * w2[:, None]).T @ xx / w2.sum()
        hid = torch.cat([(F.silu(F.linear(Xtr[i:i+2048], Wt[0].bfloat16())) * F.linear(Xtr[i:i+2048], Wt[1].bfloat16())).float() for i in range(0, len(Xtr), 2048)])
        Hh = (hid * w2[:, None]).T @ hid / w2.sum(); del hid, xx
        for i, H in enumerate([Hx, Hx, Hh]):
            H = H + 0.01 * H.diagonal().mean() * torch.eye(len(H), device=H.device)
            ev, evec = torch.linalg.eigh(H)
            ev = ev.clamp_min(1e-12)
            Hs = (evec * ev.sqrt()) @ evec.T; Hi = (evec / ev.sqrt()) @ evec.T
            Eerr = Wt[i].T.float() - m.weight(i)  # [in,out]
            U_, S_, V_ = torch.linalg.svd(Hs @ Eerr, full_matrices=False)
            r = a.rank
            getattr(m, f'U{i}').copy_(Hi @ (U_[:, :r] * S_[:r]))
            getattr(m, f'V{i}').copy_(V_[:r])
        del Hx, Hh

def fwd(x):
    Ws = [m.weight(i).T.bfloat16() for i in range(3)]
    bs = [None if b is None else (b * sc).bfloat16() for b, sc in zip(m.biases(), bscale)]
    return expert(x, Ws, bs)

@torch.no_grad()
def loss_full(X, P, Y, den):
    Ws = [m.weight(i).T.bfloat16() for i in range(3)]
    bs = [None if b is None else (b * sc).bfloat16() for b, sc in zip(m.biases(), bscale)]
    num = 0.
    for i in range(0, len(X), 4096):
        num += float(((expert(X[i:i+4096], Ws, bs) - Y[i:i+4096].float()).square().sum(-1) * P[i:i+4096]).sum())
    return 100 * (num / den) ** .5

params = [p for p in m.parameters()]
hist = []
init = dict(train=loss_full(Xtr, Ptr, Ytr, den_tr), hold=loss_full(Xh, Ph, Yh, den_h))
best = dict(epoch=0, **init)
best_state = {n: p.detach().clone() for n, p in m.named_parameters()}
if params:
    opt = torch.optim.Adam(params, lr=a.lr)
    steps_per_epoch = max(1, len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * steps_per_epoch)
    gen = torch.Generator(device='cuda').manual_seed(7)
    bad = 0
    for ep in range(1, a.epochs + 1):
        order = torch.randperm(len(Xtr), device='cuda', generator=gen)
        for st in range(steps_per_epoch):
            ids = order[st * a.batch:(st + 1) * a.batch]
            y = fwd(Xtr[ids])
            loss = ((y - Ytr[ids].float()).square().sum(-1) * Ptr[ids]).sum() / (Ytr[ids].float().square().sum(-1) * Ptr[ids]).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        h = loss_full(Xh, Ph, Yh, den_h)
        hist.append(dict(epoch=ep, hold=h, train=loss_full(Xtr, Ptr, Ytr, den_tr) if a.track_train else None))
        if h < best['hold'] - 1e-4:
            best = dict(epoch=ep, hold=h); best_state = {n: p.detach().clone() for n, p in m.named_parameters()}; bad = 0
        else:
            bad += 1
            if bad >= a.patience: break
    if a.no_select:
        best = dict(epoch=hist[-1]['epoch'], hold=hist[-1]['hold'], selection='final (no early stopping)')
    else:
        with torch.no_grad():
            for n, p in m.named_parameters(): p.copy_(best_state[n])
    # storage rounding: fp16 for everything, 8-bit uniform code for tile log-scales
    with torch.no_grad():
        for n, p in m.named_parameters():
            if re.fullmatch(r't\d', n):
                lo, hi = float(p.min()), float(p.max())
                if hi > lo:
                    q = ((p - lo) / (hi - lo) * 255).round() / 255 * (hi - lo) + lo; p.copy_(q)
            elif re.fullmatch(r'[ab]\d', n):
                # su/sv are stored fp16 in the native format: round the product
                pass
            else:
                p.copy_(p.half().float())
        for i in range(3):
            if 'suv' in groups:
                su = (m.__getattr__(f'su{i}') * getattr(m, f'a{i}').exp()).half().float()
                sv = (m.__getattr__(f'sv{i}') * getattr(m, f'b{i}').exp()).half().float()
                getattr(m, f'su{i}').copy_(su); getattr(m, f'sv{i}').copy_(sv)
                getattr(m, f'a{i}').zero_(); getattr(m, f'b{i}').zero_()
    best['train'] = loss_full(Xtr, Ptr, Ytr, den_tr); best['hold_rounded'] = loss_full(Xh, Ph, Yh, den_h)

out = dict(model=a.model, bits=a.bits, groups=groups, rank=a.rank, rows=len(Xtr), held_out_rows=len(Xh), lr=a.lr,
           extra_bpw=m.extra_bits(), init=init, best=best, history=hist, seconds=None)
if not a.no_eval:
    Ws, bs = m.dense()
    bs = [None if b is None else b * sc for b, sc in zip(bs, bscale)]
    ev = eval_captures(a.model, Wt, {'base': (ref, [None] * 3), 'tuned': (Ws, bs)})
    out['eval'] = ev
out['seconds'] = time.time() - t0
out['peak_cuda_mib'] = torch.cuda.max_memory_allocated() / 2**20
name = f"{a.model}{a.bits}_{a.groups.replace(',', '+')}" + (f"_r{a.rank}" if a.rank else '') + (f"_n{a.rows}" if a.rows else '') + (f"_{a.tag}" if a.tag else '')
os.makedirs('results', exist_ok=True)
json.dump(out, open(f'results/{name}.json', 'w'), indent=1)
summ = {k: (round(v['base'], 3), round(v['tuned'], 3)) for k, v in out.get('eval', {}).items()}
print(name, 'bpw+%.4f' % out['extra_bpw'], 'init', {k: round(v, 3) for k, v in init.items()}, 'best', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()}, 'secs %.0f' % out['seconds'])
print(' eval', summ, flush=True)
