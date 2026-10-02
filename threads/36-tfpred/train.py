"""train the nq-tfpred predictor.
  train.py --data ids|cap --content 0|1 --sal 0|1 --out ckpt.pt [--steps 20000 --bs 8 --d 64 --nblk 2 --dev cuda:0]
ids: expert-predict decode streams (train = TASKS_IDS minus test/val tasks); cap: captured traces (ds/cap-<task>.npz)."""
import argparse, json, math, os, sys, time, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import data as D, model as M

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='ids'); ap.add_argument('--content', type=int, default=0); ap.add_argument('--sal', type=int, default=0)
ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--d', type=int, default=64); ap.add_argument('--nblk', type=int, default=2); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--dev', default='cuda:0'); ap.add_argument('--max_train_blocks', type=int, default=0); ap.add_argument('--eval_every', type=int, default=1000)
ap.add_argument('--tasks', default=''); ap.add_argument('--val', default=''); ap.add_argument('--nval', type=int, default=256)
ap.add_argument('--data_dev', default='cpu'); ap.add_argument('--mem_gb', type=float, default=0); ap.add_argument('--noans', type=int, default=0)
ap.add_argument('--init', default=''); ap.add_argument('--frac', type=float, default=1.0); ap.add_argument('--exclude', default='')
ap.add_argument('--mmap', type=int, default=0)   # page-cache-backed big arrays (ids pretrain; not with --sal)
a = ap.parse_args()
dev = torch.device(a.dev); torch.manual_seed(0)
if a.mem_gb and dev.type == 'cuda':      # hard cap: this process OOMs itself before it can squeeze a co-resident serve
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory, dev)


def streams(kind, tasks):
    out = []
    for t in tasks:
        f = f'/rawdata/Jarrel/nq-tfpred/ds/{kind}-{t}.npz'
        if os.path.exists(f):
            out.append((t, D.ids_blocks(t, kind)))
    return out


if a.data == 'ids':
    all_t = D.TASKS_IDS
else:
    all_t = sorted(os.path.basename(f)[4:-4] for f in glob.glob('/rawdata/Jarrel/nq-tfpred/ds/cap-*.npz'))
ex_t = a.exclude.split(',') if a.exclude else []
tr_t = [t for t in all_t if t not in D.TEST_TASKS + D.VAL_TASKS + ex_t] if not a.tasks else a.tasks.split(',')
va_t = [t for t in all_t if t in D.VAL_TASKS] if not a.val else a.val.split(',')
print('train', tr_t, 'val', va_t, flush=True)
tr = streams(a.data, tr_t); va = streams(a.data, va_t)
if a.frac < 1:                           # learning curve: the first frac of each training stream
    tr = [(t, {k: (v[:max(64, int(len(d['cnt']) * a.frac))] if k not in ('pf', 'pfn') else v) for k, v in d.items()}) for t, d in tr]
if a.max_train_blocks:
    tr = [(t, {k: (v[:a.max_train_blocks] if k not in ('pf', 'pfn') else v) for k, v in d.items()}) for t, d in tr]
scale = None
if a.sal:
    s = np.concatenate([d['sal'].astype(np.float32).sum((0, 2)) / np.maximum(d['cnt'].astype(np.float32).sum((0, 2)), 1)
                        for _, d in tr]).reshape(len(tr), -1).mean(0)
    scale = s
ddev = torch.device(a.data_dev)
mmd = (lambda ts, tag: f'/rawdata/Jarrel/nq-tfpred/mm/{a.data}-{tag}-' + '+'.join(ts) + f'-f{a.frac}-m{a.max_train_blocks}') if (a.mmap and not a.sal and ddev.type == 'cpu') else (lambda ts, tag: None)
Btr = D.Blocks([d for _, d in tr], ddev, use_sal=bool(a.sal), cdev=dev, mmap_dir=mmd([t for t, _ in tr], 'tr'))
Bva = D.Blocks([d for _, d in va], ddev, use_sal=bool(a.sal), cdev=dev, mmap_dir=mmd([t for t, _ in va], 'va'))
if a.sal:
    for B_ in (Btr, Bva):
        B_.sal = (B_.sal.float() / torch.as_tensor(scale, device=ddev)[None, :, None]).half(); B_.ch = (B_.ch.float() / torch.as_tensor(scale, device=ddev)[None, :, None]).half()
print('blocks train', Btr.nb, 'val', Bva.nb, flush=True)
net = M.TFPred(d=a.d, nblk=a.nblk, content=bool(a.content), scale=None).to(dev)
if a.init:                               # e.g. content fine-tune from the ids-only pretrain (new content params start fresh)
    ck0 = torch.load(a.init, map_location='cpu', weights_only=False)
    miss = net.load_state_dict({k: v for k, v in ck0['state'].items() if k != 'scale'}, strict=False)
    print('init from', a.init, 'missing', miss.missing_keys, flush=True)
npar = sum(p.numel() for p in net.parameters()); print('params', npar, flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))
wlen = torch.tensor(np.diff(D.WIN), dtype=torch.float32, device=dev)
gen = torch.Generator().manual_seed(0)
vgen = torch.Generator().manual_seed(1); vidx = torch.randint(0, Bva.nb, (a.nval,), generator=vgen)


def fwd(B_, b):
    x = B_.inputs(b)
    if not a.content:
        x.pop('tok', None); x.pop('tokm', None); x.pop('hp', None)
    if a.noans:
        x['ans'] = torch.zeros_like(x['ans'])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        return net(**x)


@torch.no_grad()
def evaluate():
    net.eval(); L = 0; n = 0; rec = np.zeros(len(D.WIN) - 1); base = np.zeros(len(D.WIN) - 1)
    for i in range(0, len(vidx), a.bs):
        b = vidx[i:i + a.bs]; lr = fwd(Bva, b); y, m = Bva.targets(b)
        L += M.tweedie(lr, y, m, wlen).item(); n += 1
        # recall@77 of the predicted top-77 per layer vs the window's true top-77 (hit-weighted), per window
        for w in range(lr.shape[-1]):
            top = lr[..., w].float().topk(77, -1).indices
            yy = y[..., w]; tot = yy.sum(-1).clamp(min=1e-9)
            rec[w] += (yy.gather(-1, top).sum(-1) / tot).mean().item()
            base[w] += (yy.topk(77, -1).values.sum(-1) / tot).mean().item()
    net.train(); return L / n, rec / n, base / n


t0 = time.time(); best = 1e9; hist = []
for it in range(a.steps + 1):
    if it % a.eval_every == 0:
        vl, rec, orc = evaluate()
        hist.append(dict(it=it, val=vl, rec77=rec.round(4).tolist(), oracle77=orc.round(4).tolist(), secs=round(time.time() - t0)))
        print(json.dumps(hist[-1]), flush=True)
        if vl < best:
            best = vl
            torch.save(dict(state=net.state_dict(), cfg=net.cfg, args=vars(a), scale=scale, hist=hist, params=npar,
                            win=D.WIN, train=tr_t, val=va_t), a.out)
    if it == a.steps: break
    b = Btr.sample_index(a.bs, gen)
    lr = fwd(Btr, b); y, m = Btr.targets(b)
    loss = M.tweedie(lr, y, m, wlen)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if it % 100 == 0:
        print(it, round(loss.item(), 5), '%.0fs' % (time.time() - t0), flush=True)
print('done best', best)
