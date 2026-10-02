"""parity: TFGPUPredictor (serve class, fed block-by-block through step()) vs the trainer's Blocks.inputs + TFPred forward
on the same decode stream.  Prefill counts are fed as one big step before each request's first decode step.
  test_pred_parity.py <ckpt> [task] [nblocks] [dev]"""
import sys, os, numpy as np, torch
sys.path.insert(0, '/data/Jarrel/nq-tfpred/src'); sys.path.insert(0, '/data/Jarrel/nq-tfpred/nq-src/sm120/serve')
os.environ['NQ_HOME'] = '/data/Jarrel/nq-tfpred/nq-src'
import data as D, model as M
sys.modules['model'] = M
import nq_tfpred_gpu as P

ck, task = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'embedding-drift-monitor')
nbk = int(sys.argv[3]) if len(sys.argv) > 3 else 300; dev = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
z = dict(np.load(f'/rawdata/Jarrel/nq-tfpred/ds/ids-{task}.npz'))
n = nbk * 16; zz = {k: (v[:n] if k in ('ex', 'tok', 'think', 'rq') else v) for k, v in z.items()}
d = D.blocks_from_stream(zz)
B = D.Blocks([d], torch.device('cpu'), cdev=torch.device(dev))
c = torch.load(ck, map_location='cpu', weights_only=False)
net = M.TFPred(**c['cfg'], scale=c.get('scale')).to(dev).eval(); net.load_state_dict(c['state'])
layers = list(range(3, 78)); fixed = {L: [] for L in layers}
p = P.TFGPUPredictor(layers, fixed, ck, device=dev, graph=dev != 'cpu')
wl = torch.tensor(np.diff(D.WIN), dtype=torch.float32, device=dev)
ex = zz['ex']; rq = zz['rq']; tok = zz['tok']; th = zz['think']
err = []; seen = set()
for t in range(n):
    q = int(rq[t]); nr = q not in seen
    if nr:
        seen.add(q); p.step(z['pf'][q], ntok=max(int(z['pfn'][q]), 17))     # prefill counts as one big step
    cnt = np.zeros((75, 256), np.float32); np.add.at(cnt, (np.repeat(np.arange(75), 8), ex[t].reshape(-1).astype(np.int64)), 1)
    tid = [154842] if (t > 0 and not th[t] and th[t - 1]) else None          # </think> at the transition token
    if p.step(cnt, 1, token_ids=tid):
        b = p.nblk - 1
        with torch.no_grad():
            x = B.inputs(torch.tensor([b]))
            if c['args'].get('noans'): x['ans'] = torch.zeros_like(x['ans'])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev != 'cpu'):
                ref = (torch.exp(net(**x).float()[0]) * wl)
        mu = p.mu
        err.append(float(((mu - ref).abs() / (ref.abs() + 1e-3)).max()))
print('blocks', len(err), 'max rel err', max(err), 'median', float(np.median(err)))
