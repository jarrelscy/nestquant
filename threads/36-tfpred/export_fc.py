"""export a trained TFPred checkpoint as nq-algo window forecasts: /data/Jarrel/nq-algo/fc/<task>.<name>.npy [NB,75,256,W]
float16 = expected hits in windows data.WIN after the end of each block (causal: row b only reads blocks <= b), + .json edges.
  export_fc.py --ckpt ckpt.pt --name tf1 --tasks a,b,c [--data ids|cap] [--dev cuda:0] [--bs 16]"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import data as D, model as M

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--name', required=True); ap.add_argument('--tasks', default=','.join(D.TEST_TASKS))
ap.add_argument('--data', default='ids'); ap.add_argument('--dev', default='cuda:0'); ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--out', default='/data/Jarrel/nq-algo/fc')
a = ap.parse_args()
dev = torch.device(a.dev)
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); cfg = ck['cfg']; ar = ck['args']
net = M.TFPred(**cfg, scale=ck.get('scale')).to(dev).eval(); net.load_state_dict(ck['state'])
wlen = torch.tensor(np.diff(D.WIN), dtype=torch.float32, device=dev)
for t in a.tasks.split(','):
    d = D.ids_blocks(t, a.data)
    B = D.Blocks([d], torch.device('cpu'), use_sal=bool(ar.get('sal')), cdev=dev)
    if ar.get('sal'):
        sc = torch.as_tensor(ck['scale'])[None, :, None]
        B.sal = (B.sal.float() / sc).half(); B.ch = (B.ch.float() / sc).half()
    p = f'{a.out}/{t}.{a.name}.npy'
    F = np.lib.format.open_memmap(p + '.tmp.npy', mode='w+', dtype=np.float16, shape=(B.nb, 75, 256, len(D.WIN) - 1))
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, B.nb, a.bs):
            b = torch.arange(i, min(i + a.bs, B.nb), device=dev)
            x = B.inputs(b)
            if not cfg['content']:
                x.pop('tok', None); x.pop('tokm', None); x.pop('hp', None)
            if ar.get('noans'):
                x['ans'] = torch.zeros_like(x['ans'])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                lr = net(**x)
            mu = torch.exp(lr.float()) * wlen                              # rate per token -> hits per window
            if ar.get('sal'):
                mu = mu * torch.as_tensor(ck['scale'], device=dev)[None, :, None, None]
            F[i:i + len(b)] = mu.clamp(max=6e4).half().cpu().numpy()
    F.flush(); del F
    os.replace(p + '.tmp.npy', p)
    json.dump(dict(edges=D.WIN, ckpt=os.path.abspath(a.ckpt), sal=bool(ar.get('sal'))), open(p[:-4] + '.json', 'w'))
    print(t, B.nb, 'blocks', '%.0fs' % (time.time() - t0), flush=True)
