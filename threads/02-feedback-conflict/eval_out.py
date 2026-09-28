"""Relative expert-output L2: training_sample rows (fit decisions) and the frozen matched capture (final, via thread-05 harness)."""
import sys, json, os, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness'); sys.path.insert(0, '/home/coder/git/orbit-duet')
import harness as h; h.gpu_cap(12)
from orbit_duet.evaluate import teacher
fam = sys.argv[1]                      # deq | deq_trellis
tags = sys.argv[2].split(',')
root = f'/tmp/nestquant/02-feedback-conflict/{fam}'
data = h.load_expert(16, 36)
native = data.teacher
ts = torch.load('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/statistics/l16_e36_training_sample.pt', map_location='cpu', mmap=True, weights_only=False)
methods = {}
for t in tags:
    parts = [torch.load(f'{root}/{m}/{t}.pt') for m in ['gate', 'up', 'down']]
    methods[f'{t}@2'] = [p['w2'].cuda().float() for p in parts]
    methods[f'{t}@4'] = [p['w4'].cuda().float() for p in parts]
for b in [2, 4]:
    methods[f'EXL3-{b}'] = h.load_exl3_bin(f'/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/expert_{b}.bin')
train = {k: [0., 0.] for k in methods}
with torch.no_grad():
    for a in range(0, len(ts['x']), 1024):
        x = ts['x'][a:a+1024].cuda(); p2 = ts['p'][a:a+1024].cuda().double().square()
        y = teacher(x, native).double(); den = float((y.square().sum(-1) * p2).sum())
        for k, w in methods.items():
            e = (teacher(x, w).double() - y).square().sum(-1)
            train[k][0] += float((e * p2).sum()); train[k][1] += den
train = {k: (v[0] / v[1]) ** .5 for k, v in train.items()}
ev = h.evaluate(data, methods)
tab = h.table(ev)
out = dict(family=fam, train_sample_rel_l2=train, capture=tab)
json.dump(out, open(f'/home/coder/git/nestquant/threads/02-feedback-conflict/results/output_{fam}.json', 'w'), indent=1)
for k in methods:
    print(f'{k:24s} train {100*train[k]:7.3f}  ', {kk: v for kk, v in tab[k].items()})
