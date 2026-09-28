"""Per-projection sensitivity: quantize one projection at a time (others teacher) from thread-02 deq files."""
import sys, json, torch
sys.path.insert(0, '/home/coder/git/nestquant/threads/05-exl3-harness')
import harness as h; h.gpu_cap(12)
data = h.load_expert(16, 36); T = data.teacher
D = '/tmp/nestquant/02-feedback-conflict/deq_mix_full'
P = ['gate', 'up', 'down']
m = {}
for tag, key in [('EXL3mix_4', 'w4'), ('EXL3mix_2', 'w4'), ('blend_0.3', 'w4'), ('blend_0.3', 'w2'), ('innov_exact_Minv', 'w4')]:
    ws = [torch.load(f'{D}/{p}/{tag}.pt')[key].cuda().float() for p in P]
    m[f'{tag}:{key}:all'] = ws
    for i in range(3):
        m[f'{tag}:{key}:only_{P[i]}'] = [ws[j] if j == i else T[j] for j in range(3)]
tab = h.table(h.evaluate(data, m))
for k, v in tab.items(): print(f'{k:40s} routed {v["all/routed"]:7.3f} forced {v["all/forced"]:7.3f} ood {v["ood/forced"]:7.3f}')
json.dump(tab, open('results_sens.json', 'w'), indent=1)
