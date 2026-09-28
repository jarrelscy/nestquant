# MiMo L55 E70 down: real EXL3 trellis (thread-05 harness), proxy + routed rel-L2 on training-sample rows (in-sample).
import sys, json, torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/05-exl3-harness")
import harness as h
from orbit_duet.source import weights
h.gpu_cap()
SRC = '/tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source'
ST = '/home/coder/git/orbit-duet/runs/full55_statistics/l55_e70'
sig = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
T = [w.cuda().float() for w in weights(SRC, 55, 70)]
st = torch.load(ST + '.pt', weights_only=True, mmap=True); cnt = st['metadata']['training_rows']
Hx = st['grams'][0].cuda().float(); Hh = st['grams'][1].cuda().float()
smp = torch.load(ST + '_training_sample.pt', weights_only=True, mmap=True)
x = smp['x'][:8192].cuda(); p2 = smp['p'][:8192].cuda().double().square()
tb = [t.bfloat16() for t in T]
tgt = torch.cat([h._teacher(x[i:i+512], tb) for i in range(0, len(x), 512)]).double()
den = float((tgt.square().sum(-1) * p2).sum())
def outerr(ws):
    wb = [w.bfloat16() for w in ws]
    y = torch.cat([h._teacher(x[i:i+512], wb) for i in range(0, len(x), 512)]).double()
    return (float(((y - tgt).square().sum(-1) * p2).sum()) / den) ** .5
k = Hh.shape[0]; nblk = k // 16; shard = k // 8
srt = torch.argsort(Hh.diagonal()).cpu(); perm = torch.cat([srt[j::8] for j in range(8)]).cuda(); inv = torch.argsort(perm)
Hp = Hh[perm][:, perm].contiguous(); Wp = T[2][:, perm].contiguous()
def prof(Kb, d):  # shard-local low-energy half Kb-d, high-energy half Kb+d
    return [Kb - d if (j % (shard // 16)) < shard // 32 else Kb + d for j in range(nblk)]
res = {}
for Kb in [2, 3, 4]:
    g, _ = h.quantize_exl3_like(T[0], Hx, Kb, count=cnt, sigma_reg=sig); h.free_scratch()
    u, _ = h.quantize_exl3_like(T[1], Hx, Kb, count=cnt, sigma_reg=sig); h.free_scratch()
    runs = {"uniform": None, "ashard_uniform": 0, "ashard_pm1": 1}
    if Kb < 4: runs["ashard_pm0.5"] = 0.5
    for nm, d in runs.items():
        if d is None: dq, inf = h.quantize_exl3_like(T[2], Hh, Kb, count=cnt, sigma_reg=sig)
        else:
            dq, inf = h.quantize_exl3_like(Wp, Hp, prof(Kb, d), count=cnt, sigma_reg=sig); dq = dq[:, inv].contiguous()
        h.free_scratch()
        r = dict(bpw_down=inf["bpw"], proxy_down=h.proxy_loss(T[2], dq, Hh), routed_train=outerr([g, u, dq]))
        res[f"s{sig}_K{Kb}_{nm}"] = r
        print(f"MiMo s{sig} K{Kb} {nm:15s} bpw {r['bpw_down']:.3f} proxy {r['proxy_down']:.5f} routed(train rows) {r['routed_train']:.4f}", flush=True)
json.dump(res, open(f"/home/coder/git/nestquant/threads/01-rate-allocation/results/trellis_validate_mimo_{sig}.json", "w"), indent=1)
