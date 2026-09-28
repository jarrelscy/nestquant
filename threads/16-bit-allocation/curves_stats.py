import sys, torch
from alloc_lib import SCR, PROJ, allocate, shard_of
L, E = sys.argv[1], sys.argv[2]
for mi, p in enumerate(PROJ):
    c = torch.load(f'{SCR}/L{L}E{E}/{p}/curves_uni.pt')
    for lev, cur, rates, R in (('L2', c['c2'], (1.5, 2, 2.5), 2.0), ('L4', c['c4'], (0, 1.5, 2, 2.5, 3), 2.0), ('L4b', c['c4'], (1.5, 2, 2.5), 2.0)):
        cur = {k: v for k, v in cur.items() if k in rates}
        nc, nr = cur[2].shape; sh = shard_of(nc, nr, mi == 2)
        uni = float(cur[2].sum())
        a = allocate(cur, R, sh, rates); alloc = float(sum(cur[k][a == k].sum() for k in rates))
        a0 = allocate(cur, R, torch.zeros_like(sh), rates); alloc0 = float(sum(cur[k][a0 == k].sum() for k in rates))
        v = cur[2]; cv = float(v.std() / v.mean())
        # spread by column chunk vs row group
        colcv = float(v.sum(1).std() / v.sum(1).mean()); rowcv = float(v.sum(0).std() / v.sum(0).mean())
        print(f'{p:5s} {lev:3s} unit-cost CV {cv:.3f} (col {colcv:.3f}, row {rowcv:.3f}) | block-greedy est gain shard-const {10*torch.log10(torch.tensor(uni/alloc)):.3f} dB, free {10*torch.log10(torch.tensor(uni/alloc0)):.3f} dB | rate hist {[(k, int((a==k).sum())) for k in rates]}')
