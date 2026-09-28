import time
from alloc_lib import *
data, Hs = load(16, 36)
for mi in range(3):
    for rot in ('had', 'orth'):
        t = time.time(); P = problem(data.teacher[mi], Hs[mi], mi, rot); nc, nr = P.Wn.shape[1]//U, P.Wn.shape[0]//U
        two = torch.full((nc, nr), 2.0)
        o = fit(P, 0.3, two, two, cand_b=(1.5, 2, 2.5), cand_r=(0, 1.5, 2, 2.5, 3)) if rot == 'had' else fit(P, 0.3, two, two)
        print(PROJ[mi], rot, 'time', round(time.time()-t,1), 'l2 %.6f l4 %.6f' % (o['l2'], o['l4']), 'mem', round(torch.cuda.max_memory_allocated()/2**30,2), flush=True)
        save_w(P, o, f'{SCR}/L16E36/{PROJ[mi]}/uni_{rot}.pt')
        if rot == 'had': torch.save(dict(c2=o['c2'], c4=o['c4']), f'{SCR}/L16E36/{PROJ[mi]}/curves_uni.pt')
        del P, o; torch.cuda.empty_cache()
