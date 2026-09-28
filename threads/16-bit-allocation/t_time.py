import time
from alloc_lib import *
data, Hs = load(16, 36)
for mi in (0, 2):
    t = time.time(); P = problem(data.teacher[mi], Hs[mi], mi); t1 = time.time()
    nb = P.Wn.shape[1] // 16
    o = fit(P, 0.3, [2]*nb, [2]*nb); t2 = time.time()
    o2 = fit(P, 0.3, [2]*nb, [2]*nb, cand_b=(1.5,2,2.5), cand_r=(0,1,2,3)); t3 = time.time()
    print(PROJ[mi], 'setup', t1-t, 'fit', t2-t1, 'fit+curves', t3-t2, o['l2'], o['l4'], torch.cuda.max_memory_allocated()/2**30, flush=True)
