import time, torch, sys
sys.path.insert(0, '.')
import nq_run as R, nq_encode as NE, nq_decode as D, harness as h
torch.cuda.set_per_process_memory_fraction(12/80); torch.backends.cuda.matmul.allow_tf32 = False
data = h.load_expert(16, 36); HG = R.glm_H(data, 16, 36)
for pi, pn in [(2,'down'), (0,'gate')]:
    t=time.time(); P = NE.prep(data.teacher[pi], HG['H'][pi], 1, R.SIG[pn], G=HG['G'][pi]); print(pn,'prep',time.time()-t, P['gs'], P['gsr'], flush=True)
    t=time.time()
    planes, dn, inf, P, enc = NE.encode_projection(None,None,None,None,P=P, shard_axis=R.AXIS[pn])
    print(pn,'enc',time.time()-t, inf['proxy_rot'], inf['bits'], flush=True)
    rot = D.rotated_levels(planes)
    print('rot exact', [torch.equal(rot[L], {2:enc['Q2'],4:enc['Q4']}.get(L, rot[L])) for L in (2,3,4)])
    print('dense exact', [torch.equal(D.decode_matrix(planes, L, rot=rot), dn[L]) for L in (2,3,4)])
    Wq, info = h.quantize_exl3_like(data.teacher[pi], HG['H'][pi], 2, count=1, sigma_reg=R.SIG[pn]); print('exl3-2 proxy', info['proxy'])
    Wq4, info4 = h.quantize_exl3_like(data.teacher[pi], HG['H'][pi], 4, count=1, sigma_reg=R.SIG[pn]); print('exl3-4 proxy', info4['proxy'])
    Ho = None
    def pr(Wd):
        E = (Wd - data.teacher[pi]).T; H = HG['H'][pi].cuda(); return float((E*(H@E)).sum()/ (data.teacher[pi].T*(H@data.teacher[pi].T)).sum())
    print('orig-basis proxy nq', [pr(dn[L]) for L in (2,3,4)], 'exl3', pr(Wq), pr(Wq4))
    print('mem', torch.cuda.max_memory_allocated()/2**30, D.plane_bytes(planes))
    NE.free(P); h.free_scratch()
