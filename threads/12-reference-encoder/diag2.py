import time, torch, sys, json
sys.path.insert(0, '.')
import nq_run as R, nq_encode as NE, nq_decode as D, harness as h
torch.cuda.set_per_process_memory_fraction(12/80); torch.backends.cuda.matmul.allow_tf32 = False
data = h.load_expert(16, 36); HG = R.glm_H(data, 16, 36)
out = {}
for pi, pn in [(2,'down'), (0,'gate')]:
    W = data.teacher[pi]; H = HG['H'][pi].cuda()
    def pr(Wd):
        E = (Wd - W).T; return float((E*(H@E)).sum()/ (W.T*(H@W.T)).sum())
    P = NE.prep(W, HG['H'][pi], 1, R.SIG[pn], G=HG['G'][pi])
    for lam in (0.0, 0.3):
      for inner in (1, 2, 4):
        t=time.time()
        _, dn, inf, _, _ = NE.encode_projection(None,None,None,None,P=P, shard_axis=R.AXIS[pn], lam=lam, inner=inner)
        out[f'{pn}/lam{lam}/inner{inner}'] = [round(pr(dn[L]),6) for L in (2,3,4)]
        print(pn, lam, inner, out[f'{pn}/lam{lam}/inner{inner}'], round(time.time()-t,1), flush=True)
    NE.free(P)
json.dump(out, open('results/diag2_l16e36_inner.json','w'), indent=1)
