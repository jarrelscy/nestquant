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
    for K in (2,3,4):
        Wq,_ = h.quantize_exl3_like(W, HG['H'][pi], K, count=1, sigma_reg=R.SIG[pn]); out[f'{pn}/exl3-{K}'] = pr(Wq); h.free_scratch()
    for lam in (0.0, 0.3):
        for gain in (0.85, 1.0, 1.15):
            _, dn, inf, _, _ = NE.encode_projection(None,None,None,None,P=P, shard_axis=R.AXIS[pn], lam=lam, gain=gain)
            out[f'{pn}/lam{lam}/gain{gain}'] = [pr(dn[L]) for L in (2,3,4)]
            print(pn, lam, gain, out[f'{pn}/lam{lam}/gain{gain}'], flush=True)
    # one-sided gate (no G) to separate G effect
    if pn == 'gate':
        P1 = NE.prep(W, HG['H'][pi], 1, R.SIG[pn], G=None)
        _, dn, inf, _, _ = NE.encode_projection(None,None,None,None,P=P1, shard_axis='n', lam=0.0)
        out['gate/noG/lam0'] = [pr(dn[L]) for L in (2,3,4)]
    NE.free(P)
print(json.dumps(out, indent=1))
json.dump(out, open('results/diag1_l16e36_proxy.json','w'), indent=1)
