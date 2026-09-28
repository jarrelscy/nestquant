from common import *
from fb import *
Ws, Hs = glm()
P = Problem(Ws[0], Hs[0])
n2 = run(P,'nat2'); n4 = run(P,'nat4')
L2, L4 = n2['l2'], n4['l4']
print('native', L2, L4, 'nat4 2-bit view', n4['l2']/L2)
# diagnostics from seq pass
Q2,Q4,idx,T2,T4 = fit(P, torch.tensor(n2['c2']), torch.tensor(n2['c4']), 'seq')
W=P.Wn
print('var eta2 %.4f var E2 %.4f var(t2-w) %.4f var(t4-w) %.4f var(t2-t4) %.4f'%(
 float((Q2-T2).var()), float((Q2-W).var()), float((T2-W).var()), float((T4-W).var()), float((T2-T4).var())))
print('frac cell(t4) != base', float(((T4[...,None]-torch.tensor(n4['c4'],device='cuda')).abs().argmin(-1)>>2 != (idx.long()>>2)).float().mean()))
def rep(tag, r): print(tag, 'r2 %.4f r4 %.4f'%(r['l2']/L2, r['l4']/L4), flush=True)
for lam in [0.5, 0.8, 0.9, 0.95]:
    for mu in [0.5, lam]:
        rep(f'shared lam={lam} mu={mu}', run(P,'shared',lam,mu=mu))
c2 = torch.tensor(n2['c2'])
for d in [0.3,0.5,0.8,1.2]:
    c4 = (c2[:,None] + d*(torch.arange(4)[None]-1.5)).flatten()
    rep(f'seq widechildren d={d}', run(P,'seq',c2=c2.clone(),c4=c4,passes=4))
    rep(f'joint0.7 widechildren d={d}', run(P,'joint',0.7,c2=c2.clone(),c4=c4,passes=4))
