from common import *
from fb import *
Ws, Hs = glm()
for name, W, H in [('gate',Ws[0],Hs[0]),('down',Ws[2],Hs[2])]:
    P = Problem(W, H)
    n2 = run(P,'nat2'); n4 = run(P,'nat4'); L2,L4=n2['l2'],n4['l4']
    Mtrue = P.M
    ev, V = torch.linalg.eigh(P.Hr)
    ev, V = ev.flip(0), V.flip(1)
    print(name, 'eig top/median', [round(float(ev[i]/ev[len(ev)//2]),1) for i in [0,1,3,7,15,63,255]], flush=True)
    for k in [0, 8, 32, 128, 512]:
        Lk = (V[:, :k] * ev[:k]) @ V[:, :k].T
        Hk = Lk + torch.diag((P.Hr - Lk).diagonal())
        P.M = udu(Hk)[0]
        r2 = run(P, 'nat2')
        r = innov_refine(P, torch.tensor(r2['c2']), torch.tensor(n2['c4']))
        print(name, 'k=%d r2 %.4f innov r4 %.4f'%(k, r['l2']/L2, r['l4']/L4), flush=True)
    P.M = Mtrue
