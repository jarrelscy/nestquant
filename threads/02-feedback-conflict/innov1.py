from common import *
from fb import *
Ws, Hs = glm()
for name, W, H in [('gate',Ws[0],Hs[0]),('down',Ws[2],Hs[2])]:
    P = Problem(W, H)
    n2 = run(P,'nat2'); n4 = run(P,'nat4'); L2,L4=n2['l2'],n4['l4']
    r = innov_refine(P, torch.tensor(n2['c2']), torch.tensor(n2['c4']))
    print(name, 'native', L2, L4, 'innov r2 %.4f r4 %.4f (no-corr r4 %.3f)'%(r['l2']/L2, r['l4']/L4, r['l4_nocorr']/L4), flush=True)
    Mi = torch.linalg.solve_triangular(P.M, torch.eye(P.M.shape[0],device='cuda'), upper=True)
    A = Mi - torch.eye(Mi.shape[0], device='cuda')
    S = torch.linalg.svdvals(A.double())
    e = (S**2).cumsum(0)/(S**2).sum()
    print(name, '||Minv-I||_F^2', float((S**2).sum()), 'rank for 50/80/90/99% energy', [int((e<t).sum())+1 for t in [.5,.8,.9,.99]])
    for B in [64,128,256,512,1024]:
        bd = torch.block_diag(*[Mi[a:a+B,a:a+B] for a in range(0,Mi.shape[0],B)])
        r = innov_refine(P, torch.tensor(n2['c2']), torch.tensor(n2['c4']), G=bd)
        print(name, 'blockdiag G B=%d r4 %.4f'%(B, r['l4']/L4), flush=True)
    for k in [16,64,256]:
        U,S_,Vh = torch.linalg.svd(A.double())
        G = torch.eye(A.shape[0],device='cuda') + ((U[:,:k]*S_[:k])@Vh[:k]).float()
        r = innov_refine(P, torch.tensor(n2['c2']), torch.tensor(n2['c4']), G=G)
        print(name, 'lowrank G k=%d r4 %.4f'%(k, r['l4']/L4), flush=True)
