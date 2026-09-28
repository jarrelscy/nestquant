from common import *
from fbt import *
import sys
mi = int(sys.argv[1]); name=['gate','up','down'][mi]
Ws, Hs = glm(); P = Problem(Ws[mi], Hs[mi])
n2 = run_trellis(P, 'nat2'); n4 = run_trellis(P, 'nat4'); L2, L4 = n2['l2'], n4['l4']
Q2 = n2['Q2']; n = Q2.shape[1]
Mi = torch.linalg.solve_triangular(P.M, torch.eye(n, device='cuda'), upper=True)
print(name, 'native', L2, L4)
for tag, G in [('I', torch.eye(n, device='cuda')), ('Minv', Mi)] + [
        (f'blockdiag_Minv_B{B}', torch.block_diag(*[Mi[a:a+B, a:a+B] for a in range(0, n, B)])) for B in [16, 32, 64, 128, 256]]:
    W4 = g_refine(P, Q2, G)
    print(name, f'G={tag:20s} r4 {P.loss(W4)/L4:.4f}', flush=True)
