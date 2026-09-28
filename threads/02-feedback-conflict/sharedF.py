from common import *
from fb import *
Ws, Hs = glm()
st = {e: torch.load(f'/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/statistics/l16_e{e}.pt', map_location='cpu', mmap=True, weights_only=True)['grams'][0].cuda() for e in [92,165]}
P = Problem(Ws[0], Hs[0])
n2 = run(P,'nat2'); n4 = run(P,'nat4'); L2,L4=n2['l2'],n4['l4']
Mtrue = P.M
Hmean = (Hs[0]/Hs[0].trace() + st[92]/st[92].trace() + st[165]/st[165].trace())/3
for name, Hsrc in [('E92', st[92]), ('E165', st[165]), ('mean3', Hmean)]:
    P.M = udu(P.Qi.T @ Hsrc.double() @ P.Qi)[0]
    r2 = run(P, 'nat2')
    r = innov_refine(P, torch.tensor(r2['c2']), torch.tensor(n2['c4']))
    print('feedback from', name, 'r2 %.4f  innov r4 %.4f'%(r['l2']/L2, r['l4']/L4), flush=True)
P.M = Mtrue
