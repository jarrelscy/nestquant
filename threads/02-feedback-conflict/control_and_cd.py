from common import *
from fb import *
import json
Ws, Hs = glm()
out = {}
# control: H = I (no feedback), real rotated gate weights
PI = Problem(Ws[0], torch.eye(6144, device='cuda'))
n2 = run(PI,'nat2',passes=5); n4 = run(PI,'nat4',passes=5); L2,L4=n2['l2'],n4['l4']
print('H=I control native', L2, L4, flush=True)
for rule,lam in [('seq',0),('joint',0.5),('joint',0.8),('joint',0.95)]:
    r = run(PI, rule, lam, passes=5); print('H=I', rule, lam, 'r2 %.4f r4 %.4f'%(r['l2']/L2, r['l4']/L4), flush=True)
# CD on real H
P = Problem(Ws[0], Hs[0])
n2 = run(P,'nat2'); n4 = run(P,'nat4'); L2,L4=n2['l2'],n4['l4']
for mu in [0.5, 0.7, 0.9]:
    r = run(P,'joint',mu)
    Q2,Q4,i = cd_joint(P, r['idx'], r['c2'], r['c4'], (1-mu)/L2, mu/L4)
    print('joint+CD mu=%.2f greedy r2 %.4f r4 %.4f -> CD r2 %.4f r4 %.4f'%(mu, r['l2']/L2, r['l4']/L4, P.loss(Q2)/L2, P.loss(Q4)/L4), flush=True)
