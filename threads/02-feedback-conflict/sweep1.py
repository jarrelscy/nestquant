from common import *
from fb import *
import json
Ws, Hs = glm()
P = Problem(Ws[0], Hs[0])
n2 = run(P,'nat2'); n4 = run(P,'nat4')
print('native', n2['l2'], n4['l4'])
for rule, lams in [('blend',[0.1,0.2,0.3,0.5,0.7,0.9,1.0]),('joint',[0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.97])]:
    for l in lams:
        r = run(P, rule, l)
        print(rule, l, 'r2 %.4f r4 %.4f'%(r['l2']/n2['l2'], r['l4']/n4['l4']), flush=True)
