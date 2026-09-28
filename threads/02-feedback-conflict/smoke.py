from common import *
import time
from fb import *
Ws, Hs = glm()
P = Problem(Ws[0], Hs[0])
print('AM/GM D', float(P.D.mean()/P.D.log().mean().exp()), 'kurtosis', float((P.Wn**4).mean()))
for rule in ['nat2','nat4','seq']:
    t=time.time(); r = run(P, rule, passes=3); print(rule, r['hist'], time.time()-t)
