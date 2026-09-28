from common import *
init()
import time
for E in [36,92,165]:
    t=time.time();w=native(E);x,p=sample(E);s=stats(E)
    G,D=grams(x,p,w)
    print(E,len(x),'p stats',p.min().item(),p.max().item(),(p[:1821]).mean().item() if E==36 else '', p[-100:].mean().item(),
      'gram match',((G-s['grams'][0].cuda()).norm()/s['grams'][0].norm()).item(),((D-s['grams'][1].cuda()).norm()/s['grams'][1].norm()).item(),time.time()-t,flush=True)
