from t14 import *
from orbit_duet.source import weights
for L in (16, 49, 66):
    for E in (36, 92, 165):
        Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E); hessians(L, E, Ws); print(L, E, flush=True); torch.cuda.empty_cache()
