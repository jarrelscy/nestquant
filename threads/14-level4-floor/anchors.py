"""Same-H EXL3 anchors (quantize_exl3_like, thread-08 H, sigma 0.5/0.5/1.0) for the 9 GLM experts."""
from t14 import *
from orbit_duet.source import weights
for L in (16, 49, 66):
    for E in (36, 92, 165):
        Ws = weights('/tmp/nestquant/glm53-fp8-experts', L, E); Hs = hessians(L, E, Ws)
        for K in (2, 4):
            f = f'{SCR}/exl3_l{L}_e{E}_K{K}.pt'
            if os.path.exists(f): continue
            out = []
            for mi in range(3):
                Wq, info = h.quantize_exl3_like(Ws[mi], Hs[mi], K, count=1, sigma_reg=DAMP[mi])
                out.append(Wq.bfloat16().cpu()); h.free_scratch()
            torch.save(out, f); print(L, E, K, flush=True)
        del Hs; torch.cuda.empty_cache()
