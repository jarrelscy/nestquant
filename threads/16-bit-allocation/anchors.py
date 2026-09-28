"""EXL3-2/4 anchors refit under the same thread-08 H (quantize_exl3_like, sigma_reg = DAMP), saved as EXL3_{K}.pt."""
import sys
from alloc_lib import *
L, E = int(sys.argv[1]), int(sys.argv[2])
data, Hs = load(L, E)
for mi, pn in enumerate(PROJ):
    for K in (2, 4):
        Wq, info = hh.quantize_exl3_like(data.teacher[mi], Hs[mi], K, count=1, sigma_reg=DAMP[mi])
        os.makedirs(f'{SCR}/L{L}E{E}/{pn}', exist_ok=True)
        torch.save(dict(w2=Wq.bfloat16().cpu(), w4=Wq.bfloat16().cpu()), f'{SCR}/L{L}E{E}/{pn}/EXL3_{K}.pt')
        hh.free_scratch(); torch.cuda.empty_cache()
    print(L, E, pn, 'anchors done', flush=True)
