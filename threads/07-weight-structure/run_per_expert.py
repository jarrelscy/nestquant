import ws,json,torch,sys,time
out={}
jobs=[('glm',L,E) for L in (16,49,66) for E in (36,92,165)]+[('mimo',55,70)]
for m,L,E in jobs:
    t=time.time()
    if m=='glm': w=ws.glm_expert(L,E)
    else: w=[ws.dequantize(a,b) for a,b in ws.mimo_raw(L,E)]
    out[f'{m}_l{L}_e{E}']=ws.per_expert(w); print(m,L,E,time.time()-t,flush=True)
    json.dump(out,open('per_expert.json','w'),indent=1)
