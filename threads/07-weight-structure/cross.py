"""Cross-expert sharing within a layer: mean-expert delta coding, shared input/output KLT basis, row alignment."""
import ws,json,torch,math,time,sys
L=int(sys.argv[1]); NPOOL=int(sys.argv[2]) if len(sys.argv)>2 else 64
targets=[36,92,165]
pool=[E for E in range(256) if E not in targets][:NPOOL] if L!=66 else []
t0=time.time()
def mats(w):  # (name, matrix whose ROWS are samples in the shared 6144-d space)
    g,u,d=[x.double() for x in w]
    return dict(gate_in=g,up_in=u,gateup_in=torch.cat([g,u]),down_out=d.T)
acc={}; S={}; n=0
def add(w,sign=1):
    for k,m in mats(w).items():
        acc[k]=acc.get(k,0)+sign*(m.T@m)
    for i,p in enumerate(ws.PROJ): S[p]=S.get(p,0)+sign*w[i].double()
cache={}
for E in pool:
    add(ws.glm_expert(L,E)); n+=1
for E in targets:
    cache[E]=ws.glm_expert(L,E)
print('loaded',time.time()-t0,flush=True)
res={}
gctrl=[torch.randn(2048,6144,dtype=torch.float64,generator=torch.Generator().manual_seed(9)) for _ in range(2)]
for E in targets:
    w=cache[E]; r={}
    # leave-one-out pool: pool experts + the other targets (never e itself)
    C={k:acc[k].clone() for k in acc}; Ssum={p:S[p].clone() for p in S}; cnt=n
    for F in targets:
        if F==E: continue
        for k,m in mats(cache[F]).items(): C[k]+=m.T@m
        for i,p in enumerate(ws.PROJ): Ssum[p]+=cache[F][i].double()
        cnt+=1
    for i,p in enumerate(ws.PROJ):
        W=w[i].double(); M=Ssum[p]/cnt
        a=float((W*M).sum()/(M*M).sum()); resid=(W-a*M).square().sum()/W.square().sum()
        r[f'mean_{p}']=dict(cos=float((W*M).sum()/W.norm()/M.norm()),delta_gain_db=float(-10*math.log10(resid)),n_experts=cnt)
    for k,m in mats(w).items():
        ev,U=torch.linalg.eigh(C[k])
        v=((m@U).square().mean(0))
        vh=ws.rot_in(m.float()).double().square().mean(0); vi=m.square().mean(0)
        gc=gctrl[0] if m.shape[0]==2048 else torch.cat(gctrl); vc=(gc@U).square().mean(0)*float(m.square().mean())
        # energy captured by top-q shared components
        order=ev.argsort(descending=True); vs=v[order]; cap={q:float(vs[:q].sum()/vs.sum()) for q in (64,256,1024,2048)}
        r[k]=dict(shared_klt=dict(amgm_db=ws.amgm_db(v),rwf2=ws.rwf_gain(v,2),rwf4=ws.rwf_gain(v,4)),
                  gauss_ctrl_in_klt=dict(amgm_db=ws.amgm_db(vc),rwf2=ws.rwf_gain(vc,2),rwf4=ws.rwf_gain(vc,4)),
                  hadamard=dict(amgm_db=ws.amgm_db(vh),rwf2=ws.rwf_gain(vh,2),rwf4=ws.rwf_gain(vh,4)),
                  identity=dict(amgm_db=ws.amgm_db(vi),rwf2=ws.rwf_gain(vi,2),rwf4=ws.rwf_gain(vi,4)),
                  top_q_energy=cap,chance={q:q/6144 for q in (64,256,1024,2048)})
    # row alignment (upcycling check): best |cos| of each gate row of E against gate rows of another expert
    F=[x for x in targets if x!=E][0]
    for i,p in enumerate(['gate_proj','up_proj']):
        A=torch.nn.functional.normalize(w[i].float(),dim=1); B=torch.nn.functional.normalize(cache[F][i].float(),dim=1)
        c=(A@B.T).abs(); r[f'rowmatch_{p}']=dict(mean_best=float(c.max(1).values.mean()),same_index=float(c.diagonal().mean()),ctrl_expected=math.sqrt(2*math.log(2048)/6144))
    res[E]=r; print(E,time.time()-t0,flush=True)
json.dump(res,open(f'cross_l{L}.json','w'),indent=1)
