"""EXL3 1.5.1 fused MoE decode path (ext.exl3_moe_coop = the kernels behind BC_BlockSparseMLP.run_bszN),
random mul1 trellis weights, GLM shapes. Mixed 2/4 bit: EXL3 needs one K per launch, so a mixed layer is two coop
launches over expert ranges [0,n2) and [n2,n) (out-of-range picks masked to zero in-kernel); the second launch
merges the first's output through the shared-expert input (sh_out), so there is no extra add kernel."""
import torch
from exllamav3.ext import exllamav3_ext as ext
class EXL3Group:
    def __init__(s,n,K,H=6144,I=2048,seed=0):
        g=torch.Generator(device='cuda').manual_seed(seed)
        ri=lambda *sh:torch.randint(-32768,32767,sh,device='cuda',dtype=torch.int16,generator=g)
        sg=lambda m:(torch.randint(0,2,(m,),device='cuda',generator=g)*2-1).half()
        s.K=K;s.n=n
        s.gt=[ri(H//16,I//16,16*K) for _ in range(n)];s.ut=[ri(H//16,I//16,16*K) for _ in range(n)];s.dt=[ri(I//16,H//16,16*K) for _ in range(n)]
        s.gsu=[sg(H) for _ in range(n)];s.gsv=[sg(I) for _ in range(n)];s.usu=[sg(H) for _ in range(n)];s.usv=[sg(I) for _ in range(n)]
        s.dsu=[sg(I) for _ in range(n)];s.dsv=[sg(H) for _ in range(n)]
        P=lambda L:torch.tensor([t.data_ptr() for t in L],dtype=torch.long,device='cuda')
        s.ptr=[P(x) for x in (s.gt,s.gsu,s.gsv,s.ut,s.usu,s.usv,s.dt,s.dsu,s.dsv)]
        s.bytes_per_expert=sum(t.numel()*2 for t in (s.gt[0],s.ut[0],s.dt[0]))
class EXL3MoE:
    """groups: list of (EXL3Group, first_expert_id). Each group is one coop launch."""
    def __init__(s,groups,B,H=6144,I=2048,smax=None):
        """smax: scratch rows (slots_max). EXL3_MOE_COOP_KSPLIT=k needs smax >= k*B*8 (partials at slot + ks*slots)."""
        s.groups=groups;s.B=B;S=smax or B*8
        s.had_g=torch.empty(S,H,device='cuda').half();s.had_u=torch.empty_like(s.had_g)
        s.gu_g=torch.empty(S,I,device='cuda').half();s.gu_u=torch.empty_like(s.gu_g);s.act=torch.empty(S,I,device='cuda').half()
        s.d_out=torch.empty(S,H,device='cuda')
        L=S*(I//128)+B*(H//128)+2+(S+1)+S
        s.ctr=[torch.zeros(L,dtype=torch.int32,device='cuda') for _ in groups]
        s.outs=[torch.zeros(B,H,device='cuda') for _ in groups]
    def __call__(s,x,sel,rw):
        prev=None
        for (G,e0),ctr,out in zip(s.groups,s.ctr,s.outs):
            lo,hi=(e0,e0+G.n) if len(s.groups)>1 else (-1,-1)
            ext.exl3_moe_coop(x,sel,rw,lo,hi,x.shape[1],*G.ptr,None,None,None,float(G.K),float(G.K),float(G.K),False,True,0,0.0,True,
                              s.had_g,s.had_u,s.gu_g,s.gu_u,s.act,s.d_out,ctr,out,prev,None)
            prev=out
        return prev
    def dense_ref(s,x,sel,rw):
        """fp32 reference from EXL3's own decoded weights (via LinearEXL3.get_weight_tensor)."""
        from exllamav3.modules.quant.exl3 import LinearEXL3
        y=torch.zeros(x.shape[0],x.shape[1],device='cuda')
        for b in range(x.shape[0]):
            for k in range(sel.shape[1]):
                e=int(sel[b,k])
                for G,e0 in s.groups:
                    if e0<=e<e0+G.n:
                        i=e-e0;mk=lambda t,su,sv,a,b_:LinearEXL3(None,a,b_,suh=su,svh=sv,trellis=t,mul1=torch.tensor(1,dtype=torch.int32)).get_weight_tensor().float()
                        Wg=mk(G.gt[i],G.gsu[i],G.gsv[i],x.shape[1],G.gsv[i].numel());Wu=mk(G.ut[i],G.usu[i],G.usv[i],x.shape[1],G.usv[i].numel())
                        Wd=mk(G.dt[i],G.dsu[i],G.dsv[i],G.dsu[i].numel(),x.shape[1])
                        xv=x[b].float();h=torch.nn.functional.silu(xv@Wg)*(xv@Wu);y[b]+=float(rw[b,k])*(h@Wd)
        return y
