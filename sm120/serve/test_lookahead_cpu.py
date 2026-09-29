"""CPU-only test of nq_lookahead (fake routers + scheduler + executor): plan / oplog / landed accounting / accuracy stats.
usage: [MODE=lookahead|chunk] [WS=0.05] python test_lookahead_cpu.py"""
import sys,os,types,collections,numpy as np,torch,logging
logging.basicConfig(level=logging.INFO)
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../../streaming',HERE+'/..']
os.environ.update(NQ_PREFILL_ADAPT=os.environ.get('MODE','lookahead'),NQ_LA_D='2',NQ_LA_MEASURE='1',NQ_LA_OUT='/tmp/la/la_stats.json',NQ_HOME='/data/Jarrel/nestquant',
                  NQ_DELTA_TABLE='/data/Jarrel/nq-serve/predictor/delta_table.json',NQ_PREDICTOR='ema')
import nq_lookahead as LAH,scheduler as SC
NE=256;Ls=list(range(3,78));H=64
g=torch.Generator().manual_seed(0)
Wg={L:torch.randn(NE,H,generator=g)*float(os.environ.get("WS","1")) for L in Ls};Bg={L:torch.randn(NE,generator=g)*0.1 for L in Ls}
class FakeGate:
    def __init__(s,L):s.L=L;s.weight=Wg[L]
    def __call__(s,x):return x.float()@Wg[s.L].T,None
class FakeRouter:
    def __init__(s,L):s.L=L;s.capture_fn=None
    def select_experts(s,hidden_states,router_logits):
        sc=router_logits.sigmoid();ids=torch.topk(sc+Bg[s.L],8,1).indices;w=sc.gather(1,ids);w=w/w.sum(1,keepdim=True)*2.5;return w,ids
for L in Ls:LAH._run[L]=types.SimpleNamespace(gate=FakeGate(L),router=FakeRouter(L))
fj=__import__('json').load(open('/data/Jarrel/nestquant/threads/22-boundary-experts/fixed_set.json'))
fx={L:fj['fixed_set'][str(L)][:26] for L in Ls};dflt={L:[e for e in range(NE) if e not in fx[L]][:51] for L in Ls}
S=SC.Scheduler(Ls,fx,dflt,2.56e6*4,NE=NE,n_float=51,slots=56*len(Ls),cap_GBps=1e6,predictor='ema')
for L in Ls:
    for e in dflt[L]:S.state[S.li[L],e]=2
class FX:
    def __init__(s):s.calls=[]
    def apply(s,u,d,S_):s.calls.append((list(u),list(d)))
class Ev:
    def __init__(s,*a,**k):pass
    def record(s):pass
    def query(s):return True
    def elapsed_time(s,o):return 0.1
torch.cuda.Event=Ev
class OL:
    def __init__(s):s.recs=[]
    def put(s,u,d):s.recs.append((u,d))
rt=types.SimpleNamespace(lay={L:None for L in Ls},dev=torch.device('cpu'),rank=0,log=OL(),F=None,S=S,wake=types.SimpleNamespace(set=lambda:None))
_pin=torch.Tensor.pin_memory;torch.Tensor.pin_memory=lambda s:s
la=LAH.LA(rt);X=FX()
tables={L:torch.full((NE,20),2,dtype=torch.int64) for L in Ls}
for L in Ls:tables[L][fx[L],0]=4;tables[L][dflt[L],0]=4
res=[]
for chunk in range(3):
    x=torch.randn(512,H,generator=g)
    for L in Ls:
        xl=x+0.3*torch.randn(512,H,generator=g)       # slowly drifting residual
        r=LAH.route(L,xl);ids,w=r
        la.pre(L,xl.bfloat16(),ids,w.half(),tables[L])
        n=la.service(S,X,rt.log)
        # emulate: upgrades land immediately except every 3rd
        for u,d in X.calls:
            for (Lx,e) in u:
                if e%3:S.landed(Lx,e);tables[Lx][e,0]=4
            for (Lx,e) in d:S.released(Lx,e);tables[Lx][e,0]=2
        X.calls=[]
        x=xl
ups=sum(len(u) for u,d in rt.log.recs);downs=sum(len(d) for u,d in rt.log.recs)
print('oplog ups',ups,'downs',downs,'state counts',np.bincount(S.state.ravel(),minlength=4),'in use',int((S.state>0).sum()),'slots',S.slots)
for i,L in enumerate(Ls):
    bad=np.nonzero(S.fixed[i]&(S.state[i]>0))[0]
    if len(bad):print("L",L,"bad",bad[:5],S.state[i][bad[:5]],[e in dflt[L] for e in bad[:5]]);break
    assert not (S.fixed[i]&(S.state[i]>0)).any(),'fixed expert streamed'
    assert (S.want[i]&S.fixed[i]).sum()==0
    assert S.want[i].sum()<=51,S.want[i].sum()
per=collections.Counter(L for u,d in rt.log.recs for L,e in u);assert max(per.values())<=45*3,per
import json;d=json.load(open('/tmp/la/la_stats.json'));a=d['bands']['all']
for sn in a:print(sn,{rn:a[sn][rn]['count']['45'] for rn in ('count','gate','dsal') if rn in a[sn]},'rec',a[sn]['count']['rec']['45'] if 'count' in a[sn] else '')
assert abs(a['d0']['count']['count']['45']-a['oracle']['count']['count']['45'])<2e-3,'d0 != oracle'
print('TEST PASS')
