"""CPU tests of NQ_LMPF (no GPU, no vLLM): CUDA_VISIBLE_DEVICES= python test_lmpf_cpu.py"""
import os,sys,json,types,time,tempfile,dataclasses,collections
os.environ['NQ_LMPF']='1'
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,torch
import nq_lmpf_engine as EN,nq_lmpf as LP

# ---------------- engine side ----------------
class Req:
    def __init__(s,rid,prompt,computed=0):
        s.request_id=rid;s.num_prompt_tokens=prompt;s.num_tokens=prompt;s.num_computed_tokens=computed
        s.has_encoder_inputs=False;s.lora_request=None;s.pooling_params=None;s.status=types.SimpleNamespace(name='WAITING')
class Q(list):
    def peek_request(s):return s[0]
class Sched:
    """mock: schedules min(budget, left) tokens of the first running / waiting request (alone if max_num_running_reqs 1)"""
    def __init__(s,mnbt=4096):
        s.max_num_scheduled_tokens=mnbt;s.max_num_running_reqs=4;s.running=[];s.waiting=Q();s.requests={}
        s.vllm_config=types.SimpleNamespace(parallel_config=types.SimpleNamespace(world_size=2));s.log=[]
    def _select_waiting_queue_for_scheduling(s):return s.waiting if s.waiting else None
    def add(s,r):s.requests[r.request_id]=r;s.waiting.append(r)
    def schedule(s):
        s.log.append((s.max_num_scheduled_tokens,s.max_num_running_reqs))
        b=s.max_num_scheduled_tokens;nst={}
        for r in list(s.running):
            if r.num_computed_tokens>=r.num_prompt_tokens:n=1
            else:n=min(b,r.num_prompt_tokens-r.num_computed_tokens)
            if b<=0:break
            n=min(n,b);r.num_computed_tokens+=n;nst[r.request_id]=n;b-=n
        while s.waiting and len(s.running)<s.max_num_running_reqs and b>0:
            r=s.waiting.pop(0);r.status.name='RUNNING';s.running.append(r)
            n=min(b,r.num_prompt_tokens-r.num_computed_tokens);r.num_computed_tokens+=n;nst[r.request_id]=n;b-=n
        return types.SimpleNamespace(num_scheduled_tokens=nst,scheduled_spec_decode_tokens={})

def mkengine(world=2,W=None,mnbt=4096):
    mod=types.SimpleNamespace(Scheduler=type('S',(Sched,),{}))
    EN.patch_scheduler(mod);sc=mod.Scheduler(mnbt);sc.vllm_config.parallel_config.world_size=world
    key=f'test{os.getpid()}_{time.time_ns()}'
    st=EN.State(sc,key=key);EN._ST[id(sc)]=st
    if W is not None:
        for r in range(world):LP.write_json(EN.ready_path(key,r),dict(W=W[r] if isinstance(W,list) else W,C=256,mnbt=mnbt))
    return sc,st
def cleanup(st):
    for r in range(4):
        p=EN.ready_path(st.key,r)
        if os.path.exists(p):os.remove(p)
def drive(sc,n=200):
    outs=[]
    for _ in range(n):
        o=sc.schedule()
        if not o.num_scheduled_tokens:break
        outs.append(o)
        if all(r.num_computed_tokens>=r.num_prompt_tokens for r in sc.running):break
    return outs

def test_windows_decide():
    assert EN.windows(100000,32768)==(4,25000)
    assert EN.windows(32768,32768)==(1,32768)
    assert EN.windows(1,32768)==(1,1)
    assert EN.windows(131080,65536,4096)==(3,45056)
    assert EN.windows(65544,65536,4096)==(2,36864)
    assert EN.windows(131072,65536,4096)==(2,65536)
    assert EN.decide(40000,32768,1024,2.)=='full' and EN.decide(2000,32768,1024,2.)=='bud'
    assert EN.decide(2000,32768,1024,0.) is None and EN.decide(500,32768,1024,2.) is None

def test_full_windows():
    sc,st=mkengine(W=32768)
    try:
        sc.add(Req('a',100000));outs=drive(sc)
        ann=[o.nq_lmpf for o in outs]
        assert [a['n'] for a in ann]==[28672,24576,24576,22176],[a['n'] for a in ann]
        assert all(a['m']=='full' and a['rid']=='a' and a['end']==100000 for a in ann)
        assert [a['start'] for a in ann]==[0,28672,53248,77824] and [a['last'] for a in ann]==[False]*3+[True]
        assert sc.max_num_scheduled_tokens==4096 and sc.max_num_running_reqs==4    # restored
        assert sc.log==[(28672,1),(24576,1),(24576,1),(24576,1)],sc.log
    finally:cleanup(st)

def test_ready_min_and_missing():
    sc,st=mkengine(W=[32768,16384])
    try:
        sc.add(Req('a',40000));outs=drive(sc)
        assert [o.nq_lmpf['n'] for o in outs]==[16384,12288,11328],[o.nq_lmpf['n'] for o in outs]
    finally:cleanup(st)
    sc,st=mkengine(W=None)                                       # no ready files: prod path, no annotation
    sc.add(Req('a',40000));outs=drive(sc)
    assert all(o.nq_lmpf is None for o in outs) and [sum(o.num_scheduled_tokens.values()) for o in outs][0]==4096

def test_bud_single_and_multi():
    sc,st=mkengine(W=32768)
    try:
        sc.add(Req('s',2048));o=drive(sc)
        assert len(o)==1 and o[0].nq_lmpf['m']=='bud' and o[0].nq_lmpf['n']==2048 and o[0].nq_lmpf['last']
        assert sc.log[-1]==(4096,4)                              # single step: no grant
        sc.add(Req('t',500));o=drive(sc)
        assert o[-1].nq_lmpf is None                             # below BUD_MIN (and decode of s next to it)
    finally:cleanup(st)
    sc,st=mkengine(W=32768)
    try:
        sc.add(Req('m',20000));o=drive(sc)
        assert len(o)==1 and o[0].nq_lmpf['m']=='bud' and o[0].nq_lmpf['n']==20000
    finally:cleanup(st)

def test_plain_and_off():
    sc,st=mkengine(W=32768)
    try:
        sc.add(Req('a',20000));sc.requests['a'].has_encoder_inputs=True;o=drive(sc)
        assert all(x.nq_lmpf is None or x.nq_lmpf['m']=='plain' for x in o)
        assert all(x.nq_lmpf is None for x in o if sum(x.num_scheduled_tokens.values())<=4096)
    finally:cleanup(st)
    sc,st=mkengine(W=32768)
    try:
        open(EN.OFF,'w').close()
        sc.add(Req('a',40000));o=drive(sc)
        assert all(x.nq_lmpf is None for x in o) and sc.log[0]==(4096,4)
    finally:
        os.remove(EN.OFF);cleanup(st)

def test_knob_file():
    p=tempfile.mktemp(dir='/dev/shm');k=EN.Knob('NQ_LMPF_TEST_KNOB',7,p,int)
    assert k()==7
    open(p,'w').write('12\n');assert k()==12
    time.sleep(0.01);open(p,'w').write('bad');os.utime(p,ns=(time.time_ns()+10**9,)*2)
    assert k()==7                                                 # unparseable -> default
    os.remove(p);assert k()==7

# ---------------- worker side: pure helpers ----------------
def test_split_sequence():
    assert LP.split(10000,4096)==[(0,4096),(4096,4096),(8192,1808)]
    assert LP.split(4096,4096)==[(0,4096)] and LP.split(1,4096)==[(0,1)]
    s=LP.sequence([0,1,2,3],[2,3],2,'layer')
    assert s==[(0,0,None,True,False),(0,1,None,False,True),(1,0,None,True,False),(1,1,None,False,True),
               (2,0,0,True,False),(2,1,0,False,True),(3,0,1,True,False),(3,1,1,False,True)]
    c=LP.sequence([0,2,3],[2,3],2,'chunk')
    assert [x[2] for x in c]==[None,0,1,None,2,3] and all(x[3] and x[4] for x in c)

def test_select_budget():
    lv=np.full(256,2);lv[:10]=4;pop=np.arange(256.)
    assert LP.full_select(lv,300,pop)==list(range(10,256))
    sel=LP.full_select(lv,5,pop);assert sel==[251,252,253,254,255]
    cnt=np.zeros(256,int);cnt[[0,5,20,30,40]]=[100,90,50,70,10]
    assert LP.bud_select(cnt,lv,2,256)==[30,20] and LP.bud_select(cnt,lv,10,256)==[30,20,40] and LP.bud_select(cnt,lv,0,256)==[]
    assert LP.visits_left(75,0,100000,0,25000)==75*4 and LP.visits_left(75,74,2048,0,2048)==1
    x,a=LP.bud_x(2.0,75,2e9,2.56e6);assert abs(a-2/75)<1e-12 and x==int(2e9*a//2.56e6)
    assert LP.bud_x(-1,10,2e9,2.56e6)[0]==0

def test_size_plan():
    rb=2_560_000;pt=2*6144*2+2048*4;G=2**30
    assert LP.size_plan(10*G,rb,pt,32768,4096,256)==(32768,256)
    W,C=LP.size_plan(1.6*G,rb,pt,32768,4096,256);assert C==256 and 8192<=W<32768 and 2*C*rb+W*pt<=1.6*G
    W,C=LP.size_plan(0.5*G,rb,pt,32768,4096,256);assert 2*C*rb+W*pt<=0.5*G and C>=32
    assert LP.size_plan(0.1*G,rb,pt,32768,4096,256)==(0,16) or LP.size_plan(0.1*G,rb,pt,32768,4096,256) is None
    assert LP.size_plan(0.01*G,rb,pt,32768,4096,256) is None
    W,C=LP.size_plan(0.3*G,rb,pt,32768,4096,256);assert W==0 and 2*C*rb<=0.3*G

def test_snap():
    NT=collections.namedtuple('NT','a b')
    @dataclasses.dataclass
    class D:x:torch.Tensor;y:dict;z:object
    big=torch.zeros(1000);t=torch.ones(3);other=object()
    o=D(x=t,y={'k':t,'l':[t,np.ones(2)],'n':NT(t,5),'big':big,'ns':types.SimpleNamespace(q=t)},z=other)
    sh=[];c=LP.snap(o,limit=1000,shared=sh)
    assert c is not o and c.x is not t and torch.equal(c.x,t)
    assert c.y['k'] is c.x and c.y['l'][0] is c.x and c.y['n'].a is c.x and c.y['ns'].q is c.x   # aliasing kept
    assert c.y['big'] is big and sh==[(1000,)] and c.z is other and isinstance(c.y['n'],NT)
    t.add_(1);assert float(c.x[0])==1.                                   # independent of later in-place writes
    assert c.y['l'][1] is not o.y['l'][1]

# ---------------- ring + LM visit logic on CPU ----------------
class MockEng:
    def __init__(s,fail=()):s.q=[];s.fail=set(fail);s.done=[]
    def upgrade(s,tag,rec,dst,stage,row,seqp,seq):
        assert row.dtype==torch.int64 and row.numel()==20;s.q.append((tag,rec,dst))
    def poll(s):
        r=[(tag,1,-1. if rec in s.fail else 1e-3,1e-3) for tag,rec,dst in s.q];s.done+=s.q;s.q=[];return r
def mkrows(layers):
    rows={}
    for L in layers:
        for E in range(256):
            z=np.zeros(20,np.int64);z[0]=4;z[1]=L*1000+E;m=np.zeros(20,np.int64);m[2]=1;rows[L,E]=(None,z,m)
    return rows
def test_ring():
    dev=torch.device('cpu');L_=[3,4];eng=MockEng(fail={4*256+7})
    R=LP.Ring(eng,64,8,dev,mkrows(L_),lambda L,E:L*256+E)
    base=torch.zeros(256,20,dtype=torch.int64);base[:,0]=2
    R.issue(1,4,[5,6,7]);res,nf,_=R.wait(1)
    assert nf==1 and [E for E,_ in res]==[5,6]
    wt=R.table(1,base,res)
    assert wt[5,0]==4 and wt[5,1]==4005 and wt[5,2]==R.addr(1,0) and wt[6,2]==R.addr(1,1) and wt[7,0]==2 and wt[0,0]==2
    assert R.st['failed']==1 and R.st['recs']==2
    try:R.issue(0,3,list(range(9)));assert False
    except AssertionError as e:assert 'busy' not in str(e)

class FakeM:
    def __init__(s):s.table=torch.zeros(256,20,dtype=torch.int64);s.table[:,0]=2;s.table[:3,0]=4
def mklm(mode,K=1,order='layer'):
    rt=types.SimpleNamespace(rank=0,dev=torch.device('cpu'),lay={L:{'M':FakeM(),'MB':types.SimpleNamespace(apply=lambda:None)} for L in (3,4,5)})
    lm=LP.LM(rt);lm.ring_layers=[3,4,5];lm.vidx={3:0,4:1,5:2};lm.pop={L:np.arange(256.) for L in (3,4,5)}
    lm.ring=LP.Ring(MockEng(),64,256,rt.dev,mkrows([3,4,5]),lambda L,E:L*256+E)
    lm.acc={m:torch.zeros((),dtype=torch.int64) for m in ('full','bud')};lm.tot={'full':0,'bud':0}
    lm.snapt={L:rt.lay[L]['M'].table.clone() for L in (3,4,5)};lm.lv=torch.stack([lm.snapt[L][:,0] for L in (3,4,5)]).numpy()
    lm.begin(dict(m=mode,rid='r',start=0,n=2048,end=2048,budget_s=1.0),K,op=False)
    return lm,rt
def test_lm_full_visits():
    lm,rt=mklm('full',K=2)
    eng=lm.ring.eng;assert len(eng.q)==(2*253 if LP.EARLY else 253)          # visit 0 (+1 when EARLY) reads issued at begin
    ids=torch.randint(0,256,(16,8))
    for L in (3,4,5):
        M=rt.lay[L]['M'];v=lm.vidx[L]
        wt=lm.vstart(v,L,M,ids)
        assert int((wt[:,0]==4).sum())==256 and wt[3,1]==L*1000+3            # every expert at 4-bit, rows of layer L
        assert torch.equal(wt[:3],lm.snapt[L][:3])                           # resident rows untouched
        if LP.EARLY:lm.visit_end(v)                                          # exec: refill issued at the end of the visit
        if v+1<3:assert lm.ring.L[(v+1)%2]==lm.ring_layers[v+1]              # next visit's reads already queued
    lm.end()
def test_lm_bud():
    lm,rt=mklm('bud')
    assert not lm.ring.eng.q
    ids=torch.zeros(4,8,dtype=torch.long);ids[:,0]=10;ids[:2,1]=11;ids[0,2]=1      # 1 = resident: never loaded
    lm.rate=2e9;lm.bud['r']=1.0;lm.nvis=3
    wt=lm.vstart(0,3,rt.lay[3]['M'],ids)
    assert wt[10,0]==4 and wt[11,0]==4 and wt[12,0]==2 and lm.last['bud_recs']>=2
    lm.bud['r']=0.;wt=lm.vstart(1,4,rt.lay[4]['M'],ids)
    assert wt[10,0]==2                                                       # budget spent: nothing loaded

def test_ready_roundtrip():
    k=EN.boot_key();assert k.startswith(str(os.getpid())+'_')
    p=EN.ready_path(k,0);LP.write_json(p,dict(W=8192,C=128,mnbt=4096))
    try:assert json.load(open(p))['W']==8192
    finally:os.remove(p)

if __name__=='__main__':
    fs=[(n,f) for n,f in sorted(globals().items()) if n.startswith('test_') and callable(f)];bad=0
    for n,f in fs:
        try:f();print('ok  ',n)
        except Exception:
            import traceback;bad+=1;print('FAIL',n);traceback.print_exc()
    print(f'{len(fs)-bad}/{len(fs)} passed');sys.exit(1 if bad else 0)
