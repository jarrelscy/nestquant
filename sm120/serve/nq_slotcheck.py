# nq-probe slot integrity check (debug, default inert). Runs on every rank's streaming thread (nq_vllm.Runtime.loop),
# between iterations, when the trigger file changes. Nothing is checked or touched unless the trigger file is written.
#
# Trigger: write /dev/shm/nq_slotcheck (env NQ_SLOTCHECK_KNOB). Content (first word):
#   all    every resident slot's bytes compared with its record in the store (~rb x residents of SSD reads per rank,
#          ~15 GB per rank with 6000 slots of ~2.5 MB: several seconds of streaming stall, use only when idle)
#   <n>    n random resident slots (default 256) + the full table / bookkeeping checks (cheap)
#   rows   table / bookkeeping checks only, no byte reads
# Every rank runs it once per new trigger mtime and writes /dev/shm/nq_slotcheck_r<rank>.json (tmp + rename).
#
# Checks (per rank, keys = (layer, expert)):
#   B  bytes: slots[slot_of[k]] == record rf.rec(L, E) of the store (pread from this rank's record file), for resident
#      keys that are not in transit (no op in flight, no mailbox apply pending)
#   T  table: M.table[E] == the executor's level-4 row for that slot (z + m * slot address)
#   O  orphans: every level-4 table row of a floating expert points into the slot pool at a slot slot_of says it owns
#      (rows that point elsewhere: fixed pool / LMPF ring / prefill-borrow pool are reported by kind, not as errors
#      unless they point into the normal pool at a slot owned by another key or lent / free)
#   K  bookkeeping: slot_of injective; no resident slot in free or lent; free has no duplicates; free and lent disjoint
#   S  leader only: S.state == 2 <=> key resident and landed (not in transit); followers: F.up == landed set
# Mismatches on the table are re-read once after 50 ms (a captured mailbox apply may be mid-copy); only persistent
# mismatches are reported. Nothing is repaired.
import os,json,time,random,contextlib,numpy as np,torch
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant.slotcheck')
except Exception:
    import logging;log=logging.getLogger('nestquant')
KNOB=os.environ.get('NQ_SLOTCHECK_KNOB','/dev/shm/nq_slotcheck')
OUT=os.environ.get('NQ_SLOTCHECK_OUT','/dev/shm/nq_slotcheck')

class SlotCheck:
    def __init__(s):
        s.m=None;s.n=0;s.buf=None;s.dbuf=None;s.st=None
        try:s.m=os.stat(KNOB).st_mtime_ns         # a trigger that predates the boot is ignored
        except OSError:s.m=None
    def due(s):
        try:m=os.stat(KNOB).st_mtime_ns
        except OSError:return False
        return m!=s.m
    def run(s,rt,cap):
        """called at the top of a streaming-loop iteration (in_iter set, nothing else touches X / S / F).
        -> True when it ran (the caller skips the rest of the iteration)"""
        if cap:return False                       # vLLM graph capture in progress: no CUDA calls from this thread
        try:m=os.stat(KNOB).st_mtime_ns;arg=(open(KNOB).read().split() or ['256'])[0]
        except OSError:return False
        s.m=m;s.n+=1;t0=time.time()
        try:r=s._check(rt,arg)
        except Exception as e:
            log.exception('NestQuant slotcheck rank %d failed',rt.rank);r=dict(error=repr(e))
        r.update(rank=rt.rank,trigger=arg,run=s.n,t_wall=time.time(),secs=round(time.time()-t0,3))
        p=f'{OUT}_r{rt.rank}.json';tmp=p+'.tmp'
        with open(tmp,'w') as f:json.dump(r,f)
        os.replace(tmp,p)
        lvl=log.error if r.get('bad') else log.info
        lvl('NestQuant slotcheck rank %d: %s',rt.rank,{k:r[k] for k in ('bad','checked_bytes','bytes_bad','table_bad','orphans','book_bad','state_bad','secs') if k in r})
        return True
    def _check(s,rt,arg):
        X=rt.X;S=rt.S;F=rt.F;lead=F is None;rb=X.rb;ns=X.nslot;slot0=X.slot0
        dev=X.slots.device
        cuda=dev.type=='cuda'
        if s.st is None and cuda:s.st=torch.cuda.Stream(device=dev)
        sctx=(lambda:torch.cuda.stream(s.st)) if cuda else contextlib.nullcontext
        ssync=(lambda:s.st.synchronize()) if cuda else (lambda:None)
        # ---- keys in transit (skipped by B / T)
        trans=set((o[0],o[1]) for o in list(X.ops.values()))|set(X.wait_apply)
        res={k:sl for k,sl in X.slot_of.items()}
        normal={k:sl for k,sl in res.items() if sl<ns}
        landed={k:sl for k,sl in normal.items() if k not in trans}
        out=dict(nslot=ns,resident=len(normal),xpool=len(res)-len(normal),in_transit=len(trans),free=len(X.free),lent=len(X.lent),
                 pend=len(X.pend),lmpf_borrow_held=bool(getattr(getattr(rt,'LM',None),'bwi',None)),
                 lmpf_active=bool(getattr(getattr(rt,'LM',None),'active',False)),xep=int(getattr(X,'xep',0)),
                 oplog=(int(rt.log.gen),int(rt.log.off)) if rt.log is not None else None)
        # ---- K bookkeeping
        book=[]
        inv={}
        for k,sl in normal.items():
            if sl in inv:book.append(f'slot {sl} held by {inv[sl]} and {k}')
            inv[sl]=k
        fr=list(X.free);frs=set(fr)
        if len(fr)!=len(frs):book.append(f'free list has {len(fr)-len(frs)} duplicates')
        for sl in frs&set(inv):book.append(f'slot {sl} in free but held by {inv[sl]}')
        for sl in X.lent&set(inv):book.append(f'slot {sl} lent but held by {inv[sl]}')
        if frs&X.lent:book.append(f'{len(frs&X.lent)} slots both free and lent')
        acc=len(frs)+len(X.lent)+len(inv)
        if acc!=ns and not X.lent:book.append(f'slot accounting free {len(frs)} + lent {len(X.lent)} + held {len(inv)} != nslot {ns}')
        out['book_bad']=len(book);out['book']=book[:20];out['slot_accounting']=[len(frs),len(X.lent),len(inv),ns]
        # ---- T table rows + O orphans (per layer, one D2H copy of the table)
        def tables():
            with sctx():tb={L:rt.lay[L]['M'].table.to('cpu',non_blocking=False).numpy().copy() for L in X.layers}
            ssync();return tb
        tb=tables();tbad=[];orph=[];kinds={}
        def addr_of(L,E,row):
            r2,z,m=X.rc[L,E];i=np.nonzero(m)[0]
            if not len(i):return None
            j=i[0];d=int(row[j])-int(z[j])
            return d//int(m[j]) if int(m[j]) else None
        def scan(tb):
            tb_=[];or_=[];kd={}
            for L in X.layers:
                t=tb[L];lv=t[:,0]
                for E in np.nonzero(lv==4)[0].tolist():
                    k=(L,E)
                    if k in landed:
                        want=X._row(L,E,4,landed[k]).numpy()
                        if not np.array_equal(t[E],want):tb_.append((L,E,landed[k]))
                        continue
                    if k in trans:kd['transit']=kd.get('transit',0)+1;continue
                    a=addr_of(L,E,t[E])
                    if a is None:kd['noaddr']=kd.get('noaddr',0)+1;continue
                    if slot0<=a<slot0+ns*rb:
                        sl=(a-slot0)//rb;why='lent' if sl in X.lent else ('free' if sl in frs else f'owned by {inv.get(sl)}')
                        or_.append((L,E,int(sl),why))
                    elif k in res:kd['xpool']=kd.get('xpool',0)+1
                    elif S is not None and S.fixed[S.li[L],E]:kd['fixed']=kd.get('fixed',0)+1
                    else:kd['elsewhere']=kd.get('elsewhere',0)+1      # LMPF ring rows never reach M.table; anything here is odd
                # landed keys whose table row is NOT level 4
                for k,sl in landed.items():
                    if k[0]==L and lv[k[1]]!=4:tb_.append((L,k[1],sl))
            return tb_,or_,kd
        tbad,orph,kinds=scan(tb)
        if tbad or orph:                           # re-read once: a captured apply may have been mid-copy
            time.sleep(0.05);tb=tables();t2,o2,kinds=scan(tb)
            tbad=[x for x in tbad if x in set(t2)];orph=[x for x in orph if x in set(o2)]
        out['table_bad']=len(tbad);out['table_bad_ex']=[list(map(int,x)) for x in tbad[:20]]
        out['orphans']=len(orph);out['orphans_ex']=[[int(a),int(b),int(c),d] for a,b,c,d in orph[:20]];out['level4_other']=kinds
        # ---- S scheduler / follower view vs landed
        sb=[]
        if lead:
            st=S.state
            for k in landed:
                if st[S.li[k[0]],k[1]]!=2:sb.append(('landed_not_state2',k,int(st[S.li[k[0]],k[1]])))
            for i,e in zip(*np.nonzero((st==2)&~S.fixed)):
                k=(S.layers[i],int(e))
                if k not in normal and k not in res:sb.append(('state2_not_resident',k,2))
        elif hasattr(F,'up'):
            up=set(F.up) if not callable(getattr(F,'up',None)) else set()
            for k in set(landed)-up:sb.append(('landed_not_F.up',k,0))
            for k in up-set(res):sb.append(('F.up_not_resident',k,0))
        out['state_bad']=len(sb);out['state_bad_ex']=[[a,list(map(int,b)),c] for a,b,c in sb[:20]]
        # ---- B bytes
        keys=sorted(landed)
        if arg=='rows':pick=[]
        elif arg=='all':pick=keys
        else:
            try:n=int(arg)
            except ValueError:n=256
            pick=random.sample(keys,min(n,len(keys)))
        bb=[];nb=0
        if pick:
            if s.buf is None:s.buf=torch.empty(rb,dtype=torch.uint8);s.buf=s.buf.pin_memory() if cuda else s.buf;s.dbuf=torch.empty(rb,dtype=torch.uint8,device=dev)
            fd=os.open(X.rf.path,os.O_RDONLY);mv=memoryview(s.buf.numpy())
            try:
                for k in pick:
                    sl=landed[k]
                    if k in X.ops or k in X.wait_apply or X.slot_of.get(k)!=sl:continue
                    n=os.preadv(fd,[mv],X.rf.rec(*k)*rb)
                    if n!=rb:bb.append((k,sl,-1));continue
                    with sctx():
                        s.dbuf.copy_(s.buf,non_blocking=False)
                        d=int((X.slots[sl]!=s.dbuf).sum())
                    nb+=1
                    if d:bb.append((k,sl,d))
            finally:os.close(fd)
            ssync()
        out['checked_bytes']=nb;out['bytes_bad']=len(bb);out['bytes_bad_ex']=[[list(map(int,k)),int(sl),int(d)] for k,sl,d in bb[:20]]
        out['bad']=bool(out['book_bad'] or out['table_bad'] or out['orphans'] or out['state_bad'] or out['bytes_bad'])
        out['landed_hash']=hash(tuple(keys))&0xffffffffffff;out['landed_n']=len(keys)
        return out

SC=SlotCheck() if os.environ.get('NQ_SLOTCHECK','1')!='0' else None
