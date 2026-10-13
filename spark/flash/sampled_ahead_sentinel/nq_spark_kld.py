import os, logging, torch, numpy as np, json, time
log=logging.getLogger("nestquant.kld")
OUT=os.environ.get("NQ_KLD_DIR","/data/kld-spark")
def _rank():return 0
# ---- teacher-forced DECODE (same NQ_KLD_HOOK=1 install) -------------------------------------------------------------
# While the file NQ_KLD_FKNOB (default /dev/shm/nq_kld_force) holds "<tag> <tokens.npy>", every rank overrides the
# target logits of GPUModelRunner.sample so argmax = teacher token[pos+1] (rows with pos+1 < len only). Greedy sampling
# and greedy MTP rejection then commit exactly the teacher sequence: drafts are accepted iff they equal the teacher's
# tokens, so the verify shapes / acceptance / expert streaming are those of real decode on that text. Before the
# override, rank 0 copies the true fp32 log_softmax of every logits row (async, pinned ring) with its position, input
# token and validity (all inputs of the step up to that row == teacher tokens, i.e. the row is a committed position,
# not a rejected draft tail). Removing the knob (next sample call, e.g. a tiny follow-up request) flushes
# NQ_KLD_DIR/<tag>/decode.pt = dict(pos, lp [n,V] fp32 of valid rows sorted by pos, steps, rows, first_mismatch).
FKNOB=os.environ.get('NQ_KLD_FKNOB','/dev/shm/nq_kld_force');RING=int(os.environ.get('NQ_KLD_RING','64'))
class _F:cur=None;T=None;c=0;steps=0;lp=pos=val=first=mt=None

def _fknob():
    try:p=open(FKNOB).read().split()
    except OSError:return None
    return (p[0],p[1]) if len(p)>=2 else None

def _fstart(tag,path,dev):
    import numpy as np
    _F.cur=tag;_F.T=torch.from_numpy(np.load(path).astype(np.int64)).to(dev);_F.c=0;_F.steps=0
    _F.started=time.perf_counter();_F.drain_seconds=0.;_F.force_seconds=0.;_F.total_rows=0;_F.mismatch=0;_F.V=154880
    _F.counts=np.zeros(len(_F.T)-1,dtype=np.int32)
    os.makedirs(f'{OUT}/{tag}',exist_ok=True);_F.file=open(f'{OUT}/{tag}/decode.f32','wb')
    _F.file.truncate((len(_F.T)-1)*_F.V*4)
    log.info('nq-kld: forced decode %s (%d teacher tokens) rank %d',tag,_F.T.shape[0],_rank())

def _compact():
    t=time.perf_counter();torch.cuda.synchronize();c=_F.c
    if c:
        pos=_F.pos[:c].numpy();valid=_F.val[:c].numpy()
        _F.mismatch+=int(((~_F.mt[:c])&_F.first[:c]&(_F.pos[:c]<_F.T.shape[0])).sum())
        for j in np.flatnonzero(valid):
            p=int(pos[j]); _F.counts[p]+=1
            _F.file.seek(p*_F.V*4);_F.file.write(_F.lp[j].numpy().tobytes())
        _F.total_rows+=c
    _F.c=0;_F.drain_seconds+=time.perf_counter()-t

def _fflush():
    if _F.cur is None:return
    tag=_F.cur
    if _F.lp is not None:_compact()
    _F.file.flush();_F.file.close()
    counts=_F.counts
    record=dict(tag=tag,positions=int((counts>0).sum()),expected_positions=len(counts),
        missing=np.flatnonzero(counts==0).tolist(),dup=int(np.maximum(counts-1,0).sum()),
        first_mismatch=_F.mismatch,steps=_F.steps,rows=_F.total_rows,vocab=_F.V,
        drain_seconds=_F.drain_seconds,force_seconds=_F.force_seconds,
        elapsed_seconds=time.perf_counter()-_F.started,format='float32_log_softmax_rows_by_prediction_position',
        ring_rows=RING,measurement='teacher_forced_decode_with_MTP2_capture_overhead')
    with open(f'{OUT}/{tag}/decode.json','w') as f:json.dump(record,f,indent=2)
    _F.cur=None
    log.info('nq-kld: completed %s %s',tag,record)

def _force(lg,ib):
    begin=time.perf_counter()
    if lg.shape[1]!=_F.V:raise RuntimeError("KLD vocabulary mismatch")
    if ib.num_reqs!=1:raise RuntimeError('KLD requires exactly one request')
    T=_F.T;Lt=T.shape[0];li=ib.logits_indices;n=lg.shape[0]
    pos=ib.positions[li].long();inp=ib.input_ids[li].long()
    inr=(pos+1)<Lt;cur=T[pos.clamp(0,Lt-1)];tgt=T[(pos+1).clamp(0,Lt-1)]
    match=(inp==cur)&(pos<Lt);val=torch.cumprod(match.int(),0).bool()&inr
    if _rank()==0:
        if _F.lp is None:
            V=lg.shape[1];_F.lp=torch.empty(RING,V,dtype=torch.float32,pin_memory=True)
            _F.pos=torch.empty(RING,dtype=torch.int64,pin_memory=True);_F.val=torch.empty(RING,dtype=torch.bool,pin_memory=True)
            _F.first=torch.zeros(RING,dtype=torch.bool,pin_memory=True);_F.mt=torch.empty(RING,dtype=torch.bool,pin_memory=True)
            log.info('nq-kld: pinned ring %d x %d fp32',RING,V)
        if _F.c+n>RING:_compact()
        if _F.c+n>RING:raise RuntimeError('KLD ring capacity exceeded')
        c=_F.c
        _F.lp[c:c+n].copy_(torch.log_softmax(lg.float(),-1),non_blocking=True)
        _F.pos[c:c+n].copy_(pos,non_blocking=True);_F.val[c:c+n].copy_(val,non_blocking=True);_F.mt[c:c+n].copy_(match,non_blocking=True)
        _F.first[c:c+n]=False;_F.first[c]=True;_F.c+=n
    _F.steps+=1
    g=lg.gather(1,tgt[:,None]).squeeze(1)
    lg.scatter_(1,tgt[:,None],torch.where(inr,torch.full_like(g,1e4),g)[:,None])
    _F.force_seconds+=time.perf_counter()-begin

def install_force():
    import vllm.v1.worker.gpu.model_runner as MR
    R=MR.GPUModelRunner
    if getattr(R,'_nq_kld_force',False):return
    orig=R.sample
    def sample(s,hidden_states,input_batch,grammar_output):
        kb=_fknob()
        if kb is None:
            if _F.cur is not None:_fflush()
            return orig(s,hidden_states,input_batch,grammar_output)
        if _F.cur!=kb[0]:
            if _F.cur is not None:_fflush()
            _fstart(kb[0],kb[1],hidden_states.device)
        m=s.model;had='compute_logits' in m.__dict__;ocl=m.compute_logits
        def cl(h):
            lg=ocl(h);_force(lg,input_batch);return lg
        m.compute_logits=cl
        try:return orig(s,hidden_states,input_batch,grammar_output)
        finally:
            if had:m.compute_logits=ocl
            else:del m.compute_logits
    R.sample=sample;R._nq_kld_force=True
    log.info('nq-kld: forced-decode hook installed (knob %s)',FKNOB)
