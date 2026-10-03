# nq-kld: full-vocab prompt-logprob dump for teacher-forced KLD (default off, eval only).
# NQ_KLD_HOOK=1 installs it (nq_vllm.py import, i.e. in every worker). It wraps the logits_fn that
# vLLM's prompt-logprob chunking already calls (model runner V2, vllm/v1/worker/gpu/sample/prompt_logprob.py),
# so it adds no collectives: on global rank 0 only, each chunk's full-vocab logits are kept as fp32 log_softmax.
# Nothing is captured unless the file NQ_KLD_KNOB (default /dev/shm/nq_kld_dump) exists; its content is
# "<tag> [minpos]" and rows for positions < minpos are dropped (long-prompt arms). Each prefill step that computes
# prompt logprobs writes NQ_KLD_DIR/<tag>/<seq>.pt = dict(start=first position, lp=[rows,V] fp32, tgt=target ids).
# Row i of lp is the distribution at position start+i (predicts token start+i+1); the client requests
# prompt_logprobs=1 to trigger the path.
import os,itertools,logging,torch
try:
    from vllm.logger import init_logger;log=init_logger('vllm.nestquant.kld')
except Exception:log=logging.getLogger('nestquant')
KNOB=os.environ.get('NQ_KLD_KNOB','/dev/shm/nq_kld_dump');OUT=os.environ.get('NQ_KLD_DIR','/dbg/kld')
_seq=itertools.count();_start=[None]

def _rank():
    try:
        import torch.distributed as d
        return d.get_rank() if d.is_initialized() else 0
    except Exception:return 0

def _knob():
    try:p=open(KNOB).read().split()
    except OSError:return None
    if not p:return None
    return p[0],(int(p[1]) if len(p)>1 else 0)

def install():
    import vllm.v1.worker.gpu.sample.prompt_logprob as PL
    if getattr(PL,'_nq_kld',False):return
    orig_chunk=PL.compute_prompt_logprobs_with_chunking;orig_cpl=PL.PromptLogprobsWorker.compute_prompt_logprobs
    def cpl(s,logits_fn,hidden_states,input_batch,*a,**k):
        try:_start[0]=int(input_batch.num_computed_prefill_tokens_np[0]) if len(input_batch.req_ids)==1 else None
        except Exception:_start[0]=None
        return orig_cpl(s,logits_fn,hidden_states,input_batch,*a,**k)
    def chunk(prompt_token_ids,prompt_hidden_states,logits_fn,num_prompt_logprobs):
        kb=_knob() if _rank()==0 else None
        if kb is None:return orig_chunk(prompt_token_ids,prompt_hidden_states,logits_fn,num_prompt_logprobs)
        tag,minpos=kb;st=_start[0];buf=[];off=[0]
        def lf(h):
            lg=logits_fn(h);i0=off[0];off[0]+=h.shape[0]
            if st is not None:
                a=max(0,minpos-(st+i0))
                if a<lg.shape[0]:buf.append(torch.log_softmax(lg[a:].float(),-1).cpu())
            return lg
        r=orig_chunk(prompt_token_ids,prompt_hidden_states,lf,num_prompt_logprobs)
        if st is None:log.warning('nq-kld: batch with !=1 request, not dumped');return r
        if buf:
            a=max(0,minpos-st);os.makedirs(f'{OUT}/{tag}',exist_ok=True);n=next(_seq)
            lp=torch.cat(buf);tgt=prompt_token_ids[a:a+lp.shape[0]].cpu()
            torch.save(dict(start=st+a,lp=lp,tgt=tgt),f'{OUT}/{tag}/{n:04d}.pt')
            log.info('nq-kld: dumped %s/%04d start %d rows %d V %d',tag,n,st+a,lp.shape[0],lp.shape[1])
        return r
    PL.compute_prompt_logprobs_with_chunking=chunk;PL.PromptLogprobsWorker.compute_prompt_logprobs=cpl;PL._nq_kld=True
    log.info('nq-kld: prompt-logprob dump hook installed (knob %s, out %s)',KNOB,OUT)

# ---- teacher-forced DECODE (same NQ_KLD_HOOK=1 install) -------------------------------------------------------------
# While the file NQ_KLD_FKNOB (default /dev/shm/nq_kld_force) holds "<tag> <tokens.npy>", every rank overrides the
# target logits of GPUModelRunner.sample so argmax = teacher token[pos+1] (rows with pos+1 < len only). Greedy sampling
# and greedy MTP rejection then commit exactly the teacher sequence: drafts are accepted iff they equal the teacher's
# tokens, so the verify shapes / acceptance / expert streaming are those of real decode on that text. Before the
# override, rank 0 copies the true fp32 log_softmax of every logits row (async, pinned ring) with its position, input
# token and validity (all inputs of the step up to that row == teacher tokens, i.e. the row is a committed position,
# not a rejected draft tail). Removing the knob (next sample call, e.g. a tiny follow-up request) flushes
# NQ_KLD_DIR/<tag>/decode.pt = dict(pos, lp [n,V] fp32 of valid rows sorted by pos, steps, rows, first_mismatch).
FKNOB=os.environ.get('NQ_KLD_FKNOB','/dev/shm/nq_kld_force');RING=int(os.environ.get('NQ_KLD_RING','4096'))
class _F:cur=None;T=None;c=0;steps=0;lp=pos=val=first=mt=None

def _fknob():
    try:p=open(FKNOB).read().split()
    except OSError:return None
    return (p[0],p[1]) if len(p)>=2 else None

def _fstart(tag,path,dev):
    import numpy as np
    _F.cur=tag;_F.T=torch.from_numpy(np.load(path).astype(np.int64)).to(dev);_F.c=0;_F.steps=0
    log.info('nq-kld: forced decode %s (%d teacher tokens) rank %d',tag,_F.T.shape[0],_rank())

def _compact():
    torch.cuda.synchronize();k=_F.val[:_F.c].nonzero().flatten();n=k.numel()
    for b in (_F.lp,_F.pos,_F.val,_F.first,_F.mt):b[:n]=b[k]
    _F.c=n

def _fflush():
    tag=_F.cur;_F.cur=None
    if _rank()!=0 or _F.lp is None:return
    torch.cuda.synchronize();c=_F.c
    k=_F.val[:c].nonzero().flatten();pos=_F.pos[k];o=pos.argsort();k=k[o];pos=pos[o]
    fm=int(((~_F.mt[:c])&_F.first[:c]&(_F.pos[:c]<_F.T.shape[0])).sum())
    os.makedirs(f'{OUT}/{tag}',exist_ok=True)
    torch.save(dict(pos=pos.clone(),lp=_F.lp[k],steps=_F.steps,rows=c,first_mismatch=fm,
                    dup=int(pos.numel()-pos.unique().numel())),f'{OUT}/{tag}/decode.pt')
    log.info('nq-kld: forced decode %s flushed: %d committed rows, %d steps, first-row teacher mismatches %d',tag,pos.numel(),_F.steps,fm)

def _force(lg,ib):
    if ib.num_reqs!=1:log.warning('nq-kld: %d reqs in a forced step, skipped',ib.num_reqs);return
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
        if _F.c+n>RING:log.error('nq-kld: ring full, rows dropped');return
        c=_F.c
        _F.lp[c:c+n].copy_(torch.log_softmax(lg.float(),-1),non_blocking=True)
        _F.pos[c:c+n].copy_(pos,non_blocking=True);_F.val[c:c+n].copy_(val,non_blocking=True);_F.mt[c:c+n].copy_(match,non_blocking=True)
        _F.first[c:c+n]=False;_F.first[c]=True;_F.c+=n
    _F.steps+=1
    g=lg.gather(1,tgt[:,None]).squeeze(1)
    lg.scatter_(1,tgt[:,None],torch.where(inr,torch.full_like(g,1e4),g)[:,None])

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
