"""Committed-token jT inference and the unmodified release floating-set policy.

The predictor has its own CUDA stream and a bounded per-layer sliding KV cache.
Only verified target rows enter this module; speculative rows stay in RowLedger.
"""
import contextlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


class IncrementalJT:
    def __init__(self,model,rope):
        self.model=model;self.rope=rope
        dev=next(model.parameters()).device
        self.cuda_stream=torch.cuda.Stream(device=dev) if dev.type=='cuda' else None
        if self.cuda_stream is not None:self.cuda_stream.wait_stream(torch.cuda.current_stream(dev))
        self.reset()

    def stream(self):
        return torch.cuda.stream(self.cuda_stream) if self.cuda_stream is not None else contextlib.nullcontext()

    def reset(self):
        # All inference/reset calls must be serialized by the owning scheduler.
        with self.stream():self.cache=[None]*len(self.model.blocks);self.position=0

    @torch.inference_mode()
    def predict(self,tokens,prev_ids,prev_q):
        with self.stream():return self._predict(tokens,prev_ids,prev_q)

    def _predict(self,tokens,prev_ids,prev_q):
        m=self.model;dev=next(m.parameters()).device
        tok=torch.as_tensor(tokens,dtype=torch.long,device=dev)[None]
        ids=torch.as_tensor(prev_ids,dtype=torch.long,device=dev)[None]
        q=torch.as_tensor(prev_q,dtype=m.re.weight.dtype,device=dev)[None]
        _,T,NL,topk=ids.shape;NE=m.bhead.out_features//NL
        if T==0:raise ValueError('Empty predictor batch')
        off=torch.arange(NL,device=dev)[:,None]*NE
        rout=m.re((ids+off).reshape(T,-1),per_sample_weights=q.reshape(T,-1)).view(1,T,-1)
        x=m.tp(m.tok(tok))+m.rn(rout)
        pos=torch.arange(self.position,self.position+T,device=dev)[None]
        for i,b in enumerate(m.blocks):
            B,_,D=x.shape
            q,k,v=b.qkv(b.l1(x)).view(B,T,3,b.nh,D//b.nh).permute(2,0,3,1,4)
            q,k=self.rope(q,pos),self.rope(k,pos)
            old=self.cache[i];old_n=0 if old is None else old[0].shape[-2]
            if old is not None:k=torch.cat((old[0],k),-2);v=torch.cat((old[1],v),-2)
            kp=torch.arange(self.position-old_n,self.position+T,device=dev)
            mask=(pos[0,:,None]>=kp)&(pos[0,:,None]-kp<m.win)
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[None,None])
            x=x+b.o(a.transpose(1,2).reshape(B,T,D))
            x=x+b.f2(F.gelu(b.f1(b.l2(x))))
            keep=m.win-1
            self.cache[i]=(k[...,-keep:,:].contiguous(),v[...,-keep:,:].contiguous()) if keep else None
        self.position+=T
        return m.bhead(m.ln(x)).view(T,NL,NE).float().softmax(-1).cpu().numpy()


class RowLedger:
    """Keep verify rows provisional until the next target start position proves acceptance.

    Example: pending positions [p,p+1]; next target start p+1 commits only p
    (draft rejected), whereas start p+2 commits both. New requests discard pending
    rows and reset all predictor state. Prefix-cache restores need saved histories.
    """
    def __init__(self):self.pending=[];self.committed_end=0
    def reset(self):self.pending=[];self.committed_end=0
    def stage(self,positions,tokens,ids,weights,xn):
        if self.pending:raise RuntimeError('Resolve pending verify rows before staging more')
        pos=np.asarray(positions)
        if len(pos) and not np.array_equal(pos,np.arange(self.committed_end,self.committed_end+len(pos))):
            raise ValueError('Non-contiguous target rows or missing prefix history')
        if not all(len(x)==len(pos) for x in (tokens,ids,weights,xn)):
            raise ValueError('Mismatched routing rows')
        self.pending=[(int(p),int(t),np.array(i,copy=True),np.array(w,copy=True),np.array(n,copy=True))
                      for p,t,i,w,n in zip(pos,tokens,ids,weights,xn)]
    def accept_before(self,next_position):
        if not self.committed_end<=next_position<=self.committed_end+len(self.pending):
            raise ValueError('Invalid committed position; preemption/prefix restore needs predictor history')
        n=next_position-self.committed_end
        result=self.pending[:n];self.pending=[];self.committed_end=next_position
        return result


def layer_budgets(meta,preset):
    """Explicit serving presets; do not alter published research allocations."""
    if preset in ('spark_128K_U_2352','spark_256K_U_2504','spark_256K_U_2630','spark_256K_U_2630_flat50'):
        total={'spark_128K_U_2352':2352,'spark_256K_U_2504':2504,'spark_256K_U_2630':2630,'spark_256K_U_2630_flat50':2630}[preset]
        filename=('u_distribution_2630_flat50.json' if preset=='spark_256K_U_2630_flat50' else ('u_distribution.json' if total==2352 else f'u_distribution_{total}.json'))
        config=json.loads((Path(__file__).with_name(filename)).read_text())
        budgets={int(L):int(n) for L,n in config['budgets'].items()}
        if sum(budgets.values())!=total or config['active_slots']!=total:raise ValueError('Wrong U-distribution slot total')
    else:
        allocation=({'3-17':74,'18-44':46} if preset=='spark_128K_74_46'
                    else meta['n_float'][preset])
        budgets={}
        for span,n in allocation.items():
            lo,hi=map(int,span.split('-'))
            for L in range(lo,hi+1):
                if L in budgets:raise ValueError('Overlapping predictor allocation')
                budgets[L]=int(n)
    if set(budgets)!=set(range(3,45)) or any(not 1<=n<=288 for n in budgets.values()):
        raise ValueError('Invalid predictor layer allocation')
    return budgets


def fixed_counts_for_fraction(budgets, fraction):
    """Same fraction per layer, largest-remainder rounding at fixed total."""
    from fractions import Fraction
    f=Fraction(str(fraction))
    if not 0<=f<1:raise ValueError('Fixed fraction must be in [0,1)')
    raw={L:n*f for L,n in budgets.items()};counts={L:int(v) for L,v in raw.items()}
    total=int(sum(budgets.values())*f+Fraction(1,2))
    for L in sorted(raw,key=lambda L:(-(raw[L]-counts[L]),L))[:total-sum(counts.values())]:counts[L]+=1
    if any(counts[L]>=budgets[L] for L in budgets):raise ValueError('Fixed share must leave floating slots')
    return counts


class CommittedPredictor:
    def __init__(self,root,preset='spark_128K',device='cuda',dtype=torch.float16,n_fixed=0,fixed_fraction=None):
        root=Path(root);pd=root/'serving/predictor'
        meta=json.loads((pd/'predictor.json').read_text())['params']
        required=dict(n_fixed=0,refresh_tokens=8,hm=4.0,mix_block=0.5,
                      ema_half_life_tokens=64,block_horizon=16,NE=288,top_k=8)
        for k,v in required.items():
            if meta.get(k)!=v:raise ValueError(f'Unsupported jT policy {k}: {meta.get(k)}')
        budgets=layer_budgets(meta,preset)
        self.layers=list(range(3,45));self.budgets=budgets
        counts=json.loads((root/'fixed_set.json').read_text())['n_routed']
        self.defaults=[np.argsort(-np.asarray(counts[str(L)]),kind='stable')[:budgets[L]] for L in self.layers]
        if not isinstance(n_fixed,int) or not 0<=n_fixed<min(budgets.values()):raise ValueError('Invalid fixed count')
        if fixed_fraction is not None and n_fixed:raise ValueError('Choose fixed count or fraction, not both')
        self.fixed_counts=(fixed_counts_for_fraction(budgets,fixed_fraction) if fixed_fraction is not None else {L:n_fixed for L in self.layers})
        self.n_fixed=sum(self.fixed_counts.values())
        self.fixed_ids=[d[:self.fixed_counts[L]].copy() for L,d in zip(self.layers,self.defaults)]
        jt=module(pd/'jt/jt_model.py','nq_flash_release_jt')
        self.policy_cls=module(pd/'jt/policy.py','nq_flash_release_policy').FloatingSet
        self.net=IncrementalJT(jt.load(pd/'jt',device=device,dtype=dtype),jt.rope)
        self.reset()

    def reset(self):
        self.net.reset();self.t=0;self.prepared_t=-1;self.block=None
        self.previous_ids=np.zeros((42,8),np.int64);self.previous_q=np.zeros((42,8),np.float32)
        if getattr(self,'n_fixed',0):
            from fixed_policy import FixedSetPolicy
            self.policies=[FixedSetPolicy(self.policy_cls,self.budgets[L],d,f) for L,d,f in zip(self.layers,self.defaults,self.fixed_ids)]
        else:
            self.policies=[self.policy_cls(self.budgets[L],d) for L,d in zip(self.layers,self.defaults)]
        self.prepare()

    def prepare(self):
        if self.prepared_t!=self.t:
            for i,p in enumerate(self.policies):p.before_row(None if self.block is None else self.block[i])
            self.prepared_t=self.t
        return np.stack([p.cur for p in self.policies])

    def loading_priority(self):
        """Rank delivery within the chosen pool using committed jT/EMA scores.

        Does not apply a second hysteresis or change FloatingSet membership.
        """
        ema=np.stack([p.state for p in self.policies])
        shares=ema/np.maximum(ema.sum(-1,keepdims=True),1e-30)
        block=np.zeros_like(shares) if self.block is None else self.block
        return .5*shares+.5*block

    def commit(self,rows):
        # Bound predictor temporaries independently of prefill length.
        for start in range(0,len(rows),128):
            chunk=rows[start:start+128]
            tok=[];prev_ids=[];prev_q=[]
            for pos,token,ids,w,xn in chunk:
                tok.append(token);prev_ids.append(self.previous_ids);prev_q.append(self.previous_q)
                q=np.square(w.astype(np.float32));q/=np.maximum(q.sum(-1,keepdims=True),1e-30)
                self.previous_ids=ids.copy();self.previous_q=q
            blocks=self.net.predict(tok,np.asarray(prev_ids),np.asarray(prev_q))
            for row,block in zip(chunk,blocks):
                pos,token,ids,w,xn=row
                if pos!=self.t:raise ValueError(f'Predictor row {pos} != committed clock {self.t}')
                self.prepare()
                for i,p in enumerate(self.policies):p.after_row(ids[i],w[i],xn[i])
                self.block=block;self.t+=1
        return self.prepare()
