# KL(ref||live) per context (copy of nq-kld/reg/live/score.py): ref = hidden_states.float() @ lm_head.float().T, float64, full vocab
import sys,json,torch
from safetensors import safe_open
R='/data/Jarrel/nq-kld/reg/root';DBG='/data/Jarrel/nq-serve/dbg/kld';O='/data/Jarrel/nq-lalloc';TAG=sys.argv[1]
torch.set_num_threads(24)
with safe_open(f'{R}/head/weight.safetensors','pt') as f:W=f.get_tensor('lm_head.weight').float()
for c in range(25):
    D=torch.load(f'{DBG}/{TAG}_c{c:02d}/decode.pt');pos=D['pos'];N=2047
    chk=dict(rows=int(pos.numel()),dup=D['dup'],first_mismatch=D['first_mismatch'],steps=D['steps'],pos_ok=bool(torch.equal(pos,torch.arange(N))))
    with safe_open(f'{R}/capture/hidden_{c:04d}.safetensors','pt') as f:H=f.get_tensor('hidden_states').float()
    kl=torch.empty(N,dtype=torch.float64);ag=0;t1=torch.empty(N,dtype=torch.bool)
    for i in range(0,N,256):
        j=min(N,i+256);lt=torch.log_softmax((H[i:j]@W.T).double(),-1)
        ls=D['lp'][i:j];assert ls.shape[1]>=W.shape[0],ls.shape
        ls=torch.log_softmax(ls[:,:W.shape[0]].double(),-1)
        kl[i:j]=(lt.exp()*(lt-ls)).sum(-1);t1[i:j]=lt.argmax(-1)==ls.argmax(-1)
    out=dict(tag=TAG,ctx=c,kld=kl.mean().item(),top1=t1.float().mean().item(),emitted_per_step=round(N/D['steps'],3),V=int(D['lp'].shape[1]),**chk)
    torch.save(dict(kl=kl,top1=t1),f'{O}/private/{TAG}_c{c:02d}.pt');del D
    print(json.dumps(out),flush=True);open(f'{O}/scores.jsonl','a').write(json.dumps(out)+'\n')
