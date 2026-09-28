"""Repack the production shards (nestquant-v1, L{L}/tp{s}.pt) into one P4 record file per TP rank (work item B1).
  repack.py ROOT OUT [tp=4] [layers=3-77] [rec_bytes]
OUT/rank{r}.bin: record of (L, E) at ((L - L0) * 256 + E) * rec_bytes, rec_bytes the same for every layer (4 KiB
aligned, = the slot size), so layers can be written in any order as the fit lands them. OUT/rank{r}.json: layout,
per-layer written flag and per-expert lr ranks. The base planes are not in the file (they load at startup).
Layers already in the index are skipped; rerun to pick up new ones."""
import os,sys,json,time,torch
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,HERE+'/../sm120']
import nqload as NQ,p4rec as PR
NE=256;L0=3

def parse_layers(s):
    a,b=(s.split('-')+[s])[:2];return list(range(int(a),int(b)+1))

def main():
    root,out=sys.argv[1],sys.argv[2];tp=int(sys.argv[3]) if len(sys.argv)>3 else 4
    layers=parse_layers(sys.argv[4]) if len(sys.argv)>4 else list(range(3,78))
    os.makedirs(out,exist_ok=True)
    for L in layers:
        if not os.path.exists(f'{root}/L{L}/manifest.json'):print(f'L{L}: not fitted yet');continue
        for r in range(tp):
            ip=f'{out}/rank{r}.json';idx=json.load(open(ip)) if os.path.exists(ip) else dict(format='nq-p4rec-v1',tp=tp,rank=r,L0=L0,NE=NE,layers={})
            if str(L) in idx['layers']:continue
            t=time.time();RL=NQ.RankLayer(root,L,r,tp)
            lay=PR.layout(next(iter(RL.ex.values())),RL.H,RL.I)
            if 'rec_bytes' not in idx:idx['rec_bytes']=int(sys.argv[5]) if len(sys.argv)>5 else lay['rec_bytes'];idx['seg']=lay['seg']
            assert lay['seg']==idx['seg'] and lay['rec_bytes']<=idx['rec_bytes'],('layout changed',L,lay,idx['seg'])
            lay['rec_bytes']=idx['rec_bytes'];rb=lay['rec_bytes']
            fd=os.open(f'{out}/rank{r}.bin',os.O_WRONLY|os.O_CREAT,0o644)
            try:
                for E,ex in RL.ex.items():os.pwrite(fd,PR.pack(ex,lay),((L-L0)*NE+E)*rb)
                os.fsync(fd)
            finally:os.close(fd)
            idx['layers'][str(L)]=dict(experts=RL.experts,rg={E:RL.ex[E].rg for E in RL.experts},rd={E:RL.ex[E].rd for E in RL.experts})
            json.dump(idx,open(ip+'.tmp','w'));os.replace(ip+'.tmp',ip)
            print(f'L{L} rank{r}: {len(RL.experts)} records x {rb} B in {time.time()-t:.1f}s',flush=True)
            del RL;torch.cuda.empty_cache()

if __name__=='__main__':main()
