import numpy as np,sys
NS=4096
for tag in sys.argv[2].split(','):
  res=[]
  for r in range(4):
    R=np.load(f'/data/Jarrel/nq-serve/dbg/pgprobe_{sys.argv[1]}_{tag}_r{r}.npz');off=int(R['off']);ts=R['ts'];h=R['hlog']
    cur=int(R['hdr'][0]);m=(h[:,0]>=cur-150)&(h[:,0]<cur-2)&(h[:,2]>0)
    lags=[];rx=[]
    for s,L,ta,tb,tc,n in h[m]:
      row=ts[s%NS,:,0].astype(np.int64);done=tc+off
      trig=ts[s%NS,L-1,0];rx.append((ta+off-trig)/1e3)
      seq=[(s,l) for l in range(L,78)]+[(s+1,l) for l in range(3,78)]
      k=next((i for i,(ss,l) in enumerate(seq) if ts[ss%NS,l,0]>=done),len(seq));lags.append(k)
    lags=np.array(lags);res.append((np.median(rx),np.percentile(lags,[50,90]),(lags==0).mean()))
  print(tag,' '.join(f'r{i}: react {a:.0f}us lag p50/p90 {b[0]:.0f}/{b[1]:.0f} layers' for i,(a,b,c) in enumerate(res)))
