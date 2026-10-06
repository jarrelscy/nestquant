import numpy as np,sys,json
NS=4096
pre=sys.argv[1];tags=sys.argv[2].split(',')
cfg=dict(off=(0,0,1),ack=(0,0,1),b1s8=(0,1,8),a1s8=(1,1,8),b2s8=(0,2,8),b4s8=(0,4,8),b1s1=(0,1,1),b2s1=(0,2,1),off2=(0,0,1))
LS=np.arange(3,78)
out={}
for tag in tags:
    pt,n,stride=cfg[tag]
    R=[np.load(f'/data/Jarrel/nq-serve/dbg/pgprobe_{pre}_{tag}_r{r}.npz') for r in range(4)]
    cur=int(R[0]['hdr'][0]);ss=np.arange(cur-150,cur-2)
    ok=np.ones(len(ss),bool);land=[];mo=[];la=[];lb=[];st=[];hr=[];hc=[]
    for Rr in R:
        ts=Rr['ts'][ss%NS]  # [S,96,4]
        T=ts[:,3,3];ok&=(T==4)
    for Rr in R:
        ts=Rr['ts'][ss%NS][ok]
        t0=ts[:,LS,0].astype(np.float64);t1=ts[:,LS,1].astype(np.float64)
        mo.append(np.median(t1-t0)/1e3)
        lb.append(np.median(t0[:,1:]-t0[:,:-1])/1e3);la.append(np.median(t0[:,1:]-t1[:,:-1])/1e3)
        st.append(np.median(t0[1:,0]-t0[:-1,0])/1e6)
        tl=[L for L in range(4,78) if L%stride==0]
        sv=ss[ok]
        land.append((ts[:,tl,2]>=sv[:,None]))
        h=Rr['hlog'];m=(h[:,5]==n)&(h[:,0]>=ss[0])&(h[:,0]<=ss[-1])&(h[:,2]>0)
        hr.append(np.median((h[m,3]-h[m,2]))/1e3 if m.sum() else np.nan);hc.append(np.median((h[m,4]-h[m,3]))/1e3 if m.sum() else np.nan)
        # reaction: host seen time vs gpu trigger time
    L_all=np.logical_and.reduce(land)
    o=dict(steps=int(ok.sum()),moe_us=np.mean(mo),lead_b_us=np.mean(lb),lead_a_us=np.mean(la),step_ms=np.mean(st),
           landed_all4=float(L_all.mean()),landed_per_rank=[float(x.mean()) for x in land],read_us=hr,copy_us=hc)
    out[tag]=o;print(tag,json.dumps(o,default=float))
json.dump(out,open(f'ana_{pre}.json','w'),default=float)
