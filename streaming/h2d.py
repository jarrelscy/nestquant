"""Pinned host -> device copy bandwidth, one process per GPU, all concurrent. usage: h2d.py GPUS SECS [MB]"""
import torch,torch.multiprocessing as mp,sys,time
def run(g,secs,mb,q,bar):
    torch.cuda.set_device(g);n=mb*2**20
    h=torch.empty(n,dtype=torch.uint8).pin_memory();d=torch.empty(n,dtype=torch.uint8,device='cuda')
    for _ in range(3):d.copy_(h,non_blocking=True)
    torch.cuda.synchronize();bar.wait();t=time.time();k=0
    while time.time()-t<secs:
        for _ in range(4):d.copy_(h,non_blocking=True)
        torch.cuda.synchronize();k+=4
    q.put((g,k*n/(time.time()-t)/1e9))
if __name__=='__main__':
    gs=[int(x) for x in sys.argv[1].split(',')];secs=float(sys.argv[2]);mb=int(sys.argv[3]) if len(sys.argv)>3 else 512
    ctx=mp.get_context('spawn');q=ctx.Queue();bar=ctx.Barrier(len(gs))
    ps=[ctx.Process(target=run,args=(g,secs,mb,q,bar)) for g in gs];[p.start() for p in ps];r=[q.get() for _ in gs];[p.join() for p in ps]
    r.sort();print({'per_gpu_GBps':[round(v,2) for _,v in r],'total_GBps':round(sum(v for _,v in r),2)},flush=True)
