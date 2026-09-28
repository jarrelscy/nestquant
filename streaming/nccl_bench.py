"""TP4 collectives at prefill (8192x6144 fp16) and decode (4x6144) sizes, median ms. usage: nccl_bench.py REPS TAG"""
import torch,torch.distributed as dist,torch.multiprocessing as mp,sys,os,statistics,json
def run(r,reps,tag):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='29533');torch.cuda.set_device(r)
    dist.init_process_group('nccl',rank=r,world_size=4,device_id=torch.device('cuda',r))
    big=torch.randn(8192,6144,device='cuda').half();sh=torch.randn(2048,6144,device='cuda').half();small=torch.randn(4,6144,device='cuda').half()
    ops={'allreduce_8192':lambda:dist.all_reduce(big),'allgather_8192':lambda:dist.all_gather_into_tensor(big,sh),
         'reducescatter_8192':lambda:dist.reduce_scatter_tensor(sh,big),'allreduce_4tok':lambda:dist.all_reduce(small)}
    res={}
    for k,f in ops.items():
        for _ in range(5):f()
        ts=[]
        for _ in range(reps):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();f();b.record();b.synchronize();ts.append(a.elapsed_time(b))
        res[k]=round(statistics.median(ts),3)
    if r==0:print(json.dumps({'tag':tag,**res}),flush=True)
    dist.destroy_process_group()
if __name__=='__main__':mp.spawn(run,args=(int(sys.argv[1]),sys.argv[2]),nprocs=4)
