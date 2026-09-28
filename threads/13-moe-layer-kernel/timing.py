import torch,random,statistics
def bench(fns,blocks=60,repeats=10,warm=5):
    """fns: dict name->callable (no args). Captures each in a CUDA graph with `repeats` calls,
    then replays in shuffled order per block; returns median us per call."""
    graphs={}
    s=torch.cuda.Stream()
    for name,f in fns.items():
        with torch.cuda.stream(s):
            for _ in range(warm):f()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(repeats):f()
        graphs[name]=g
    rows={k:[] for k in fns}
    for seed in range(blocks):
        order=list(fns);random.Random(seed).shuffle(order)
        for name in order:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            graphs[name].replay();a.record();graphs[name].replay();b.record();b.synchronize()
            rows[name].append(a.elapsed_time(b)*1000/repeats)
    q=lambda v,p:sorted(v)[int(p*(len(v)-1))]
    return {k:statistics.median(v) for k,v in rows.items()},{k:(min(v),q(v,.1),q(v,.25)) for k,v in rows.items()}
