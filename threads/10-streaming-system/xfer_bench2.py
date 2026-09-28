"""TP-shard-sized refinement chunks: back-to-back copies of plane chunks, 1 vs 2 streams, and overlap with a compute kernel."""
import json, time, torch
torch.cuda.set_per_process_memory_fraction(12/80)
dev = torch.device('cuda:0'); KB = 1024; MB = 1 << 20
POOL = 256 * MB
host = torch.empty(POOL, dtype=torch.uint8, pin_memory=True); host.fill_(1)
dst = torch.empty(POOL, dtype=torch.uint8, device=dev)
res = {}
def run(chunk, count, nstreams, reps=5):
    ss = [torch.cuda.Stream() for _ in range(nstreams)]
    def go():
        for i in range(count):
            with torch.cuda.stream(ss[i % nstreams]):
                o = (i * chunk) % (POOL - chunk)
                dst[o:o+chunk].copy_(host[o:o+chunk], non_blocking=True)
    go(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): go()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / reps
    return dict(total_us=dt*1e6, per_chunk_us=dt*1e6/count, GBps=chunk*count/dt/1e9)
for chunk in [196*KB, 590*KB, int(1.18*MB), int(4.72*MB), int(9.44*MB)]:
    for ns in [1, 2]:
        res[f'{chunk//KB}KB_x32_streams{ns}'] = run(chunk, 32, ns)
# overlap: matmul on default stream while copying on side stream
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): a @ a
torch.cuda.synchronize(); mm = (time.perf_counter() - t0) / 20
s = torch.cuda.Stream(); chunk = 590 * KB
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.cuda.stream(s):
    for i in range(64): dst[i*chunk:(i+1)*chunk].copy_(host[i*chunk:(i+1)*chunk], non_blocking=True)
for _ in range(20): a @ a
torch.cuda.synchronize(); both = time.perf_counter() - t0
res['overlap'] = dict(matmul20_ms=mm*20e3, copy64x590KB_plus_matmul20_ms=both*1e3, copy_alone_ms=run(chunk, 64, 1, 3)['total_us']/1e3)
print(json.dumps(res, indent=1))
