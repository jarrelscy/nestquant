"""Short pinned host->GPU transfer benchmark for refinement chunks (GPU-light, <=1.2 GB VRAM)."""
import json, sys, time, torch
torch.cuda.set_per_process_memory_fraction(12/80)
dev = torch.device('cuda:0')
torch.cuda.init()
res = {'device': torch.cuda.get_device_name(0), 'label': sys.argv[1] if len(sys.argv) > 1 else ''}
POOL = 320 << 20
host = torch.empty(POOL, dtype=torch.uint8, pin_memory=True); host.random_(0, 255) if False else host.fill_(7)
pageable = torch.empty(64 << 20, dtype=torch.uint8); pageable.fill_(3)
dst = torch.empty(POOL, dtype=torch.uint8, device=dev)
s = torch.cuda.Stream()
def timed(fn, reps):
    torch.cuda.synchronize()
    for _ in range(2): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter(); e0.record(s)
    for _ in range(reps): fn()
    e1.record(s); e1.synchronize(); t1 = time.perf_counter()
    return e0.elapsed_time(e1) / reps * 1e3, (t1 - t0) / reps * 1e6  # us gpu, us wall
KB = 1024; MB = 1 << 20
# chunk sizes: 16-col block (2048 rows x16) @1bit=4KB, @2bit=8KB; 128x128 tile@1bit=2KB; matrix@1bit=1.5MiB;
# expert 2->3 = 4.5MiB, 2->4 = 9MiB; layer top-8 2->4 = 72MiB
sizes = [2*KB, 4*KB, 8*KB, 16*KB, 64*KB, 256*KB, 1*MB, int(1.5*MB), int(4.5*MB), 9*MB, 72*MB, 256*MB]
single = {}
with torch.cuda.stream(s):
    for n in sizes:
        reps = 200 if n <= MB else (40 if n <= 9*MB else 6)
        g, w = timed(lambda: dst[:n].copy_(host[:n], non_blocking=True), reps)
        single[n] = dict(gpu_us=g, wall_us=w, GBps=n / (g * 1e-6) / 1e9)
    res['single_copy_pinned'] = single
    pg = {}
    for n in [64*KB, 1*MB, 9*MB, 64*MB]:
        g, w = timed(lambda: dst[:n].copy_(pageable[:n], non_blocking=True), 5 if n > MB else 50)
        pg[n] = dict(gpu_us=g, wall_us=w, GBps=n / (w * 1e-6) / 1e9)
    res['single_copy_pageable'] = pg
    # scattered: many small chunks to scattered device slots (per-expert chunk gather) via individual copies
    scat = {}
    for chunk, count in [(4*KB, 512), (64*KB, 256), (256*KB, 64), (int(1.5*MB), 48)]:
        stride = POOL // count
        offs = [(i * 7919 % count) * stride for i in range(count)]
        def many():
            for i, o in enumerate(offs):
                dst[i*stride:i*stride+chunk].copy_(host[o:o+chunk], non_blocking=True)
        g, w = timed(many, 5)
        scat[f'{chunk}x{count}'] = dict(gpu_us=g, wall_us=w, per_copy_wall_us=w / count, GBps=chunk*count / (w*1e-6) / 1e9)
    res['scattered_individual_copies'] = scat
    # zero-copy gather kernel: GPU reads mapped pinned host memory via index_select on uint8->int32 view (UVA)
    zc = {}
    import triton, triton.language as tl
    @triton.jit
    def gather(src, dst, soff, doff, CH: tl.constexpr, BLK: tl.constexpr):
        c = tl.program_id(0); b = tl.program_id(1)
        so = tl.load(soff + c); do = tl.load(doff + c)
        i = b * BLK + tl.arange(0, BLK)
        m = i < CH
        v = tl.load(src + so + i, mask=m)
        tl.store(dst + do + i, v, mask=m)
    class HostPtr:
        def __init__(self, t): self.t = t; self.dtype = t.dtype
        def data_ptr(self): return self.t.data_ptr()
    hv = host.view(torch.int32); dv = dst.view(torch.int32)
    for chunk, count in [(4*KB, 512), (64*KB, 256), (256*KB, 64), (int(1.5*MB), 48), (9*MB, 8)]:
        stride = POOL // count
        so = torch.tensor([((i * 7919) % count) * stride // 4 for i in range(count)], dtype=torch.int64, device=dev)
        do = torch.tensor([i * stride // 4 for i in range(count)], dtype=torch.int64, device=dev)
        CH = chunk // 4; BLK = 4096
        grid = (count, triton.cdiv(CH, BLK))
        try:
            fn = lambda: gather[grid](HostPtr(hv), dv, so, do, CH=CH, BLK=BLK)
            g, w = timed(fn, 5)
            # correctness
            ok = bool((dst[:chunk] == host[int(so[0])*4:int(so[0])*4+chunk].to(dev)).all())
            zc[f'{chunk}x{count}'] = dict(gpu_us=g, wall_us=w, GBps=chunk*count/(g*1e-6)/1e9, ok=ok)
        except Exception as ex:
            zc[f'{chunk}x{count}'] = f'failed: {str(ex)[:200]}'
    # graph capture of the zero-copy gather (pointer tables live on device)
    try:
        chunk, count = 64*KB, 256; stride = POOL // count
        so = torch.tensor([((i * 7919) % count) * stride // 4 for i in range(count)], dtype=torch.int64, device=dev)
        do = torch.tensor([i * stride // 4 for i in range(count)], dtype=torch.int64, device=dev)
        grid = (count, triton.cdiv(chunk//4, 4096))
        gather[grid](HostPtr(hv), dv, so, do, CH=chunk//4, BLK=4096); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gather[grid](HostPtr(hv), dv, so, do, CH=chunk//4, BLK=4096)
        so.copy_(torch.roll(so, 1)); gr.replay(); torch.cuda.synchronize()
        ok = bool((dst[:chunk] == host[int(so[0])*4:int(so[0])*4+chunk].to(dev)).all())
        g, w = timed(lambda: gr.replay(), 10)
        zc['graph_64KBx256'] = dict(gpu_us=g, wall_us=w, GBps=chunk*count/(g*1e-6)/1e9, ok_after_table_change=ok)
    except Exception as ex:
        zc['graph'] = f'failed: {str(ex)[:300]}'
    res['zero_copy'] = zc
    # D2D within GPU (HBM copy) for reference
    d2 = {}
    for n in [9*MB, 72*MB]:
        g, w = timed(lambda: dst[POOL//2:POOL//2+n].copy_(dst[:n], non_blocking=True), 20)
        d2[n] = dict(gpu_us=g, GBps=2*n / (g*1e-6) / 1e9)
    res['d2d_hbm_rw'] = d2
print(json.dumps(res, indent=1))
