"""T33l: CPU-only host copy bench for one sparse layer's experts (all 8 device shards): mmap->buffer (np.copyto) vs
preadv->buffer, NT threads, 64 MiB pieces.  cpu_copybench.py LAYER NT"""
import concurrent.futures as cf, mmap, os, sys, time
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/ceiling")
import nq_io, gen as G
li, NT = int(sys.argv[1]), int(sys.argv[2])
idx = nq_io.SafeIndex(G.FP8_DIR)
ms = [idx.map[n] for n in idx.names() if n.startswith(f"model.layers.{li}.mlp.experts.")]
P = 64 << 20
pieces = []
for f, dt, sh, off, nb in ms:
    for o in range(0, nb, P):
        pieces.append((f, off + o, min(P, nb - o)))
tot = sum(p[2] for p in pieces)
bufs = [np.empty(P, np.uint8) for _ in range(NT)]
for b in bufs: b[:] = 1
MM, FD = {}, {}
def mm(f):
    if f not in MM:
        fd = os.open(f, os.O_RDONLY); MM[f] = mmap.mmap(fd, 0, prot=mmap.PROT_READ); FD[f] = fd
    return MM[f]
for f, *_ in pieces: mm(f)
import threading
tl = threading.local()
def buf():
    if not hasattr(tl, "b"): tl.b = bufs[int(threading.current_thread().name.split("_")[-1])]
    return tl.b
def cp_mm(p):
    f, o, n = p; np.copyto(buf()[:n], np.frombuffer(MM[f], np.uint8, count=n, offset=o))
def cp_pr(p):
    f, o, n = p; b = memoryview(buf()); got = 0
    while got < n: got += os.preadv(FD[f], [b[got:n]], o + got)
for name, fn in (("pread", cp_pr), ("mmap", cp_mm), ("pread", cp_pr), ("mmap", cp_mm)):
    with cf.ThreadPoolExecutor(NT) as ex:
        t = time.time(); list(ex.map(fn, pieces)); dt = time.time() - t
    print(f"L{li} {name:5s} NT={NT} {tot/2**30:.2f} GiB {dt:.2f}s {tot/dt/2**30:.1f} GiB/s", flush=True)
