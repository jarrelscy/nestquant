#!/usr/bin/env python3
"""T33l fp8dec: expert-streaming bench + bitwise check, one sparse layer, all 8 GPUs (EP: GPU g owns e % 8 == g).
  A  pread -> per-thread pinned staging -> H2D (gen.py path), NT threads
  R  cudaHostRegister(Portable|ReadOnly) of the mmap'd safetensors runs -> async H2D straight from the page cache
     (no CPU copy), unregistered after the copies complete (finally + atexit)
  P  pageable mmap -> H2D (driver staging), one thread per device
Every mode's device bytes are compared bitwise (torch.equal) to mode A.
  sbench.py LAYER [NT] [modes=A,R,P]"""
import atexit
import concurrent.futures as cf
import mmap
import os
import sys
import time
import numpy as np
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_io  # noqa: E402
import gen as G  # noqa: E402

li = int(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 16
modes = (sys.argv[3] if len(sys.argv) > 3 else "A,R,P").split(",")
idx = nq_io.SafeIndex(G.FP8_DIR)
D, E = 8, 256
devs = [torch.device(f"cuda:{g}") for g in range(D)]
names = {g: [f"model.layers.{li}.mlp.experts.{e}.{k}{s}" for e in range(g, E, D)
             for k in ("gate_proj", "up_proj", "down_proj") for s in (".weight", ".weight_scale_inv")] for g in range(D)}
tot = sum(idx.map[n][4] for g in range(D) for n in names[g])
for d in devs:
    torch.zeros(1, device=d)
REG = []                                    # live host registrations (ptr) for cleanup


def _unreg_all():
    cr = torch.cuda.cudart()
    while REG:
        p = REG.pop()
        cr.cudaHostUnregister(p)


atexit.register(_unreg_all)


def mode_A():
    raw = G.Raw(idx)
    pool = cf.ThreadPoolExecutor(NT)
    parts = max(1, NT // D)

    def one(g, ch):
        return [raw.read_many(ch[i:i + 6], devs[g]) for i in range(0, len(ch), 6)]
    t = time.time()
    fs = []
    for g in range(D):
        ex = [names[g][i:i + 6] for i in range(0, len(names[g]), 6)]
        for p in range(parts):
            fs.append((g, pool.submit(one, g, sum(ex[p::parts], []))))
    out = {g: {} for g in range(D)}
    for g, f in fs:
        for t_ in f.result():
            for k, v in t_.items():
                if k != "__buf__":
                    out[g][k] = v
    for d in devs:
        torch.cuda.synchronize(d)
    return out, time.time() - t


def runs_of(g):
    """per file: merged byte runs (gap <= 2 MiB) covering device g's tensors."""
    by = {}
    for n in names[g]:
        f, dt, sh, off, nb = idx.map[n]
        by.setdefault(f, []).append((off, nb, n))
    R = []
    for f, v in by.items():
        v.sort()
        cur = None
        for off, nb, n in v:
            if cur and off - cur[2] <= 2 << 20:
                cur[2] = max(cur[2], off + nb); cur[3].append((off, nb, n))
            else:
                if cur:
                    R.append(tuple(cur))
                cur = [f, off, off + nb, [(off, nb, n)]]
        R.append(tuple(cur))
    return R


MM = {}


def mm(f):
    if f not in MM:
        fd = os.open(f, os.O_RDONLY)
        MM[f] = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        os.close(fd)
    return MM[f]


def mode_R():
    cr = torch.cuda.cudart()
    PG = mmap.PAGESIZE
    out = {g: {} for g in range(D)}
    treg = [0.0] * D

    def dev(g):
        d = devs[g]
        s = torch.cuda.Stream(device=d)
        mine = []
        for f, lo, hi, ts in runs_of(g):
            m = mm(f)
            a0 = lo // PG * PG
            a1 = (hi + PG - 1) // PG * PG
            base = np.frombuffer(m, dtype=np.uint8, count=a1 - a0, offset=a0)
            ptr = base.ctypes.data
            t0 = time.time()
            err = cr.cudaHostRegister(ptr, a1 - a0, 0x01 | 0x08)     # Portable | ReadOnly
            treg[g] += time.time() - t0
            assert int(err) == 0, f"cudaHostRegister err {err}"
            REG.append(ptr); mine.append(ptr)
            host = torch.from_numpy(base)                           # aliases the registered page-cache mapping
            with torch.cuda.stream(s):
                for off, nb, n in ts:
                    dt, sh = idx.map[n][1], idx.map[n][2]
                    buf = torch.empty(nb, dtype=torch.uint8, device=d)
                    buf.copy_(host[off - a0:off - a0 + nb], non_blocking=True)
                    out[g][n] = buf.view(nq_io.DT[dt]).view(sh)
        s.synchronize()
        for p in mine:
            cr.cudaHostUnregister(p); REG.remove(p)
    t = time.time()
    with cf.ThreadPoolExecutor(D) as p:
        list(p.map(dev, range(D)))
    return out, time.time() - t, max(treg)


def mode_F():
    """dec.py's actual path: dec.Reg (refcounted whole-file registration) + DModel._load_reg, 8 loader threads."""
    import types
    import dec as Dm
    fake = types.SimpleNamespace(idx=idx, reg=Dm.Reg(), ls=[torch.cuda.Stream(device=d) for d in devs], devs=devs,
                                 E=E, D=D)
    fake._exp_names = types.MethodType(Dm.DModel._exp_names, fake)
    t = time.time()
    with cf.ThreadPoolExecutor(D) as p:
        res = list(p.map(lambda g: Dm.DModel._load_reg(fake, li, g), range(D)))
    dt = time.time() - t
    out = {g: {} for g in range(D)}
    for g in range(D):
        for e, dd in res[g].items():
            for k in ("gate_proj", "up_proj", "down_proj"):
                out[g][f"model.layers.{li}.mlp.experts.{e}.{k}.weight"] = dd[k][0]
                out[g][f"model.layers.{li}.mlp.experts.{e}.{k}.weight_scale_inv"] = dd[k][1]
    live = fake.reg.live_gb(); nent = len(fake.reg.ent)
    fake.reg.close_all()
    print(f"  F: live registrations after layer {nent} ({live:.1f} GB); total register time {fake.reg.t_reg:.2f}s",
          flush=True)
    return out, dt, fake.reg.t_reg


def mode_Q(nthr, src):
    """dec.py --stream pin: PinRing (pinned slots + nthr host-copy threads per device) + DModel._load_pin."""
    import types
    import dec as Dm
    fake = types.SimpleNamespace(idx=idx, ring=Dm.PinRing(devs, nthr, 64, src),
                                 ls=[torch.cuda.Stream(device=d) for d in devs], devs=devs, E=E, D=D)
    fake._exp_names = types.MethodType(Dm.DModel._exp_names, fake)
    ts = []
    for rep in range(2):
        t = time.time()
        with cf.ThreadPoolExecutor(D) as p:
            res = list(p.map(lambda g: Dm.DModel._load_pin(fake, li, g), range(D)))
        ts.append(time.time() - t)
        if rep == 0:
            del res
    out = {g: {} for g in range(D)}
    for g in range(D):
        for e, dd in res[g].items():
            for k in ("gate_proj", "up_proj", "down_proj"):
                out[g][f"model.layers.{li}.mlp.experts.{e}.{k}.weight"] = dd[k][0]
                out[g][f"model.layers.{li}.mlp.experts.{e}.{k}.weight_scale_inv"] = dd[k][1]
    print(f"  Q({nthr},{src}): rep times {[round(x, 2) for x in ts]}", flush=True)
    return out, ts[-1]


def mode_P():
    out = {g: {} for g in range(D)}

    def dev(g):
        d = devs[g]
        for f, lo, hi, ts in runs_of(g):
            m = mm(f)
            for off, nb, n in ts:
                dt, sh = idx.map[n][1], idx.map[n][2]
                host = torch.from_numpy(np.frombuffer(m, dtype=np.uint8, count=nb, offset=off))
                out[g][n] = host.to(d).view(nq_io.DT[dt]).view(sh)
        torch.cuda.synchronize(d)
    t = time.time()
    with cf.ThreadPoolExecutor(D) as p:
        list(p.map(dev, range(D)))
    return out, time.time() - t


if __name__ == "__main__":
    ref = None
    for m in modes:
        if m.startswith("Q"):              # Q<nthr><m|p>, e.g. Q4p = 4 threads/device preadv, Q4m = mmap memcpy
            r = mode_Q(int(m[1:-1]), {"p": "pread", "m": "mmap"}[m[-1]])
        else:
            r = {"A": mode_A, "R": mode_R, "P": mode_P, "F": mode_F}[m]()
        o, dt = r[0], r[1]
        extra = f" (max per-device register {r[2]:.2f}s)" if len(r) > 2 else ""
        ok = ""
        if ref is None:
            ref = o
        else:
            ok = " bitwise " + ("OK" if all(torch.equal(o[g][n].view(torch.uint8), ref[g][n].view(torch.uint8))
                                              for g in range(D) for n in names[g]) else "MISMATCH")
        print(f"L{li} mode {m} NT {NT}: {tot/1e9:.2f} GB in {dt:.2f}s = {tot/1e9/dt:.1f} GB/s{extra}{ok}", flush=True)
        if m != "A":
            del o
        torch.cuda.empty_cache()
    _unreg_all()
    print("registrations left:", len(REG))
    for g in range(D):
        print(f"cuda:{g} mem allocated {torch.cuda.memory_allocated(devs[g])/2**30:.2f} GB", flush=True)
