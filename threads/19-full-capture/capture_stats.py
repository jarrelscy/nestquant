"""Thread 19 stage 2: per-layer routed / context second-moment statistics, accumulated over progressive shards.

For every routed expert e of a layer (FP8-source teacher, fp32 dequant; bf16 cast for the SwiGLU path):
  A2[e] = sum_routed p^2 x x^T      A0[e] = sum_routed x x^T                      (gate/up input, 6144^2)
  D2[e] = sum_routed p^2 h h^T      D0[e] = sum_routed h h^T                      (down input, 2048^2)
  Dc[e] = sum_ctx h_e h_e^T                                                        (down input on context rows)
  g[e]  = [sum_r p^2 cg^2, sum_r cg^2, sum_r p^2 cu^2, sum_r cu^2, sum_c cg^2, sum_c cu^2]   (2048 each)
  scalars n, sum p, sum p^2, sum p^4  (ESS = (sum p^2)^2 / sum p^4)
per layer:
  C_ctx = sum_ctx x x^T,  C_all = sum_all-fit-rows x x^T
with h = bf16(silu(bf16(x g_bf16)) * bf16(x u_bf16)), cg = ux sig(gx)(1 + gx(1 - sig(gx))), cu = silu(gx),
ctx(shard k) = randperm(T_k, seed 20260925 + k)[:T_k // 4]  (k = 0 at T = 65536 is orbit's CalibrationBatches set).
Dc and the ctx g rows use the first n_dc = min(n_ctx_k, 131072) ctx rows scaled by n_ctx_k / n_dc (unbiased).
Every job (layer L, shard k) adds its sums into stats/L{L} (atomic directory swap under a per-layer flock).
Grams: bf16 tensor-core GEMMs with fp32 output per 2048-row sub-chunk, fp32 accumulation. See FORMAT.md.
"""
import argparse
import fcntl
import glob
import json
import os
import queue
import shutil
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

import nq19
from nq19 import D, F as FF, NEXP, OUT, npk, pack

ROWS = 16384          # rows per GPU chunk
TC = 2048             # rows per tensor-core Gram sub-chunk (fp32 accumulation across sub-chunks)
NDC = 131072          # context rows per shard used for Dc / ctx-g (scaled to n_ctx)
ALIGN = 4096
RAW = {"A2": D, "A0": D, "D2": FF, "D0": FF, "Dc": FF}


MIN_FREE_GB = 200


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def stride_bytes(n):
    return -(-npk(n) * 4 // ALIGN) * ALIGN


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def gram_(A, X, Wt=None):
    """A += Wt^T X over rows, bf16 tensor-core GEMM with fp32 output per TC-row sub-chunk, fp32 accumulate.
    X (and Wt) must be bf16; exact products, error only from in-GEMM accumulation (~1e-6 rel at TC=2048)."""
    Wt = X if Wt is None else Wt
    for i in range(0, X.shape[0], TC):
        A += torch.mm(Wt[i:i + TC].T, X[i:i + TC], out_dtype=torch.float32)


def wgram_(A, X, w):
    """A += sum_r w_r x_r x_r^T with fp32 weights: (w X) split into bf16 hi + lo (rel 2^-17), two TC Grams."""
    wx = X.float() * w
    hi = wx.bfloat16(); lo = (wx - hi.float()).bfloat16()
    del wx
    gram_(A, X, hi); gram_(A, X, lo)


def _ew(yb, gx, ux):
    h = F.silu(yb[:, :FF]) * yb[:, FF:]
    sig = gx.sigmoid()
    cg = ux * sig * (1 + gx * (1 - sig)); cu = F.silu(gx)
    return h, cg.square(), cu.square()


_ewc = torch.compile(_ew, dynamic=True)


class Teacher:
    def __init__(self, fp32):
        g, u, d = fp32
        gu = torch.cat([g, u])                                   # [4096, 6144] fp32 teacher
        self.hi = gu.bfloat16()                                  # = pilot's g.bfloat16() / u.bfloat16()
        self.lo = (gu - self.hi.float()).bfloat16()

    def fwd(self, xb, accurate):
        """xb bf16 [m, 6144] -> h (bf16 SwiGLU with bf16 teacher), cg^2, cu^2 (fp32).
        accurate: gx, ux from the fp32 teacher (bf16 hi + lo split, fp32 out); else from the bf16 linear outputs."""
        yb = torch.mm(xb, self.hi.T)                             # bf16 out (no reduced-precision split-K)
        if accurate:
            y = torch.mm(xb, self.hi.T, out_dtype=torch.float32) + torch.mm(xb, self.lo.T, out_dtype=torch.float32)
            gx, ux = y[:, :FF], y[:, FF:]
        else:
            yf = yb.float(); gx, ux = yf[:, :FF], yf[:, FF:]
        return _ewc(yb, gx, ux)


def prefetch(X, rows_list, out_q, stop):
    """Background gather of routed rows (host) into pinned chunks."""
    for e, rows in rows_list:
        for b in range(0, len(rows), ROWS):
            if stop.is_set():
                return
            r = rows[b:b + ROWS]
            buf = torch.empty(len(r), D, dtype=torch.bfloat16, pin_memory=True)
            torch.index_select(X, 0, r, out=buf)
            out_q.put((e, b, buf))
    out_q.put(None)


class RowIO:
    """Per-expert packed rows of the raw .f32 files: background O_DIRECT pwrite of new rows from pinned staging
    and background pread of the previous cumulative rows (prefetched one expert ahead)."""

    def __init__(self, new_dir, prev_dir, nbuf=3):
        self.wfd, self.rfd = {}, {}
        for k, n in RAW.items():
            path = f"{new_dir}/{k}.f32"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, NEXP * stride_bytes(n)); os.close(fd)
            self.wfd[k] = os.open(path, os.O_WRONLY | os.O_DIRECT)
            if prev_dir:
                self.rfd[k] = os.open(f"{prev_dir}/{k}.f32", os.O_RDONLY)
        mk = lambda: {k: torch.empty(stride_bytes(n) // 4, dtype=torch.float32, pin_memory=True) for k, n in RAW.items()}
        self.wpool = queue.Queue(); self.rpool = queue.Queue()
        for _ in range(nbuf):
            self.wpool.put(mk())
        self.err = None
        self.wq = queue.Queue()
        self.wth = threading.Thread(target=self._wrun, daemon=True); self.wth.start()
        if prev_dir:
            for _ in range(2):
                self.rpool.put(mk())
            self.rq = queue.Queue()
            self.rth = threading.Thread(target=self._rrun, daemon=True); self.rth.start()

    def _rrun(self):
        for e in range(NEXP):
            bufs = self.rpool.get()
            try:
                for k, n in RAW.items():
                    mv = memoryview(bufs[k].numpy()).cast("B")
                    nb = stride_bytes(n); got = 0
                    while got < nb:
                        r = os.preadv(self.rfd[k], [mv[got:nb]], e * nb + got)
                        if r <= 0:
                            raise IOError(f"short read {k} expert {e}")
                        got += r
            except Exception as ex:
                self.err = ex
            self.rq.put(bufs)

    def prev(self, e):
        """Previous cumulative packed rows of expert e on the GPU (dict), or None when this is the first shard."""
        if not self.rfd:
            return None
        bufs = self.rq.get()
        if self.err:
            raise self.err
        out = {k: bufs[k][:npk(n)].cuda() for k, n in RAW.items()}
        torch.cuda.synchronize()
        self.rpool.put(bufs)
        return out

    def _wrun(self):
        while True:
            it = self.wq.get()
            if it is None:
                return
            e, bufs, ev = it
            try:
                ev.synchronize()
                for k, n in RAW.items():
                    mv = memoryview(bufs[k].numpy()).cast("B")
                    nb = stride_bytes(n); done = 0
                    while done < nb:
                        done += os.pwrite(self.wfd[k], mv[done:nb], e * nb + done)
            except Exception as ex:
                self.err = ex
            self.wpool.put(bufs)

    def put(self, e, packed):
        bufs = self.wpool.get()
        for k, v in packed.items():
            bufs[k][:v.numel()].copy_(v, non_blocking=True)
            bufs[k][v.numel():].zero_()
        ev = torch.cuda.Event(); ev.record()
        self.wq.put((e, bufs, ev))

    def close(self):
        self.wq.put(None); self.wth.join()
        for fd in self.wfd.values():
            os.fsync(fd); os.close(fd)
        for fd in self.rfd.values():
            os.close(fd)
        if self.err:
            raise self.err


def current(sd):
    """Resolved version directory of the cumulative stats symlink sd (None if absent)."""
    return os.path.realpath(sd) if os.path.lexists(sd) else None


def publish(sd, target):
    """Atomically point symlink sd at `target` (link text, relative to sd's directory)."""
    tmp = f"{sd}.lnk.{os.getpid()}"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, sd)


def fsync_dir(d):
    for f in os.listdir(d):
        fd = os.open(f"{d}/{f}", os.O_RDONLY); os.fsync(fd); os.close(fd)
    fd = os.open(d, os.O_RDONLY); os.fsync(fd); os.close(fd)


@torch.no_grad()
def run_job(L, acts, sd, src, cache, max_rows=None):
    """Add shard `acts` (stage-1 acts/L{L}) into cumulative stats directory sd.  Caller holds the layer lock."""
    t0 = time.time()
    meta_in = json.load(open(f"{acts}/done.json"))
    proto = meta_in["protocol"]
    shard = dict(fit_start=proto.get("fit_start", 0), fit_windows=proto["fit_windows"], acts=acts)
    k = shard["fit_start"] // nq19.SHARD_WINDOWS
    shard["shard"] = k
    pd = current(sd)
    prev = json.load(open(f"{pd}/meta.json")) if pd else None
    if prev and any(s["fit_start"] == shard["fit_start"] for s in prev["shards"]):
        return prev                                              # already merged
    T = meta_in["rows"] if max_rows is None else max_rows
    nd = f"{sd}.v{len(prev['shards']) + 1 if prev else 1}"
    shutil.rmtree(nd, ignore_errors=True); os.makedirs(nd)
    tim = {}
    # ---- host activations
    X = torch.empty(T, D, dtype=torch.bfloat16)
    with open(f"{acts}/x.bf16", "rb") as f:
        mv = memoryview(X.view(torch.uint8).numpy().reshape(-1))
        if f.readinto(mv) != T * D * 2:
            raise ValueError("short x.bf16")
    ids = torch.from_numpy(np.fromfile(f"{acts}/ids.u8", dtype=np.uint8, count=T * 8).reshape(T, 8)).long()
    p = torch.from_numpy(np.fromfile(f"{acts}/p.f32", dtype=np.float32, count=T * 8).reshape(T, 8))
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    rows_all = order // 8; p_all = p.reshape(-1)[order]
    cnt = torch.bincount(flat, minlength=NEXP); offs = [0] + cnt.cumsum(0).tolist()
    tim["read"] = time.time() - t0
    n_ctx = T // nq19.CTX_FRACTION
    ctx = torch.randperm(T, generator=torch.Generator().manual_seed(nq19.CTX_SEED + k))[:n_ctx].clone()
    n_dc = min(n_ctx, NDC); dc_scale = n_ctx / n_dc
    shard.update(T=T, n_ctx=n_ctx, n_dc=n_dc, dc_scale=dc_scale, ctx_seed=nq19.CTX_SEED + k)
    cache.load(L)
    gd = np.zeros((NEXP, 6, FF), np.float64)
    sc = np.zeros((NEXP, 4), np.float64)
    # ---- per-layer grams: context rows and all fit rows
    tc = time.time()
    A = torch.zeros(D, D, device="cuda")
    for b in range(0, n_ctx, ROWS):
        gram_(A, X[ctx[b:b + ROWS]].cuda())
    C_ctx = pack(A).cpu().numpy()
    A.zero_()
    for b in range(0, T, ROWS):
        gram_(A, X[b:b + ROWS].cuda())
    C_all = pack(A).cpu().numpy()
    del A
    Xc = X[ctx[:n_dc]].cuda()                                    # Dc / ctx-g rows, resident
    tim["layer_grams"] = time.time() - tc
    # ---- experts
    io = RowIO(nd, pd)
    q = queue.Queue(maxsize=3); stop = threading.Event()
    rl = [(e, rows_all[offs[e]:offs[e + 1]]) for e in range(NEXP)]
    th = threading.Thread(target=prefetch, args=(X, rl, q, stop), daemon=True); th.start()
    tr = tctx = 0.
    item = q.get()
    for e in range(NEXP):
        ta = time.time()
        te = Teacher(cache.expert(e, dtype=torch.float32))
        A2 = torch.zeros(D, D, device="cuda"); A0 = torch.zeros(D, D, device="cuda")
        Dm = [torch.zeros(FF, FF, device="cuda") for _ in range(3)]
        g = torch.zeros(6, FF, device="cuda", dtype=torch.float64)
        pe = p_all[offs[e]:offs[e + 1]].double()
        sc[e] = [len(pe), pe.sum(), pe.square().sum(), pe.pow(4).sum()]
        while item is not None and item[0] == e:
            _, b, buf = item
            xb = buf.cuda(non_blocking=True)
            pp = p_all[offs[e] + b:offs[e] + b + len(xb)].cuda()
            p2 = pp.square()
            gram_(A0, xb); wgram_(A2, xb, p2[:, None])
            h, cg2, cu2 = te.fwd(xb, accurate=True)
            gram_(Dm[1], h); wgram_(Dm[0], h, p2[:, None])
            g[0] += (p2 @ cg2).double(); g[1] += cg2.sum(0).double()
            g[2] += (p2 @ cu2).double(); g[3] += cu2.sum(0).double()
            del xb, h, cg2, cu2
            item = q.get()
        torch.cuda.synchronize(); tb = time.time(); tr += tb - ta
        for b in range(0, n_dc, ROWS):
            h, cg2, cu2 = te.fwd(Xc[b:b + ROWS], accurate=False)
            gram_(Dm[2], h)
            g[4] += cg2.sum(0).double(); g[5] += cu2.sum(0).double()
            del h, cg2, cu2
        del te
        if dc_scale != 1:
            Dm[2] *= dc_scale; g[4:] *= dc_scale
        new = dict(A2=pack(A2), A0=pack(A0), D2=pack(Dm[0]), D0=pack(Dm[1]), Dc=pack(Dm[2]))
        del A2, A0, Dm
        old = io.prev(e)
        if old is not None:
            for kk in new:
                new[kk] += old[kk]
            del old
        io.put(e, new)
        del new
        gd[e] = g.cpu().numpy()
        tctx += time.time() - tb
        if e % 64 == 0:
            print(json.dumps(dict(layer=L, shard=k, expert=e, n=int(sc[e, 0]), routed_s=round(tr, 1), ctx_s=round(tctx, 1),
                                  elapsed=round(time.time() - t0, 1))), flush=True)
    stop.set(); th.join(timeout=5)
    io.close()
    shards = (prev["shards"] if prev else []) + [shard]
    if prev:
        C_ctx += np.load(f"{pd}/C_ctx.npy"); C_all += np.load(f"{pd}/C_all.npy")
        gd += np.load(f"{pd}/gdiag.npy"); sc += np.load(f"{pd}/scalars.npy")
    np.save(f"{nd}/C_ctx.npy", C_ctx); np.save(f"{nd}/C_all.npy", C_all)
    np.save(f"{nd}/gdiag.npy", gd); np.save(f"{nd}/scalars.npy", sc)
    ess = sc[:, 2] ** 2 / np.maximum(sc[:, 3], 1e-300)
    tim.update(routed=tr, ctx=tctx, total=time.time() - t0)
    meta = dict(schema="nestquant-19-stats-v2", layer=L, shards=shards,
                T_fit=sum(s["T"] for s in shards), n_ctx=sum(s["n_ctx"] for s in shards),
                files=dict(raw={kk: dict(file=f"{kk}.f32", rows=NEXP, n=n, packed=npk(n), stride_bytes=stride_bytes(n))
                                for kk, n in RAW.items()},
                           A2="sum_routed p^2 x x^T", A0="sum_routed x x^T", D2="sum_routed p^2 h h^T",
                           D0="sum_routed h h^T", Dc="sum_ctx h h^T (per shard: n_dc rows x n_ctx/n_dc)",
                           C_ctx="[npk(6144)] f32: sum_ctx x x^T", C_all="[npk(6144)] f32: sum over all fit rows x x^T",
                           gdiag="[256, 6, 2048] f64", scalars="[256, 4] f64"),
                packing="row-major upper triangle i <= j (nq19.pack / nq19.unpack)",
                gdiag_rows=["routed sum p^2 cg^2", "routed sum cg^2", "routed sum p^2 cu^2", "routed sum cu^2",
                            "ctx sum cg^2 (bf16 gx/ux)", "ctx sum cu^2 (bf16 gx/ux)"],
                scalars_cols=["n_routed", "sum_p", "sum_p2", "sum_p4"],
                arithmetic="bf16 tensor-core Grams, fp32 out per 2048-row sub-chunk, fp32 accumulation; p^2-weighted "
                           "Grams via bf16 hi/lo split of p^2 x; routed gx/ux fp32 (bf16 hi+lo teacher); "
                           "h = bf16 silu(x g_bf16) * (x u_bf16), no reduced-precision split-K",
                n_routed=sc[:, 0].astype(int).tolist(), ess=ess.round(1).tolist(),
                ess_summary=dict(min=float(ess.min()), p01=float(np.percentile(ess, 1)), p05=float(np.percentile(ess, 5)),
                                 median=float(np.median(ess)), max=float(ess.max())),
                n_summary=dict(min=int(sc[:, 0].min()), median=float(np.median(sc[:, 0])), max=int(sc[:, 0].max())),
                last_job_seconds={kk: round(v, 1) for kk, v in tim.items()}, torch=torch.__version__, complete=True)
    write_json(f"{nd}/meta.json", meta)
    fsync_dir(nd)
    publish(sd, os.path.basename(nd))                                              # readers see old or new, never partial
    if [s["shard"] for s in shards] == [0]:                      # frozen shard-0 snapshot (never auto-deleted)
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(sd)), "stats0"), exist_ok=True)
        publish(os.path.join(os.path.dirname(os.path.dirname(sd)), "stats0", os.path.basename(sd)),
                os.path.join("..", "stats", os.path.basename(nd)))
    if pd and not (len(prev["shards"]) == 1 and prev["shards"][0]["shard"] == 0):
        shutil.rmtree(pd, ignore_errors=True)                    # superseded (open readers keep their inodes)
    return meta


class LayerLock:
    def __init__(self, path):
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)

    def try_acquire(self):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB); return True
        except BlockingIOError:
            return False

    def release(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd)


def finish_acts(acts, delete_x):
    open(f"{acts}/merged", "w").close()
    if delete_x and os.path.exists(f"{acts}/x.bf16"):
        os.remove(f"{acts}/x.bf16")


def jobs(shards_root, last_layer):
    """(shard, layer, acts dir) ready for merging, in shard-major order (earliest shard first)."""
    out = []
    for d in sorted(glob.glob(f"{shards_root}/s*")):
        k = int(os.path.basename(d)[1:])
        for a in glob.glob(f"{d}/acts/L*"):
            L = int(os.path.basename(a)[1:])
            if L <= last_layer and os.path.exists(f"{a}/done.json") and not os.path.exists(f"{a}/merged"):
                out.append((k, L, a))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=OUT, help="capture root: shards in ROOT/shards/s*, cumulative stats in ROOT/stats")
    ap.add_argument("--layers", default="auto", help="comma list (single-dir mode: ROOT/acts -> ROOT/stats), or auto")
    ap.add_argument("--last-layer", type=int, default=77)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--keep-x-shards", default="0", help="shards whose x.bf16 are kept after merging")
    ap.add_argument("--exit-when-idle", type=int, default=0, help="exit after this many idle seconds (0 = never)")
    a = ap.parse_args()
    nq19.gpu_cap()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    src = nq19.Src(); cache = nq19.ExpertCache(src)
    os.makedirs(f"{a.root}/stats", exist_ok=True)
    if a.layers != "auto":
        for L in [int(v) for v in a.layers.split(",")]:
            m = run_job(L, f"{a.root}/acts/L{L}", f"{a.root}/stats/L{L}", src, cache, a.max_rows)
            print(json.dumps(m["last_job_seconds"]), flush=True)
        return
    keep = {int(v) for v in a.keep_x_shards.split(",") if v}
    idle = 0
    while True:
        did = False
        while free_gb(a.root) < MIN_FREE_GB + 50:
            print(json.dumps(dict(paused="disk", free_gb=round(free_gb(a.root)))), flush=True); time.sleep(120)
        for k, L, acts in jobs(f"{a.root}/shards", a.last_layer):
            lk = LayerLock(f"{a.root}/stats/L{L}.lock")
            if not lk.try_acquire():
                os.close(lk.fd); continue
            try:
                if os.path.exists(f"{acts}/merged"):
                    continue
                m = run_job(L, acts, f"{a.root}/stats/L{L}", src, cache)
                finish_acts(acts, k not in keep)
                print(json.dumps(dict(layer=L, shard=k, shards=len(m["shards"]), seconds=m["last_job_seconds"],
                                      ess=m["ess_summary"])), flush=True)
            finally:
                lk.release()
            did = True
            break
        if not did:
            idle += 20
            if a.exit_when_idle and idle >= a.exit_when_idle:
                return
            time.sleep(20)
        else:
            idle = 0


if __name__ == "__main__":
    main()
