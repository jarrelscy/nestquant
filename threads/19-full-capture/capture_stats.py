"""Thread 19 stage 2: per-layer routed / context second-moment statistics from stage-1 activations.

For every routed expert e of a layer (FP8-source teacher, fp32 dequant; bf16 cast for the SwiGLU path):
  A2[e] = sum_routed p^2 x x^T      A0[e] = sum_routed x x^T                      (gate/up input, 6144^2)
  D2[e] = sum_routed p^2 h h^T      D0[e] = sum_routed h h^T                      (down input, 2048^2)
  Dc[e] = sum_ctx h_e h_e^T                                                        (down input on context rows)
  g[e]  = [sum_r p^2 cg^2, sum_r cg^2, sum_r p^2 cu^2, sum_r cu^2, sum_c cg^2, sum_c cu^2]   (2048 each)
  scalars n, sum p, sum p^2, sum p^4  (ESS = (sum p^2)^2 / sum p^4)
per layer:
  C_ctx = sum_ctx x x^T,  C_all = sum_all-fit-rows x x^T
with h = bf16(silu(bf16(x g_bf16)) * bf16(x u_bf16)), cg = ux sig(gx)(1 + gx(1 - sig(gx))), cu = silu(gx) (gx, ux fp32),
ctx = randperm(T_fit, seed 20260925)[:T_fit // 4]  (identical to orbit CalibrationBatches at T = 65536).
All Grams fp32 (no TF32), stored as packed upper triangles; vectors/scalars fp64.  See FORMAT.md.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

import nq19
from nq19 import D, F as FF, NEXP, OUT, npk, pack

ROWS = 16384          # rows per GPU chunk
TC = 2048             # rows per tensor-core Gram sub-chunk (fp32 accumulation across sub-chunks)


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


class RowWriter:
    """Pinned staging + background pwrite of per-expert packed rows into preallocated .npy files."""

    def __init__(self, files, nbuf=3):
        self.files = {}
        for k, (path, n) in files.items():
            m = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(NEXP, npk(n)))
            off = m.offset; del m
            self.files[k] = (os.open(path, os.O_WRONLY), off, npk(n) * 4)
        self.pool = queue.Queue()
        for _ in range(nbuf):
            self.pool.put({k: torch.empty(npk(n), dtype=torch.float32, pin_memory=True) for k, (_, n) in files.items()})
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
        self.err = None

    def _run(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            e, bufs, ev = it
            try:
                ev.synchronize()
                for k, b in bufs.items():
                    fd, off, nb = self.files[k]
                    mv = memoryview(b.numpy()).cast("B")
                    done = 0
                    while done < nb:
                        done += os.pwrite(fd, mv[done:], off + e * nb + done)
            except Exception as ex:          # surfaced in close()
                self.err = ex
            self.pool.put(bufs)

    def put(self, e, mats):
        bufs = self.pool.get()
        for k, A in mats.items():
            bufs[k].copy_(pack(A), non_blocking=True)
        ev = torch.cuda.Event(); ev.record()
        self.q.put((e, bufs, ev))

    def close(self):
        self.q.put(None); self.th.join()
        for fd, _, _ in self.files.values():
            os.fsync(fd); os.close(fd)
        if self.err:
            raise self.err


@torch.no_grad()
def run_layer(L, acts, outd, src, cache, max_rows=None):
    t0 = time.time()
    meta_in = json.load(open(f"{acts}/done.json"))
    T = meta_in["rows"] if max_rows is None else max_rows
    os.makedirs(outd, exist_ok=True)
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
    ctx = torch.randperm(T, generator=torch.Generator().manual_seed(nq19.CTX_SEED))[:n_ctx].clone()
    ctx.numpy().astype(np.int64).tofile(f"{outd}/ctx_idx.i64")
    cache.load(L)
    gd = np.zeros((NEXP, 6, FF), np.float64)
    sc = np.zeros((NEXP, 4), np.float64)
    # ---- per-layer grams: context rows and all fit rows
    tc = time.time()
    Xc = X[ctx].cuda()                                            # [n_ctx, 6144] bf16, resident
    A = torch.zeros(D, D, device="cuda")
    gram_(A, Xc)
    np.save(f"{outd}/C_ctx.npy", pack(A).cpu().numpy())
    A.zero_()
    for b in range(0, T, ROWS):
        gram_(A, X[b:b + ROWS].cuda())
    np.save(f"{outd}/C_all.npy", pack(A).cpu().numpy())
    del A
    tim["layer_grams"] = time.time() - tc
    # ---- experts
    W = RowWriter({k: (f"{outd}/{k}.npy", n) for k, n in (("A2", D), ("A0", D), ("D2", FF), ("D0", FF), ("Dc", FF))})
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
        for b in range(0, n_ctx, ROWS):
            h, cg2, cu2 = te.fwd(Xc[b:b + ROWS], accurate=False)
            gram_(Dm[2], h)
            g[4] += cg2.sum(0).double(); g[5] += cu2.sum(0).double()
            del h, cg2, cu2
        del te
        W.put(e, dict(A2=A2, A0=A0, D2=Dm[0], D0=Dm[1], Dc=Dm[2]))
        gd[e] = g.cpu().numpy()
        del A2, A0, Dm
        tctx += time.time() - tb
        if e % 32 == 0:
            print(json.dumps(dict(layer=L, expert=e, n=int(sc[e, 0]), routed_s=round(tr, 1), ctx_s=round(tctx, 1),
                                  elapsed=round(time.time() - t0, 1))), flush=True)
    stop.set(); th.join(timeout=5)
    W.close()
    np.save(f"{outd}/gdiag.npy", gd)
    np.save(f"{outd}/scalars.npy", sc)
    ess = sc[:, 2] ** 2 / np.maximum(sc[:, 3], 1e-300)
    tim.update(routed=tr, ctx=tctx, total=time.time() - t0)
    meta = dict(layer=L, T_fit=T, n_ctx=n_ctx, ctx_seed=nq19.CTX_SEED, ctx_rule="randperm(T_fit, seed)[:T_fit//4]",
                acts=acts, source=src.root,
                files=dict(A2="[256, npk(6144)] f32 packed upper: sum_routed p^2 x x^T", A0="sum_routed x x^T",
                           D2="[256, npk(2048)]: sum_routed p^2 h h^T", D0="sum_routed h h^T", Dc="sum_ctx h h^T",
                           C_ctx="[npk(6144)]: sum_ctx x x^T", C_all="[npk(6144)]: sum over all T_fit rows x x^T",
                           gdiag="[256, 6, 2048] f64", scalars="[256, 4] f64", ctx_idx="[n_ctx] int64 raw"),
                gdiag_rows=["routed sum p^2 cg^2", "routed sum cg^2", "routed sum p^2 cu^2", "routed sum cu^2",
                            "ctx sum cg^2 (bf16 gx/ux)", "ctx sum cu^2 (bf16 gx/ux)"],
                scalars_cols=["n_routed", "sum_p", "sum_p2", "sum_p4"],
                arithmetic="bf16 tensor-core Grams, fp32 out per 2048-row sub-chunk, fp32 accumulation; p^2-weighted "
                           "Grams via bf16 hi/lo split of p^2 x; h = bf16 silu(x g_bf16) * (x u_bf16), no reduced-precision split-K",
                n_routed=sc[:, 0].astype(int).tolist(), ess=ess.round(1).tolist(),
                ess_summary=dict(min=float(ess.min()), p05=float(np.percentile(ess, 5)), median=float(np.median(ess)), max=float(ess.max())),
                n_summary=dict(min=int(sc[:, 0].min()), median=float(np.median(sc[:, 0])), max=int(sc[:, 0].max())),
                seconds={k: round(v, 1) for k, v in tim.items()}, torch=torch.__version__, complete=True)
    write_json(f"{outd}/meta.json", meta)
    return meta


def claim(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {os.environ.get('CUDA_VISIBLE_DEVICES')}\n".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=OUT)
    ap.add_argument("--layers", default="auto", help="comma list, or 'auto' = claim layers as stage 1 finishes them")
    ap.add_argument("--last-layer", type=int, default=77)
    ap.add_argument("--max-rows", type=int)
    a = ap.parse_args()
    nq19.gpu_cap()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    src = nq19.Src(); cache = nq19.ExpertCache(src)
    os.makedirs(f"{a.root}/stats", exist_ok=True)
    if a.layers != "auto":
        for L in [int(v) for v in a.layers.split(",")]:
            print(json.dumps(run_layer(L, f"{a.root}/acts/L{L}", f"{a.root}/stats/L{L}", src, cache, a.max_rows)["seconds"]), flush=True)
        return
    todo = list(range(src.config["first_k_dense_replace"], a.last_layer + 1))
    while todo:
        for L in list(todo):
            sd = f"{a.root}/stats/L{L}"
            if os.path.exists(f"{sd}/meta.json"):
                todo.remove(L); continue
            if not os.path.exists(f"{a.root}/acts/L{L}/done.json"):
                continue
            os.makedirs(sd, exist_ok=True)
            if not claim(f"{sd}/claim"):
                todo.remove(L); continue
            m = run_layer(L, f"{a.root}/acts/L{L}", sd, src, cache)
            print(json.dumps(dict(layer=L, seconds=m["seconds"], ess=m["ess_summary"])), flush=True)
            todo.remove(L)
            break
        else:
            time.sleep(20)


if __name__ == "__main__":
    main()
