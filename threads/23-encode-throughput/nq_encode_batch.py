"""Thread 23: layer-batched NestQuant encoder, byte-identical to thread 12's production single pass.

    encode_experts([(L, E), ...], group=4) -> yields (L, E, artifact) with the artifact nq_layer / nq_encode write
    python nq_encode_batch.py --experts 16:36,16:92 --out DIR   (or --layer L --range 0:256)

Structure (so T12 stage changes carry over): every numerically relevant stage is T12's own function, called in T12's
order with T12's arguments -- NE.prep, NE.encode_projection (tr proxies, NE.refit_dense, NE.pack, info), the decode
check (D.rotated_levels / D.decode_matrix), NE.mbn_candidates, D.fold / D.hsum / D.q2_values ... . Only two things
are re-implemented, as line-by-line mirrors of T12 code:
  * encode_rotated_batch  <- NE.encode_rotated (+ NE.base_quant): the same per-member tensor ops, but the Viterbi
    launches of all members of a step (G experts x {gate, up}, or G x down) go out as ONE launch, the 12 (Mb, N)
    fold candidates are scored in one batched call, and all host syncs are gone (nonzero / tolist / item / masked
    assignment -> precomputed indices / torch.where);
  * encode_group          <- NE.encode_expert (orchestration + meta; gate/up of the group first, then down).
Exact savings on top (all inside T12's call graph via scoped monkeypatches, see `patched`):
  * g_scale search: gs(K2) == gsr[2] == gsr[2.0] -> computed once (memo on identical sample tensors); the 10-point
    coarse / 5-point fine grids go out as one quantizer launch each (per-scale mse on the same-shape slice);
  * gate and up share H, seed and sigma -> the block LDL of the rotated H (and its 16-blocks) is computed once;
  * unrotate_H (CPU) is memoised: gate/up x (L2, L4) refits reuse one result;
  * frac Viterbi scratch 64 -> 216 tiles (2 waves of 108 SMs), EXL3_QT_OPTIMIZED=1 (both bit-identical).
Batched reductions / bmm whose cuBLAS / reduction config could depend on the batch size go through `Seg`, which
checks once per signature (on random data of the exact shapes) that the batched call equals the per-segment loop and
otherwise runs the loop.  check_bitid.py gates all of it against the live T12 reference.
"""
import os, sys, time, math, contextlib, collections, threading, argparse, json
os.environ.setdefault("OMP_NUM_THREADS", "16")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import torch
import nq_encode as NE
import nq_decode as D
import nq_patvit as PV
import harness as h

DEFAULT_OPTS = dict(qt_opt=True, frac_tiles=216, gdedup=True, gbatch=True, ldl_share=True, unrot_cache=True,
                    batched=True, cand_chunk_units=6144, seg_check=True, k2vit=False)


# ================================================================================================ exact patches
def _qkey(q):
    return ("ext", q.codebook) if isinstance(q, h.ExtTileQuantizer) else ("fn", id(q))


_K2Q = [None]           # k2vit.quantize_tiles when opts["k2vit"] (exact stand-in for ExtTileQuantizer("mul1") at K2)


def _mse_multi(tiles, ss, K, quantizer):
    """[mse(tiles, s) for s in ss] of harness._g_scale_search, one quantizer launch (tiles are independent)."""
    n = tiles.shape[0]
    x = torch.cat([tiles * s for s in ss])
    if _K2Q[0] is not None and _qkey(quantizer) == ("ext", "mul1") and float(K) == 2.0 and x.shape[1] == 256:
        q, _ = _K2Q[0](x.float())
    else:
        q, _ = quantizer(x, K)
    return torch.stack([(q[i * n:(i + 1) * n] / s - tiles).square().mean() for i, s in enumerate(ss)]).tolist()


def g_scale_search_batched(samples, K, quantizer):
    """harness._g_scale_search with the grid points of each stage in one launch (same floats, same decisions)."""
    coarse = [0.1 + 0.2 * i for i in range(10)]
    sub = samples[::3]
    mc = _mse_multi(sub, coarse, K, quantizer)
    c = coarse[min(range(10), key=lambda i: mc[i])]
    step = 0.075
    fine = [c + step * (i - 2) for i in range(5)]
    m = _mse_multi(samples, fine, K, quantizer)
    best = min(range(5), key=lambda i: m[i])
    off = 0.0
    if 0 < best < 4:
        den = m[best - 1] - 2 * m[best] + m[best + 1]
        off = max(-.5, min(.5, 0.5 * (m[best - 1] - m[best + 1]) / den)) if den > 0 else 0.0
    return max(fine[best] + off * step, 0.01), m[best]


class Memo:
    """Result memo for pure functions of tensors: hit iff same scalar args and every tensor arg is the very same
    (unmodified) view or has identical contents (content check only for tensors >= `big` elements)."""
    def __init__(self, keep=64, big=1 << 20):
        self.e, self.keep, self.big = [], keep, big
        self.hits = self.miss = 0

    @staticmethod
    def _same(x, y, big):
        if torch.is_tensor(x) != torch.is_tensor(y):
            return False
        if not torch.is_tensor(x):
            return x == y
        if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
            return False
        if x.data_ptr() == y.data_ptr() and x.stride() == y.stride() and x._version == y._version:
            return True
        return x.numel() >= big and torch.equal(x, y)

    def wrap(self, fn, norm=lambda a: a):
        def g(*args, **kw):
            key = norm(args) + tuple(sorted(kw.items()))
            for k, ver, r in self.e:
                if len(k) == len(key) and all(self._same(a, b, self.big) for a, b in zip(k, key)) and \
                        all(not torch.is_tensor(a) or a._version == v for a, v in zip(k, ver)):
                    self.hits += 1
                    return r
            r = fn(*args, **kw)
            self.miss += 1
            self.e.append((key, [a._version if torch.is_tensor(a) else None for a in key], r))
            del self.e[:-self.keep]
            return r
        return g

    def clear(self):
        self.e.clear()


LDL_MEMO, GS_MEMO, UNROT_MEMO = Memo(keep=128), Memo(keep=8, big=0), Memo(keep=2, big=0)


@contextlib.contextmanager
def patched(opts):
    """Scoped exact speed-ups inside T12's call graph (restored on exit, so a reference encode in the same process
    is unaffected)."""
    saved = []

    def setp(obj, name, val):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    if opts.get("qt_opt"):
        os.environ["EXL3_QT_OPTIMIZED"] = "1"          # read per launch by the extension (layout via get_temp_buffers)
        h.free_scratch()
    if opts.get("frac_tiles") and PV.TMP_TILES != opts["frac_tiles"]:
        setp(PV, "TMP_TILES", opts["frac_tiles"]); PV._TMP.clear()
    gs = g_scale_search_batched if opts.get("gbatch") else h._g_scale_search
    if opts.get("gdedup"):
        gs = GS_MEMO.wrap(gs, norm=lambda a: (a[0], float(a[1]), _qkey(a[2])) + tuple(a[3:]))
    setp(h, "_g_scale_search", gs)
    if opts.get("ldl_share"):
        setp(NE, "ldl_blocks", LDL_MEMO.wrap(NE.ldl_blocks))
    if opts.get("unrot_cache"):
        Qm = h._ex()
        setp(Qm, "unrotate_H", UNROT_MEMO.wrap(Qm.unrotate_H))
    if opts.get("k2vit"):
        import k2vit
        v0 = NE.viterbi

        def viterbi(rings, K):
            if not PV.is_pat(K) and float(K) == 2.0 and rings.dim() == 2 and rings.shape[1] == 256:
                return k2vit.states(rings)
            return v0(rings, K)
        setp(NE, "viterbi", viterbi)
        _K2Q[0] = k2vit.quantize_tiles
    try:
        yield
    finally:
        for obj, name, val in reversed(saved):
            setattr(obj, name, val)
        for m in (LDL_MEMO, GS_MEMO, UNROT_MEMO):
            m.clear()
        _K2Q[0] = None
        if opts.get("qt_opt"):
            os.environ.pop("EXL3_QT_OPTIMIZED", None)
            h.free_scratch()
        PV._TMP.clear()


# ================================================================================================ batch-invariance
class Seg:
    """fn(*args) on args stacked along dim 0 from `nseg` equal segments == cat(fn(segment args))?  Checked once per
    signature on random data of the same shapes (reduction / cuBLAS configs depend on shapes, not values); if not,
    the per-segment loop runs (which is exactly the reference call: contiguous same-shape segments)."""
    def __init__(self, check=True):
        self.ok, self.check = {}, check
        self.n = collections.Counter()

    @staticmethod
    def _split(args, nseg, i):
        return [a.narrow(0, i * (a.shape[0] // nseg), a.shape[0] // nseg) for a in args]

    def _loop(self, fn, args, nseg):
        return torch.cat([fn(*self._split(args, nseg, i)) for i in range(nseg)])

    @staticmethod
    def _eq(x, y):
        return x.shape == y.shape and x.dtype == y.dtype and torch.equal(x.contiguous().view(torch.uint8),
                                                                           y.contiguous().view(torch.uint8))

    def __call__(self, name, fn, args, nseg):
        if nseg == 1:
            return fn(*args)
        for a in args:
            assert a.is_contiguous() and a.shape[0] % nseg == 0, (name, a.shape)
        key = (name, nseg) + tuple((tuple(a.shape), a.dtype) for a in args)
        ok = self.ok.get(key)
        if ok is None:
            ok = True
            if self.check:
                g = torch.Generator(device=args[0].device); g.manual_seed(len(self.ok) + 1)
                for _ in range(2):
                    ra = [torch.randn(a.shape, generator=g, device=a.device, dtype=a.dtype) for a in args]
                    if not self._eq(fn(*ra), self._loop(fn, ra, nseg)):
                        ok = False; break
            self.ok[key] = ok
        self.n[(name, ok)] += 1
        return fn(*args) if ok else self._loop(fn, args, nseg)

    def report(self):
        bad = sorted({(k[0], k[1], k[2][0][0] // k[1]) for k, v in self.ok.items() if not v})
        return dict(calls={f"{a}:{'batched' if b else 'loop'}": c for (a, b), c in self.n.items()},
                    loop_signatures=[f"{a} nseg={n} T={t}" for a, n, t in bad])


# ================================================================================================ batched LDLQ
class _Member:
    """One projection's encode_rotated state (T12 NE.encode_rotated locals)."""
    def __init__(self, P, meta, dev, lkt_cache):
        Wt = P["weight"]; k, n = Wt.shape
        self.P, self.meta = P, meta
        self.k, self.n, self.tk, self.tn = k, n, k // 128, n // 16
        self.two = "Ln" in P
        self.Ln, self.Dout, self.Din = P.get("Ln"), P.get("Dout"), P["Din"]
        self.Kr = D.res_K_units(meta, meta.get("mask_flat"), dev).view(self.tk, self.tn)
        self.Kr_cpu = self.Kr.cpu()
        self.W4 = Wt.view(self.tk, 128, self.tn, 16)
        self.M = torch.zeros(2, k, n, device=dev)
        self.Q4 = torch.zeros(self.tk, self.tn, 128, 16, device=dev)
        self.Q2 = torch.zeros_like(self.Q4)
        self.sb = torch.zeros(self.tk, self.tn, 8, 256, dtype=torch.int32, device=dev)
        self.var = torch.zeros(self.tk, self.tn, 8, dtype=torch.uint8, device=dev)
        self.sr = torch.zeros(self.tk, self.tn, 8, 256, dtype=torch.int32, device=dev)
        self.Mb = torch.zeros(self.tk, self.tn, dtype=torch.long, device=dev); self.N = torch.zeros_like(self.Mb)
        self.cost2 = torch.zeros(self.tk, self.tn, device=dev); self.cost4 = torch.zeros(self.tk, self.tn, device=dev)
        key = id(P["Lk"])                                       # gate/up share Lk (ldl memo) -> one Lk^T
        if key not in lkt_cache:
            lkt_cache[key] = P["Lk"].T.contiguous()
        self.Lkt = lkt_cache[key]
        self.gsr = P["gsr"]


def _steps(tk, tn, two, dev):
    """NE.encode_rotated schedule + host-side copies (no syncs in the loop)."""
    out = []
    ar128 = torch.arange(128, device=dev); ar16 = torch.arange(16, device=dev)
    if two:
        for s in range(tk + tn - 2, -1, -1):
            a0, a1 = max(0, s - (tn - 1)), min(tk - 1, s)
            a_cpu = torch.arange(a0, a1 + 1); c_cpu = s - a_cpu
            out.append((a_cpu, c_cpu, a0, a1))
    else:
        for a in range(tk - 1, -1, -1):
            out.append((torch.full((tn,), a, dtype=torch.long), torch.arange(tn), a, a))
    res = []
    for a_cpu, c_cpu, a0, a1 in out:
        a_idx, c_idx = a_cpu.to(dev), c_cpu.to(dev)
        rows = a_idx.unsqueeze(1) * 128 + ar128
        cols = c_idx.unsqueeze(1) * 16 + ar16
        res.append(dict(a_cpu=a_cpu, c_cpu=c_cpu, a_idx=a_idx, c_idx=c_idx, T=len(a_cpu), r0=a0 * 128, r1=(a1 + 1) * 128,
                        cols=cols.flatten(), n_hi=int(c_cpu.max()) * 16 + 16))
    return res


@torch.no_grad()
def encode_rotated_batch(Ps, metas, lam=0.3, base_var=None, opts=None, seg=None):
    """[NE.encode_rotated(P, meta, lam, base=None, base_var, inner=0) for P in Ps] (all P of one shape / sidedness),
    with every step's Viterbi work of all members in one launch. Returns the per-member enc builders."""
    opts = opts or DEFAULT_OPTS
    seg = seg or Seg(opts.get("seg_check", True))
    dev = Ps[0]["weight"].device
    R = NE.Ring(dev)
    lkt = {}
    mem = [_Member(P, m, dev, lkt) for P, m in zip(Ps, metas)]
    m0 = mem[0]
    assert all((m.k, m.n, m.two) == (m0.k, m0.n, m0.two) for m in mem)
    tk, tn, k, n, two = m0.tk, m0.tn, m0.k, m0.n, m0.two
    tab = D.variant_table(base_var, dev) if base_var else torch.ones(1, device=dev, dtype=torch.float64)
    tabf = [float(x) for x in (D.variant_table(base_var, "cpu") if base_var else torch.ones(1, dtype=torch.float64))]
    nv = len(tabf)
    steps = _steps(tk, tn, two, dev)
    cchunk = max(1, opts.get("cand_chunk_units", 6144))
    MB = len(mem)
    stream = h._ex().get_quant_stream(dev)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for st in steps:
            a_idx, c_idx, T = st["a_idx"], st["c_idx"], st["T"]
            r0, r1, cols = st["r0"], st["r1"], st["cols"]
            # ---------------------------------------------------------------- A: targets (per member, T12 ops)
            per = []
            shared = {}
            for m in mem:
                Wu = m.W4[a_idx, :, c_idx, :]
                if two:
                    Lsel = m.Lkt[r0:r1].view(T, 128, k)
                    Ms = m.M[:, :, cols].view(2, k, T, 16).permute(0, 2, 1, 3)
                    F2 = torch.bmm(Lsel, Ms[0]); F4 = torch.bmm(Lsel, Ms[1])
                else:
                    Lsel = m.Lkt[r0:r1]
                    F2 = (Lsel @ m.M[0]).view(128, tn, 16).permute(1, 0, 2)
                    F4 = (Lsel @ m.M[1]).view(128, tn, 16).permute(1, 0, 2)
                T2 = Wu + F2; T4 = Wu + F4
                dk_key = id(m.Din)
                if dk_key not in shared:
                    shared[dk_key] = m.Din[a_idx]
                Dk = shared[dk_key]
                Do = m.Dout[c_idx] if two else None
                tb = (1 - lam) * T2 + lam * T4 if lam else T2
                per.append(dict(Wu=Wu, T2=T2, T4=T4, Dk=Dk, Do=Do, tb=tb))
            # ---------------------------------------------------------------- B: base Viterbi, all members x variants
            rings = [R.to_rings(p["tb"]) for p in per]
            nr = rings[0].shape[0]
            st_all = NE.viterbi(torch.cat([r / a for r in rings for a in tabf]), 2)
            # ---------------------------------------------------------------- C: variant pick, residual targets
            resid_jobs = collections.defaultdict(list)          # K -> [(member idx, sel or None, rings)]
            for mi, (m, p) in enumerate(zip(mem, per)):
                tgt = p["tb"]
                tkr = R.to_kring(tgt).double()
                best = None
                for vi in range(nv):
                    a = tab[vi]
                    o = (mi * nv + vi) * nr
                    sk = R.kstates(st_all[o:o + nr])
                    q = D.q2_values(D.hsum(sk), a if base_var else None)
                    mm = (q - tkr).square().sum(-1)
                    if best is None:
                        best, bsk, bq = mm, sk, q
                        bsel = torch.zeros_like(mm, dtype=torch.long)
                    else:
                        w = mm < best
                        best = torch.where(w, mm, best); bsel = torch.where(w, vi, bsel)
                        bsk = torch.where(w.unsqueeze(-1), sk, bsk); bq = torch.where(w.unsqueeze(-1), q, bq)
                av = tab[bsel].unsqueeze(-1) if base_var else None
                q2, sb, vv = R.to_unit(bq), bsk, bsel.to(torch.uint8)
                m.sb[a_idx, c_idx] = sb.int(); m.var[a_idx, c_idx] = vv; m.Q2[a_idx, c_idx] = q2
                amap = R.ring_map(av.squeeze(-1)) if av is not None else torch.ones_like(p["Wu"])
                r = p["T4"] - q2
                rms = r.square().mean((1, 2)).sqrt().clamp_min(1e-12)
                p.update(q2=q2, sb=sb, av=av, amap=amap, r=r)
                Ku = m.Kr_cpu[st["a_cpu"], st["c_cpu"]]
                Ks = sorted(set(Ku.tolist()))
                for K in Ks:
                    if len(Ks) == 1:
                        sel = None
                        s0 = (rms / (NE.CB_RMS * m.gsr[NE._k(K)])).view(-1, 1, 1)
                        rg = R.to_rings(r / (amap * s0))
                    else:
                        sel = (Ku == K).nonzero().flatten().to(dev)
                        s0 = (rms[sel] / (NE.CB_RMS * m.gsr[NE._k(K)])).view(-1, 1, 1)
                        rg = R.to_rings(r[sel] / (amap[sel] * s0))
                    resid_jobs[K].append((mi, sel, rg))
            # ---------------------------------------------------------------- D: residual Viterbi, one launch per K
            srs = [torch.empty(T, 8, 256, dtype=torch.long, device=dev) for _ in mem]
            for K, jobs in resid_jobs.items():
                stK = NE.viterbi(torch.cat([j[2] for j in jobs]), K)
                o = 0
                for mi, sel, rg in jobs:
                    sk = R.kstates(stK[o:o + rg.shape[0]]); o += rg.shape[0]
                    if sel is None:
                        srs[mi] = sk
                    else:
                        srs[mi][sel] = sk
            # ---------------------------------------------------------------- E: ref15 fold candidates, feedback
            for mi, (m, p) in enumerate(zip(mem, per)):
                sr, q2, av, amap, r = srs[mi], p["q2"], p["av"], p["amap"], p["r"]
                Dk, Do, T4, Wu = p["Dk"], p["Do"], p["T4"], p["Wu"]
                Sb = D.hsum(p["sb"]); Sr = D.hsum(sr)

                def mt(E, Dk=Dk, Do=Do):
                    X = torch.bmm(Dk, E)
                    return torch.bmm(X, Do) if Do is not None else X
                ag = amap * R.to_unit(NE.A_K0(Sr))
                Xg = mt(ag)
                dl = ((Xg * r).sum((1, 2)) / (Xg * ag).sum((1, 2)).clamp_min(1e-30)).clamp_min(0)
                Mb, N = NE.mbn_candidates(dl)
                nc = Mb.shape[1]
                cs = []
                per_chunk = max(1, cchunk // T)
                for c0 in range(0, nc, per_chunk):
                    cc = list(range(c0, min(nc, c0 + per_chunk)))
                    J = len(cc)
                    rep = lambda x: x.repeat(J, *([1] * (x.dim() - 1)))
                    q4c = R.to_unit(D.fold(rep(Sb), rep(Sr), Mb[:, cc].T.flatten(), N[:, cc].T.flatten(),
                                           rep(av) if av is not None else None))
                    E4 = (rep(T4) - q4c).contiguous()
                    if Do is not None:
                        fn = lambda E, Dk_, Do_: (E * torch.bmm(torch.bmm(Dk_, E), Do_)).sum((1, 2))
                        cs.append(seg("lcost2", fn, [E4, rep(Dk), rep(Do)], J).view(J, T))
                    else:
                        fn = lambda E, Dk_: (E * torch.bmm(Dk_, E)).sum((1, 2))
                        cs.append(seg("lcost1", fn, [E4, rep(Dk)], J).view(J, T))
                    del q4c, E4
                c = torch.cat(cs)                                   # [nc, T]
                cbest = c[0]; bidx = torch.zeros(T, dtype=torch.long, device=dev)
                for ci in range(1, nc):
                    w = c[ci] < cbest
                    cbest = torch.where(w, c[ci], cbest); bidx = torch.where(w, ci, bidx)
                bMb = Mb.gather(1, bidx.unsqueeze(1)).squeeze(1); bN = N.gather(1, bidx.unsqueeze(1)).squeeze(1)
                q4 = R.to_unit(D.fold(Sb, Sr, bMb, bN, av))
                m.Q4[a_idx, c_idx] = q4; m.sr[a_idx, c_idx] = sr.int()
                m.Mb[a_idx, c_idx] = bMb; m.N[a_idx, c_idx] = bN
                m.cost4[a_idx, c_idx] = cbest                   # (cost2 is info-only and not in the artifact)
                dE2 = Wu - q2; dE4 = Wu - q4
                if two:
                    n_hi = st["n_hi"]
                    Lsub = m.Ln[cols, :n_hi].view(T, 16, n_hi)
                    m.M[0, r0:r1, :n_hi] += torch.bmm(dE2, Lsub).reshape(T * 128, n_hi)
                    m.M[1, r0:r1, :n_hi] += torch.bmm(dE4, Lsub).reshape(T * 128, n_hi)
                else:
                    m.M[0, r0:r1] = dE2.permute(1, 0, 2).reshape(128, n)
                    m.M[1, r0:r1] = dE4.permute(1, 0, 2).reshape(128, n)
            del per, rings, st_all, srs
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    out = []
    for m in mem:
        m.M = None
        out.append(dict(Q2=m.Q2, Q4=m.Q4, sb=m.sb, var=m.var, sr=m.sr, Mb=m.Mb, N=m.N, Kr=m.Kr, cost2=m.cost2,
                        cost4=m.cost4, tk=tk, tn=tn, mismatch=0, base_var=base_var))
    return out


def _full_enc(e):
    k, n = e["tk"] * 128, e["tn"] * 16
    return dict(e, Q2r=e["Q2"].permute(0, 2, 1, 3).reshape(k, n), Q4r=e["Q4"].permute(0, 2, 1, 3).reshape(k, n))


# ================================================================================================ orchestration
@contextlib.contextmanager
def _precomputed_encode_rotated(table):
    """NE.encode_projection calls NE.encode_rotated(P, meta, ...) -> hand it the batched result for P."""
    orig = NE.encode_rotated

    def stub(P, meta, lam=0.3, base=None, base_var=None, inner=0):
        want, e = table.pop(id(P))
        got = (meta["tk"], meta["tn"], meta["shard_axis"], meta["res_rule"], meta.get("mask_flat"), lam, base, base_var, inner)
        assert got == want, (got, want)
        return _full_enc(e)
    NE.encode_rotated = stub
    try:
        yield
    finally:
        NE.encode_rotated = orig


def _proj_setup(W, HG, pn, *, count, sigma, seed, sigma_out, res_K):
    """NE.encode_expert's per-projection prep call + rule (production branch: res_K given, single joint pass)."""
    pi = NE.PROJ.index(pn)
    P = NE.prep(W[pi], HG["H"][pi], count, sigma[pn], seed=seed, G=HG["G"][pi], sigma_out=sigma_out,
                ks=(2, float(res_K[pn])) if res_K else None)
    rule = dict(kind="uniform", K=float(res_K[pn]))
    return P, rule


@torch.no_grad()
def encode_group(items, *, rate=None, count=1, sigma=None, sigma_out=0.03, lam=NE.PROD["lam"], base_var=NE.PROD["base_var"],
                 inner=NE.PROD["inner"], check=True, canonical_base=NE.PROD["canonical_base"], res_K=None, seed=91426,
                 opts=None, seg=None, timings=None):
    """[NE.encode_expert(W, HG, ...)[0] for (W, HG) in items], batched. Production branch only (res_K, single pass,
    inner 0); anything else -> per-expert NE.encode_expert."""
    opts = dict(DEFAULT_OPTS, **(opts or {}))
    sigma = sigma or NE.PROD["sigma"]
    if rate is None and res_K is None:
        res_K = NE.PROD["res_K"]
    tm = timings if timings is not None else collections.defaultdict(float)
    if not opts["batched"] or canonical_base or inner or not res_K:
        outs = []
        for W, HG in items:
            t = time.time()
            outs.append(NE.encode_expert(W, HG, rate=rate, count=count, sigma=sigma, sigma_out=sigma_out, lam=lam,
                                         base_var=base_var, inner=inner, check=check, canonical_base=canonical_base,
                                         res_K=res_K, seed=seed)[0])
            tm["encode_expert"] += time.time() - t
            LDL_MEMO.clear()
        return outs
    seg = seg or Seg(opts.get("seg_check", True))
    G = len(items)
    arts = [dict() for _ in items]; infos = [dict() for _ in items]
    for projs in (("gate", "up"), ("down",)):
        t = time.time()
        Ps, rules = {}, {}
        for b, (W, HG) in enumerate(items):
            for pn in projs:
                Ps[b, pn], rules[b, pn] = _proj_setup(W, HG, pn, count=count, sigma=sigma, seed=seed,
                                                      sigma_out=sigma_out, res_K=res_K)
            LDL_MEMO.clear()
        keys = [(b, pn) for pn in projs for b in range(G)]
        metas = []
        for key in keys:
            P = Ps[key]
            metas.append(dict(tk=P["k"] // 128, tn=P["n"] // 16, shard_axis=NE.PROD["axis"][key[1]],
                              res_rule=rules[key], mask_flat=None))
        torch.cuda.synchronize(); tm["prep"] += time.time() - t; t = time.time()
        encs = encode_rotated_batch([Ps[k_] for k_ in keys], metas, lam=lam, base_var=base_var, opts=opts, seg=seg)
        tm["encode_rotated"] += time.time() - t; t = time.time()
        table = {id(Ps[k_]): ((mt["tk"], mt["tn"], mt["shard_axis"], mt["res_rule"], None, lam, None, base_var, inner), e)
                 for k_, mt, e in zip(keys, metas, encs)}
        del encs
        with _precomputed_encode_rotated(table):
            for key, mt in sorted(zip(keys, metas), key=lambda x: x[0][0]):   # gate, up of one expert adjacent
                b, pn = key
                P = Ps.pop(key)
                t1 = time.time()
                planes, dn, inf, enc, _ = NE.encode_projection(P, shard_axis=mt["shard_axis"], lam=lam, base_var=base_var,
                                                               inner=inner, res_rule=mt["res_rule"])
                if check:
                    rot = D.rotated_levels(planes)
                    inf["bitexact"] = {L: bool(torch.equal(D.decode_matrix(planes, L, rot=rot), dn[L])) for L in (2, 4)}
                    assert all(inf["bitexact"].values()), (pn, inf["bitexact"])
                    del rot
                inf["time"] = time.time() - t1
                arts[b][pn] = planes
                infos[b][pn] = {k: v for k, v in inf.items() if k in ("bits", "proxy_rot", "time", "bitexact",
                                                                     "L2_equal_canonical", "stream_mismatch", "K_frac")}
                del enc, dn
                for kk in list(P.keys()):
                    P[kk] = None
        del table, Ps
        torch.cuda.synchronize(); tm["finalize"] += time.time() - t
    PV.free_tmp(); h.free_scratch()
    for b in range(G):
        info = infos[b]
        r = sum(info[p]["bits"][4] for p in NE.PROJ) / len(NE.PROJ) if res_K else rate
        arts[b]["meta"] = dict(format="nestquant-v1", rate=r, base_var=base_var, lam=lam, inner=inner, sigma=sigma,
                               canonical_base=canonical_base, res_K=dict(res_K) if res_K else None, info=info)
    return arts


def _load(L, E):
    HG, flags = C.load_HG(L, E)
    return C.teacher(L, E), HG, flags


def encode_experts(experts, group=4, opts=None, stats=None, prefetch=True):
    """Yield (L, E, artifact) for every (L, E), `group` experts per batch (production config, C.PROD_KW)."""
    opts = dict(DEFAULT_OPTS, **(opts or {}))
    seg = Seg(opts.get("seg_check", True))
    tm = stats if stats is not None else collections.defaultdict(float)
    groups = [experts[i:i + group] for i in range(0, len(experts), group)]
    with patched(opts):
        for g in groups:
            t = time.time()
            data = [_load(L, E) for L, E in g]
            torch.cuda.synchronize(); tm["load"] += time.time() - t
            arts = encode_group([(W, HG) for W, HG, _ in data], res_K=C.PK, canonical_base=False, inner=0, lam=0.3,
                                opts=opts, seg=seg, timings=tm)
            for (L, E), (_, _, flags), art in zip(g, data, arts):
                art["meta"].update(layer=L, expert=E, flags=flags)
                yield L, E, art
            del data, arts
            torch.cuda.empty_cache()
    tm["seg"] = seg.report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", help="L:E,L:E,...")
    ap.add_argument("--layer", type=int); ap.add_argument("--range", default="0:256")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C.setup()
    ex = [tuple(map(int, s.split(":"))) for s in a.experts.split(",")] if a.experts else \
        [(a.layer, E) for E in range(*map(int, a.range.split(":")))]
    os.makedirs(a.out, exist_ok=True)
    ex = [x for x in ex if not os.path.exists(f"{a.out}/L{x[0]}_E{x[1]}.pt")]
    tm = collections.defaultdict(float)
    t0 = time.time()
    for i, (L, E, art) in enumerate(encode_experts(ex, group=a.group, stats=tm)):
        p = f"{a.out}/L{L}_E{E}.pt"
        torch.save(art, p + ".tmp"); os.replace(p + ".tmp", p)
        print(f"L{L} E{E} {(time.time()-t0)/(i+1):.2f} s/expert amortised", flush=True)
    print(json.dumps({k: v for k, v in tm.items()}, default=str, indent=1))


if __name__ == "__main__":
    main()
