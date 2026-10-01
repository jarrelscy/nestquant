"""nq-res-v2 host-side checks (CPU; no kernel build needed). Spec: NQ_RES_V2.md.
 (a) pack_words / unpack_words round trip, every residual / base code 0..9 (incl. tail-bit codes 7, 9)
 (b) torch reference (moe.lane_vals / dense_W) == ref15_spec.decode_unit on every unit of small random projections,
     base code bk in {0, 1} x residual code rk in {0, 3, 7, 9}, G=4, base-variant signs on, exhaustive (Mb, N) words;
     the base is decoded from the PACKED plane (unpack_words), not from the generator's words
 (c) bk = 0 is the old layout: Proj(bk=0).base == the uint4 record words, proj_sizes base bytes unchanged
Run: python verify_res2.py   (sets a stub 'build' module when the CUDA build is unavailable)"""
import sys, types, torch, numpy as np
try:
    import build  # noqa: F401
    build.get
    if '--cpu' in sys.argv:
        raise ImportError
except Exception:
    sys.modules['build'] = types.SimpleNamespace(get=lambda *a, **k: None, get_sal=lambda *a, **k: None)
import moe
from moe import RKP, rbits, Proj, lane_vals, dense_W, pack_words, unpack_words, proj_sizes
import ref15_spec as R

# (a)
g = torch.Generator().manual_seed(0)
for c in RKP:
    b = rbits(c); nw = (b + 31) // 32
    for nrec in (32, 96, 4 * 32 * 3 + 32):
        w = torch.randint(0, 2**32, (nrec, nw), generator=g, dtype=torch.int64)
        if b % 32: w[:, -1] &= (1 << (b % 32)) - 1
        assert torch.equal(unpack_words(pack_words(w, b), nrec, b), w), (c, nrec)
print(f'(a) pack/unpack round trip: codes {sorted(RKP)} ok', flush=True)

# (b)
nbad = nun = 0
for bk in (0, 1):
    for rk in (0, 3, 7, 9):
        gen = torch.Generator().manual_seed(100 * bk + rk)
        p = Proj(64, 512, gen, None, rk, ('exh', (bk * 10 + rk) * 1000), var=True, bk=bk)   # 4 strips x 4 chunks
        q = types.SimpleNamespace(**{k: getattr(p, k) for k in ('N', 'K', 'rk', 'bk', 'z', 'base', 'p4', 'Mb', 'Nn', 'var', 'fl')})
        lv2 = lane_vals(q, 2, 4); lv4 = lane_vals(q, 4, 4); W4 = dense_W(q, 4, 4)           # q: no bw / p4w -> unpack path
        S_, C_ = p.z['S'], p.z['C']; bb = rbits(bk)
        wb = (p.base.long() & 0xFFFFFFFF).view(-1, 4) if bk == 0 else p.bw
        sg = 1 - 2 * ((p.var.long().view(S_, C_, 1) >> torch.arange(8)) & 1)                 # ring signs [S, C, 8]
        for s in range(S_):
            for c in range(C_):
                rec = (s * C_ + c) * 32
                bs = R.rings_from_lane_words(wb[rec:rec + 32].numpy(), bb)
                rs = R.rings_from_lane_words(p.p4w[rec:rec + 32].numpy(), rbits(rk))
                Mb, N = int(p.Mb[s * C_ + c]), int(p.Nn[s * C_ + c])
                Q2, Q4 = R.decode_unit(bs, rs, Mb, N, Kb=RKP[bk], Kr=RKP[rk])
                a = sg[s, c].double().numpy()[:, None]; Q2, Q4 = a * Q2, a * Q4
                t2 = lv2[s, c].reshape(8, 256).double().numpy(); t4 = lv4[s, c].reshape(8, 256).double().numpy()
                u = R.to_unit(Q4).T; d = W4[s * 16:(s + 1) * 16, c * 128:(c + 1) * 128].double().numpy()
                ok = np.array_equal(Q2, t2) and np.array_equal(Q4, t4) and np.array_equal(u, d); nbad += not ok; nun += 1
print(f'(b) torch ref vs ref15_spec: {nun} units x 2 levels, bk {{0,1}} x rk {{0,3,7,9}}, mismatching units {nbad}', flush=True)
assert nbad == 0

# (c)
gen = torch.Generator().manual_seed(7); p0 = Proj(64, 512, gen, None, 7)
gen = torch.Generator().manual_seed(7); p1 = Proj(64, 512, gen, None, 7, bk=0)
assert torch.equal(p0.base, p1.base) and torch.equal(p0.p4, p1.p4)
w = (p0.base.long() & 0xFFFFFFFF).view(-1, 4)
assert torch.equal(pack_words(w, 128), p0.base) and proj_sizes(64, 512)['base'] == p0.base.numel() * 4
assert proj_sizes(64, 512, bk=1)['base'] == 4 * 4 * 32 * 112 // 8
print('(c) bk = 0 layout unchanged (uint4 records == 128-bit sub-array layout); base bytes K=1.75 = 112/128 of K=2', flush=True)
print('verify_res2: ALL OK')
