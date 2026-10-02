"""nq-algo sim trace (traces/<name>.npz ex/tok/nr + .ans.npy + .pf.npz/.pfid.npy) -> /rawdata/Jarrel/nq-tfpred/ds/ids-<name>.npz
(prep_ids layout) so export_fc forecasts are block-aligned with nq-algo's sim on the same trace.  EVAL ONLY (e.g. satb held-out).
  prep_from_nqalgo.py satb"""
import sys, numpy as np
T = '/data/Jarrel/nq-algo/traces'; n = sys.argv[1]
z = np.load(f'{T}/{n}.npz'); ex = z['ex']; nr = z['nr'].astype(bool); N = len(ex)
ans = np.load(f'{T}/{n}.ans.npy').astype(bool)
assert nr[0]
rq = (np.cumsum(nr) - 1).astype(np.int32); rs = np.nonzero(nr)[0]; R = len(rs)
pz = np.load(f'{T}/{n}.pf.npz'); pos, PT = pz['pos'], pz['T']; ids = np.load(f'{T}/{n}.pfid.npy', mmap_mode='r')
pf = np.zeros((R, 75, 256), np.int32); pfn = np.zeros(R, np.int32); a = 0
for p, t in zip(pos, PT):
    r = int(np.searchsorted(rs, p, 'left'))            # chunk at decode index p belongs to the request starting at rs[r] >= p
    r = min(r, R - 1)
    x = np.asarray(ids[a:a + t], np.int64).reshape(t, 75, 8) % 256; a += t
    for L in range(75): pf[r, L] += np.bincount(x[:, L].ravel(), minlength=256)
    pfn[r] += t
assert a == len(ids)
think = ~ans
np.savez(f'/rawdata/Jarrel/nq-tfpred/ds/ids-{n}.npz', ex=ex, tok=z['tok'], think=think, rq=rq, pf=pf, pfn=pfn, rstart=rs.astype(np.int64))
print(n, 'rows', N, 'reqs', R, 'pfn', pfn.tolist(), 'think %.3f' % think.mean())
