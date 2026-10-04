"""step 3b c5s live-port parity (CPU only): replay offline streams through the LIVE code path (streaming/c5s.py: tfcap
row finalizer with synthetic MTP ns=3 verify steps incl. garbage rejected-draft rows -> incremental features -> C3k
forward (batch 1) -> blend with the stream's jF replay S[b]) and compare with the p-salnet export
(/data/Jarrel/nq-step3b/p-salnet/export/c5s/<stream>.npy, made by feats.build + batched forward + export.py).

Reports per stream: max / mean |log diff| of the score (rows b >= 1), the re-simmed hot count/sal % (p-salnet/fsim.py,
canonical budget sim, B per stream) for the export and for the live replay (zero lag = mC of refresh b; lag1 = mC of
refresh b-1, i.e. one refresh = 16 rows late), and jF (jF_lag1 = S[b-1], the same lag for the prod score). PASS = |live - export| <= 0.05 on both count and sal hot %.

usage (via /data/Jarrel/coord/memjob.sh 20 /data/Jarrel/nq-algo/venv/bin/python tests/test_c5s_parity.py [spec...]):
  spec = kld:3 | gen:<id> | tb:<task>   (default: kld:3 gen:biology_zh tb:cad-model)
env: C5S_CKPT (default streaming/ckpt/C3k.pt), NT torch threads (default 4)"""
import os, sys, time, resource, numpy as np
os.environ.setdefault('NT', '4')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO + '/streaming')
SAL = '/data/Jarrel/nq-step3b/p-salnet'; sys.path.insert(0, SAL)
import torch
import evalsal as E, fsim, c5s as C5

CK = os.environ.get('C5S_CKPT', REPO + '/streaming/ckpt/C3k.pt')
specs = sys.argv[1:] or ['kld:3', 'gen:biology_zh', 'tb:cad-model']
rng = np.random.default_rng(0)


def replay(st, P):
    """feed the stream to P (c5s.C5S) as MTP ns=3 verify steps; returns mC [nb+1,75,256] f32 (row b = refresh at 16b) and
    the order of emitted rows (must equal 0..N-1)"""
    ids, N = st['ids'], len(st['ids'])
    w = np.asarray(st['w_'], np.float16); xn = np.asarray(st['xn_'], np.float32); hp = np.asarray(st['hp'], np.float16)
    tok = np.asarray(st['tok_'])
    rs = list(np.asarray(st['rstart'])) + [N]; nb = N // 16
    mC = np.full((nb + 1, 75, 256), np.nan, np.float32)
    seen = []; e0 = P.R.emit; sal0 = st['sal']
    def emit(i, s_, h, t):                       # the n-th finalized row must be source row n (no garbage, no gaps)
        n = len(seen); seen.append(n if (n < N and np.array_equal(i, ids[n]) and np.array_equal(s_, sal0[n])
                                         and np.array_equal(h, hp[n])) else -1)
        e0(i, s_, h, t)
    P.R.emit = emit
    for q in range(len(rs) - 1):
        a, b_ = int(rs[q]), int(rs[q + 1])
        if st['pfc'] is not None: P.R.prefill(q, st['pfn'][q], st['pfc'][q])
        P.R.step(q, True, np.arange(1, 5), np.zeros(4, np.int32), None, None, None, None)   # short prefill tail step: dropped
        p = 0                                    # next position to compute (row a + p, position p + 1)
        while p < b_ - a:
            T = min(4, b_ - a - p); acc = int(rng.integers(0, T))   # accepted drafts beyond the first row
            rows = []
            for k in range(T):
                r = a + p + k
                if k <= acc: rows.append(r)
                else: rows.append(-1)            # rejected draft: garbage row, superseded by the next step
            pos = np.array([p + 1 + k for k in range(T)])
            I = np.empty((T, 75, 8), np.int64); W = np.empty((T, 75, 8), np.float16); X = np.empty((T, 75), np.float32)
            H = np.empty((T, 8, 256), np.float16); TK = np.empty(T, np.int64)
            for k, r in enumerate(rows):
                if r >= 0: I[k] = ids[r]; W[k] = w[r]; X[k] = xn[r]; H[k] = hp[r]; TK[k] = tok[r]
                else:
                    I[k] = rng.integers(0, 256, (75, 8)); W[k] = 1; X[k] = 1e6; H[k] = 99; TK[k] = 154842
            P.R.step(q, False, pos, TK, I, W, X, H)
            while P.ready:
                t = P.F.pend[0]; P.run_pending(); mC[t // 16] = P.cur[0]
            p += acc + 1
    P.R._flush()
    while P.ready:
        t = P.F.pend[0]; P.run_pending(); mC[t // 16] = P.cur[0]
    return mC, seen


def load(spec):
    st = E.load(spec); kind, key = spec.split(':', 1); N = len(st['ids'])
    if kind == 'tb':
        z = np.load(f'/rawdata/Jarrel/nq-tfpred/ds/cap-{key}.npz', mmap_mode='r')
        st['w_'], st['xn_'], st['tok_'] = z['w'], z['xn'], z['tok']; st['pfc'] = np.asarray(z['pf']); st['pfn'] = np.asarray(z['pfn'])
    else:
        f = f'/data/Jarrel/nq-step3/data/gen/{key}.npz' if kind == 'gen' else f'/data/Jarrel/nq-step3/data/kldD_w{key}_r0.npz'
        z = np.load(f, mmap_mode='r'); st['w_'], st['xn_'], st['tok_'] = z['w'], z['xn'], z['tok']
        if kind == 'gen':
            pf = np.asarray(z['pf']); st['pfc'] = pf[None]; st['pfn'] = [int(pf[0].sum()) // 8]
        else:
            st['pfc'] = None                    # kldD streams were featurised with pf = None (feature 53 = 0)
    return st


def main():
    ok_all = True; out = []
    for spec in specs:
        t0 = time.time(); st = load(spec); N = len(st['ids']); nb = N // 16; S = st['S']
        # sanity: the offline sal equals w^2 xn computed the live way
        sl = np.asarray(st['w_'][:64], np.float32) ** 2 * np.asarray(st['xn_'][:64], np.float32)[:, :, None]
        assert np.array_equal(sl, st['sal'][:64]), 'sal convention mismatch'
        P = C5.C5S(CK, threads=int(os.environ['NT']), maxlag=10 ** 9)
        t1 = time.time(); mC, seen = replay(st, P); t2 = time.time()
        assert seen == [r for r in range(N)], ('row finalizer order', seen[:20])
        bs = [b for b in range(1, nb) if not np.isnan(mC[b, 0, 0])]
        assert bs == list(range(1, nb)) or bs == list(range(1, nb + 1)), (bs[:5], bs[-5:], nb)
        mC[nb] = mC[nb] if not np.isnan(mC[nb, 0, 0]) else mC[nb - 1]
        live = np.zeros((nb + 1, 75, 256), np.float32); lag = np.zeros_like(live)
        for b in range(1, nb + 1):
            live[b] = C5.blend(S[b], mC[b]); lag[b] = C5.blend(S[b], mC[max(1, b - 1)])
        ex = np.load(f'{SAL}/export/c5s/{st["name"]}.npy').astype(np.float32)
        d = np.abs(np.log(np.maximum(live[1:nb], 1e-30)) - np.log(np.maximum(ex[1:nb], 1e-30)))
        # also the raw model output vs the offline score cache
        sc = f'{E.SC}/C3k_{st["name"]}.npy'; dm = None
        if os.path.exists(sc):
            O = np.load(sc, mmap_mode='r'); dm = np.abs(mC[1:nb] - np.asarray(O[1:nb, :, :, 0], np.float32))
        ids, sal = st['ids'], st['sal']
        r = {}
        for nm, f in (('export', lambda b: ex[b]), ('live', lambda b: live[b]), ('live_lag1', lambda b: lag[b]),
                      ('jF', lambda b: np.asarray(S[b], np.float32)), ('jF_lag1', lambda b: np.asarray(S[max(1, b - 1)], np.float32))):
            H, u = fsim.sim(ids, sal, f, B=st['B']); r[nm] = fsim.report(H, sal)['all'][:2] + (u,)
        dc, ds = abs(r['live'][0] - r['export'][0]), abs(r['live'][1] - r['export'][1]); ok = dc <= 0.05 and ds <= 0.05
        ok_all &= ok
        line = (f"{st['name']}: N {N} refreshes {len(bs)} | score |dlog| max {d.max():.4f} mean {d.mean():.2e}"
                + (f" | mC |d| max {dm.max():.4f} mean {dm.mean():.2e}" if dm is not None else '')
                + ' | hot count/sal: ' + ' '.join(f'{k} {v[0]:.2f}/{v[1]:.2f}' for k, v in r.items())
                + f" | d {dc:.3f}/{ds:.3f} {'PASS' if ok else 'FAIL'}"
                + f" | replay {t2 - t1:.1f}s: blk {1e3 * P.st['t_blk'] / max(1, P.st['refresh']):.2f} ms/refresh,"
                  f" feat {1e3 * P.st['t_feat'] / max(1, P.st['fwd']):.2f} ms, fwd {1e3 * P.st['t_fwd'] / max(1, P.st['fwd']):.2f} ms"
                  f" ({os.environ['NT']} thr) | {time.time() - t0:.0f}s")
        print(line, flush=True); out.append(line)
    print('maxrss %.0f MB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print('ALL PASS' if ok_all else 'SOME FAIL')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
