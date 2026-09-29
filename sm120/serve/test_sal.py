"""GBDT x mps128 serve wiring tests.
CPU (default): scheduler predictor selection (NQ_GBDT_SCALE / NQ_GBDT_MODE / NQ_GBDT_BAND), sal pass-through
  (Scheduler.step(sal=) == GBDTPredictorV2.step(sal=) score matrices), V2 scale=None == V1, refresh cost at 75 layers.
GPU (--gpu): nqsal.cu vs a torch fp64 reference (T 1..8, duplicates, w == 0, bf16/fp16, strided rows) + CUDA-graph replay.
  run: PYTHONPATH=/data/Jarrel/nq-dev/pylgb /data/Jarrel/nqenv/bin/python sm120/serve/test_sal.py [--gpu]"""
import os, sys, time, json
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [R + '/streaming', R + '/sm120']
ROUT = '/data/Jarrel/nq-eval/results/c2-gbdt-cap24_routing'


def stream(n_layers=16, T=4096, seed=0):
    Ls = sorted(int(f[1:-4]) for f in os.listdir(ROUT))[:n_layers]
    g = np.stack([np.load(f'{ROUT}/L{L}.npy')[:T].astype(np.int64) for L in Ls])        # [NL, T, 8]
    rng = np.random.default_rng(seed); em = rng.lognormal(0, 1, (len(Ls), 256))
    sv = rng.dirichlet(np.ones(8), g.shape[:2]) ** 2 * 6.25 * rng.lognormal(3, .3, g.shape[:2])[..., None] \
        * np.take_along_axis(em[:, None, :].repeat(g.shape[1], 1), g, 2)
    return Ls, g, sv


def fixed_for(Ls):
    import fixed_set as FS
    fx, _, _ = FS.load(layers=Ls); return fx


def sched(Ls, fx, **env):
    import scheduler as SC
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        dflt = {L: [e for e in range(256) if e not in fx[L]][:51] for L in Ls}
        return SC.Scheduler(Ls, fx, dflt, 1 << 20, NE=256, n_float=51, slots=56 * len(Ls), cap_GBps=1e6, predictor='gbdt')
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def cpu():
    from gbdt_predictor import GBDTPredictor
    from gbdt_predictor_v2 import GBDTPredictorV2
    Ls, g, sv = stream(); fx = fixed_for(Ls); NL, T, _ = g.shape
    # 1. selection
    for env, cls, ws, mode, rhi in ((dict(NQ_GBDT_SCALE='none'), GBDTPredictor, False, 'next_refresh', 121),
                                    (dict(NQ_GBDT_SCALE='mps'), GBDTPredictorV2, True, 'next_refresh', 121),
                                    (dict(NQ_GBDT_SCALE='mps', NQ_GBDT_MODE='sync', NQ_GBDT_BAND='all'), GBDTPredictorV2, True, 'sync', 256)):
        S = sched(Ls, fx, **env)
        assert type(S.P) is cls and S.wants_sal == ws and S.P.mode == mode and S.P.rhi == rhi, (env, type(S.P), S.wants_sal, S.P.mode, S.P.rhi)
        assert cls is GBDTPredictor or S.P.scale == 'mps'
        S.P.close()
    print('selection OK (none -> V1, mps -> V2, mode/band env)')
    # 2. Scheduler pass-through == direct V2, 3. V2 scale=None == V1   (sync mode: deterministic, same boundary)
    S = sched(Ls, fx, NQ_GBDT_SCALE='mps', NQ_GBDT_MODE='sync')
    P = GBDTPredictorV2(Ls, fx, scale='mps', mode='sync', num_threads=4)
    A = GBDTPredictor(Ls, fx, mode='sync', num_threads=4); B = GBDTPredictorV2(Ls, fx, scale=None, mode='sync', num_threads=4)
    step = 3; nref = 0; dmax = 0.; dab = 0.
    for t0 in range(0, T, step):
        t1 = min(T, t0 + step)
        c = np.stack([np.bincount(g[i, t0:t1].ravel(), minlength=256) for i in range(NL)]).astype(np.float64)
        sal = np.stack([np.bincount(g[i, t0:t1].ravel(), weights=sv[i, t0:t1].ravel(), minlength=256) for i in range(NL)])
        S.step(c, t1 - t0, sal=sal); r = P.step(c, t1 - t0, sal=sal); A.step(c, t1 - t0); B.step(c, t1 - t0, sal=sal)
        if r:
            nref += 1; dmax = max(dmax, float(np.abs(S.P.S - P.S).max())); dab = max(dab, float(np.abs(A.S - B.S).max()))
    assert nref > 100 and dmax == 0 and dab == 0, (nref, dmax, dab)
    assert not np.array_equal(P.S, A.S)
    print(f'pass-through OK ({nref} refreshes: Scheduler(sal=) == V2 direct exactly; V2 scale=None == V1 exactly; mps != none)')
    for x in (S.P, P, A, B): x.close()
    # 4. refresh cost at 75 layers (serve: 4 threads). Rows: 75 x 101 (band ema256) or 75 x 236 (band all)
    L75 = list(range(3, 78)); fx75 = fixed_for(L75)
    rng = np.random.default_rng(1)
    for scale, band in ((None, 'ema256'), ('mps', 'ema256'), ('mps', 'all')):
        kw = dict(rhi=256) if band == 'all' else {}
        Q = (GBDTPredictorV2(L75, fx75, scale=scale, mode='sync', num_threads=4, **kw) if scale else GBDTPredictor(L75, fx75, mode='sync', num_threads=4, **kw))
        ts = []; tst = []
        for b in range(40):
            for k in range(5):
                c = rng.poisson(0.1, (75, 256)).astype(np.float64) * 3; s = c * rng.lognormal(3, 1, (75, 256))
                t = time.perf_counter(); r = Q.step(c, 3, sal=s) if scale else Q.step(c, 3); dt = time.perf_counter() - t
                (ts if r else tst).append(dt)
        print(f'75 layers scale={scale} band={band}: sync refresh {np.median(ts) * 1e3:.2f} ms (p90 {np.percentile(ts, 90) * 1e3:.2f}), '
              f'non-boundary step {np.median(tst) * 1e6:.0f} us')
        Q.close()


def gpu():
    import torch, build
    m = build.get_sal(); dev = torch.device('cuda')
    rng = np.random.default_rng(0); worst = 0.
    for dt in (torch.bfloat16, torch.float16):
        for T in range(1, 9):
            for trial in range(20):
                H = 6144 if trial % 4 else 6000                  # 6000: non-vectorised path
                big = torch.randn(T, H + 64, device=dev).to(dt); x = big[:, :H] if trial % 3 == 0 else big[:, :H].contiguous()
                ids = torch.from_numpy(rng.integers(0, 12 if trial % 2 else 256, (T, 8))).to(dev)   # few experts -> many dups
                w = torch.rand(T, 8, device=dev).half(); w[0, 0] = 0 if trial % 5 == 0 else w[0, 0]
                acc = torch.rand(256, dtype=torch.float64, device=dev) * 10; acc0 = acc.clone()
                host = torch.zeros(256, dtype=torch.float64).pin_memory()
                m.sal(x, w, ids, 2.5, acc, host.data_ptr()); torch.cuda.synchronize()
                xn = x.float().pow(2).sum(-1).double()
                v = (w.float() * 2.5).pow(2).double() * xn[:, None]; v[w == 0] = 0
                ref = acc0.clone().index_add_(0, ids.reshape(-1), v.reshape(-1))
                err = float(((acc - ref).abs() / ref.abs().clamp_min(1e-9)).max()); worst = max(worst, err)
                touched = torch.unique(ids.reshape(-1)[(w != 0).reshape(-1)]).cpu()
                assert err < 1e-5, (dt, T, trial, err)
                assert torch.equal(host[touched], acc.cpu()[touched]), 'host mirror'
    print(f'nqsal == fp64 reference (max rel err {worst:.2e}), host mirror exact')
    # graph capture + replay accumulates
    x = torch.randn(3, 6144, device=dev).bfloat16(); w = torch.rand(3, 8, device=dev).half()
    ids = torch.randint(0, 256, (3, 8), device=dev); acc = torch.zeros(256, dtype=torch.float64, device=dev)
    host = torch.zeros(256, dtype=torch.float64).pin_memory(); s = torch.cuda.Stream()
    with torch.cuda.stream(s): m.sal(x, w, ids, 2.5, acc, host.data_ptr())
    torch.cuda.synchronize(); one = acc.clone(); gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph): m.sal(x, w, ids, 2.5, acc, host.data_ptr())
    for _ in range(9): gph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(acc, one * 10, rtol=1e-12) and torch.allclose(host.to(dev)[acc != 0], acc[acc != 0]), 'graph replay'
    for _ in range(3): gph.replay()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(2000): gph.replay()
    torch.cuda.synchronize(); print(f'graph replay OK; {1e6 * (time.perf_counter() - t) / 2000:.2f} us per call (incl. replay launch)')


if __name__ == '__main__':
    gpu() if '--gpu' in sys.argv else cpu()
