"""nq-tapc parity: NQ_SCHED=tap with NQ_HOSTLOOP=cpp (scheduler_tap.TapScheduler._step_cpp, nqhost.TapCore) vs the
Python TapScheduler.step. Needs no GPU.
  python tests/test_tap_parity.py            all cases       (NFILES routing-log files, default 60)
  python tests/test_tap_parity.py bench      ms/iter py vs cpp on decode steps (predictor cost excluded: replayed S)
Two TapSchedulers (hostloop='py' / 'cpp') run in lockstep on per-step routing counts from the GLM-5.3 routing logs
(test_hostloop_parity.routing_steps), a deterministic fake clock (tok/s with jitter) and a fake io_all() (4 ranks,
varying delivered rate / outstanding / slot wait / backlog, sometimes empty), and identical fake executors (landed /
released / failed incl. read errors; test_hostloop_parity.LeaderX). Per step: ups and downs identical (same order);
score / state / want / hold / doomed bitwise; the lazy-eviction list (doom), the no-slot queue (todo), the stats and
the rate state equal. At every refresh the value arrays V and V0 are compared bitwise as well. Predictors: stub
score matrices (float32 ties, all-zero layers, NaN entries, negative scores) and the jF joint predictor on CPU,
recorded on the py side and replayed on the cpp side. Configs cover the serve k0 layout, a fixed set, slot-bound
(no free slot: todo queue + eager eviction), budget-bound (MLA), adaptive H (HA), far=tail, pin and kv_pressure."""
import os, sys, random, time, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../streaming'); sys.path.insert(0, HERE)
import numpy as np
import scheduler_tap as TS
from test_hostloop_parity import routing_steps, LeaderX, cfgs, LAYERS, NE

TAPENV = ('NQ_TAP_C', 'NQ_TAP_H', 'NQ_TAP_HA', 'NQ_TAP_MLA', 'NQ_TAP_SVC_MS', 'NQ_TAP_FAR', 'NQ_TAP_TP', 'NQ_TAP_RATE_GBPS',
          'NQ_TAP_LAT', 'NQ_TAP_LAT_HL', 'NQ_TAP_LAT_Q', 'NQ_TAP_LAT_N', 'NQ_S3_ADM_CEMA', 'NQ_S3_ADM_HL', 'NQ_S3_LW_POW',
          'NQ_S3_MUL_POW', 'NQ_S3_MUL_HL', 'NQ_S3_LW_HL', 'NQ_S3_MASS', 'NQ_S3_TRACK', 'NQ_TAP_CTL')


class StubS:
    """predictor with a score matrix S refreshed every 16 tokens: float32 ties, all-zero layers, NaN, negatives"""
    bs = None
    def __init__(s, seed, nan=True):
        s.rng = np.random.default_rng(seed); s.n = 0; s.S = None; s.nan = nan; s.k = 0
    def step(s, c, ntok, tid, nr, sal=None):
        s.n += ntok
        if s.n < 16: return False
        s.n = 0; s.k += 1; r = s.rng
        S = np.round(r.gamma(0.4, 3.0, (75, NE)) * 4) / 4                 # many exact ties (multiples of 0.25)
        S += 3 * c[:, :] / max(1, c.sum() / 75)
        S[r.random((75, NE)) < 0.02] *= -1
        S[r.random(75) < 0.05] = 0                                           # layers with no mass
        if s.nan and s.k % 7 == 0: S[r.random((75, NE)) < 0.003] = np.nan
        s.S = S.astype(np.float32 if s.k % 2 else np.float64); return True
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


class RecS:
    """py side: real predictor, recording step() results and S"""
    def __init__(s, P): s.P = P; s.log = collections.deque(); s.bs = getattr(P, 'bs', None)
    @property
    def S(s): return s.P.S
    def step(s, *a, **k):
        r = s.P.step(*a, **k); s.log.append((r, None if s.P.S is None else np.array(s.P.S, copy=True))); return r
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


class PlayS:
    def __init__(s, R): s.R = R; s.bs = R.bs; s.S = None
    def step(s, *a, **k):
        r, S = s.R.log.popleft(); s.S = S; return r
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


def jf_pair():
    sys.path.insert(0, '/data/Jarrel/nq-serve/predictor/joint')
    import gpu_predictor as GJ, torch
    torch.set_num_threads(int(os.environ.get('NQ_TEST_THREADS', '8')))
    P = GJ.GPUJointPredictor(LAYERS, {L: [] for L in LAYERS}, '/data/Jarrel/nq-serve/predictor/joint/jF.pt', n_float=77, hm=0.7,
                             device='cpu', v2_model='/data/Jarrel/nq-serve/predictor/joint/v2_sal_tweedie1.5.txt')
    R = RecS(P); return R, PlayS(R)


class Clock:
    def __init__(s, seed): s.t = 1000.0; s.rng = random.Random(seed)
    def adv(s, ntok): s.t += ntok / 94.0 * (0.5 + s.rng.random()) + (0.0 if s.rng.random() < 0.9 else 0.05)
    def __call__(s): return s.t


def fake_io(seed):
    rng = random.Random(seed); n = [0]
    def io_all():
        n[0] += 1; r = rng.random()
        if r < 0.05: return {}
        if r < 0.08: raise OSError('stale publish')
        return {k: dict(delivered_GBps=rng.choice([0.0, None, rng.random() * 9]), ops_outstanding=rng.randint(0, 300),
                        slot_wait=rng.choice([0, None, rng.randint(0, 50)]), backlog=rng.randint(0, 400) if k else 0,
                        landed_total=n[0] * 7 + k, op_p50_ms=rng.choice([None, float('nan'), rng.random() * 900, rng.random() * 90]), t_wall=1e9 + n[0])
                for k in range(4)}
    return io_all


def eq(a, b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def mk(cfg, P, hl, env, clock):
    old = {k: os.environ.get(k) for k in TAPENV}
    try:
        for k in TAPENV: os.environ.pop(k, None)
        os.environ['NQ_TAP_CTL'] = '/nonexistent/nq_tap_ctl'     # never a live server's /dev/shm ctl file
        os.environ.update(env)
        S = TS.TapScheduler(LAYERS, cfg['fixed'], cfg['dflt'], 2560000, NE=NE, n_float=cfg['nf'], slots=cfg['slots'],
                            cap_GBps=cfg['cap'], predictor=P, hostloop=hl, clock=clock)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return S


def tap_parity(name, steps, mkp, cfg, env, seed=0, kv=False, pin=False, io=True, sal=False):
    pa, pb = mkp()
    CA, CB = Clock(seed), Clock(seed)
    A = mk(cfg, pa, 'py', env, CA); B = mk(cfg, pb, 'cpp', env, CB)
    assert A.tcore is None and A.core is None and B.tcore is not None
    if io: A.io_all = fake_io(seed); B.io_all = fake_io(seed)
    vcap = {}
    v0 = A._value
    def cap(lat, H):
        V, V0 = v0(lat, H); vcap['V'] = V; vcap['V0'] = V0; return V, V0
    A._value = cap
    XA, XB = LeaderX(seed), LeaderX(seed)
    init = [(L, E) for L in LAYERS for E in cfg['dflt'][L] if E not in cfg['fixed'][L]][:cfg['slots']]
    for S, X in ((A, XA), (B, XB)):
        for L, E in init: S.state[S.li[L], E] = 1
        X.apply(init, [])
    rng = random.Random(seed + 5); nops = 0; tA = tB = 0.0; nref = 0; srng = np.random.default_rng(seed + 9)
    lf = srng.gamma(2.0, 1.0, (75, 1))                    # sal=True: per-step salience = counts x layer factor x noise
    for n, (c, ntok, tid, nr) in enumerate(steps):
        c = c.astype(np.float64); CA.adv(ntok); CB.adv(ntok); vcap.clear(); r0 = A.stats['refreshes']
        kw = {}
        if sal and n >= 40: kw['sal'] = c * lf * srng.gamma(1.0, 1.0, c.shape)
        t = time.perf_counter(); ua, da = A.step(c, ntok, tid, nr, **kw); t2 = time.perf_counter(); ub, db = B.step(c, ntok, tid, nr, **kw); t3 = time.perf_counter()
        tA += t2 - t; tB += t3 - t2
        if (ua, da) != (ub, db):
            print(f'  {name}: OPS DIFF at step {n}: ups {len(ua)}/{len(ub)} downs {len(da)}/{len(db)} first ups {ua[:3]} {ub[:3]}'); return False
        for f in ('score', 'state', 'want', 'hold', 'doomed'):
            if not eq(getattr(A, f), getattr(B, f)): print(f'  {name}: {f} DIFF at step {n}'); return False
        if A.stats['refreshes'] != r0:
            nref += 1
            Ve = vcap['V'] if A.pin is None else np.where(A.pin & ~A.fixed, vcap['V'].dtype.type(1e9), vcap['V'])   # pin applied after _value
            if not (eq(Ve, B.tcore.V()) and eq(vcap['V0'], B.tcore.V0())):
                Vb = B.tcore.V(); d = np.argwhere(Ve != Vb) if Ve.dtype == Vb.dtype else 'dtype %s vs %s' % (Ve.dtype, Vb.dtype)
                V0a, V0b = vcap['V0'], B.tcore.V0()
                print(f'  {name}: VALUE DIFF at step {n} (refresh {nref}): V {d if isinstance(d, str) else (len(d), d[:3].tolist())} '
                      f'V eq {eq(Ve, Vb)} {Ve.dtype}/{Vb.dtype}; V0 eq {eq(V0a, V0b)} {V0a.dtype}/{V0b.dtype} '
                      f'V0 diff {int((V0a != V0b).sum()) if V0a.dtype == V0b.dtype else "-"}; lat {A.stats["lat_tok_sum"]}'); return False
        dl = [((i, e), (j, v)) for (i, e), (j, v) in A.doom.items()]
        if dl != B.tcore.doom_list() or list(A.todo) != B.tcore.todo_list():
            print(f'  {name}: doom/todo DIFF at step {n}: {len(dl)}/{len(B.tcore.doom_list())} {len(A.todo)}/{B.tcore.todo_len()}'); return False
        pk = (A.peak is None and B.peak is None) or (A.peak is not None and B.peak is not None and eq(A.peak, B.peak))
        sa = {k: v for k, v in A.stats.items() if not k.startswith('q_')}   # q_*: nq-kld queue-model diagnostics, py path only
        if sa != B.stats or A.lat_ema != B.lat_ema or A.tok != B.tok or A.tps != B.tps or A.n_land != B.n_land or not pk:
            print(f'  {name}: scalar DIFF at step {n}: stats {A.stats} {B.stats}'); return False
        XA.apply(ua, da); XB.apply(ub, db); XA.tick(A); XB.tick(B); nops += len(ua) + len(da)
        if kv and rng.random() < 0.01:
            k = rng.randint(1, 40)
            if A.kv_pressure(k) != B.kv_pressure(k): print(f'  {name}: kv_pressure DIFF at step {n}'); return False
        if pin and n == len(steps) // 3:
            pm = np.random.default_rng(seed).random((75, NE)) < 0.05; A.pin = pm.copy(); B.pin = pm.copy()
        if pin and n == 2 * len(steps) // 3: A.pin = B.pin = None
    st = {k: A.stats[k] for k in ('ups', 'downs', 'refreshes', 'promotions', 'budget_cut', 'eager_evict', 'no_slot_skip', 'big_steps')}
    print(f'  {name:34s} IDENTICAL over {len(steps)} steps, {nref} refreshes, {nops} ops, {st}; step() py {1e3*tA/len(steps):.3f} '
          f'cpp {1e3*tB/len(steps):.3f} ms/iter (incl. predictor)', flush=True)
    return True


def main():
    nf = int(os.environ.get('NFILES', '60'))
    steps = routing_steps(nf); print(f'tap parity: {len(steps)} model steps, {sum(s[1] for s in steps)} tokens from {nf} routing-log files', flush=True)
    k0, b = cfgs()
    k0s = dict(k0, slots=5200)                       # fewer slots than 75 x 77: no-slot paths (todo, eager eviction)
    ok = True
    C = [('stub k0 default env', k0, {}, dict(seed=1)),
         ('stub k0 slot-bound c1 kv+pin', k0s, dict(NQ_TAP_C='1'), dict(seed=2, kv=True, pin=True)),
         ('stub fixed+budget Ha2 mla0.5', b, dict(NQ_TAP_HA='2', NQ_TAP_MLA='0.5'), dict(seed=3, kv=True)),
         ('stub k0 far=ema c0.3 no-io', k0, dict(NQ_TAP_FAR='ema', NQ_TAP_C='0.3'), dict(seed=4, io=False)),
         ('stub k0s mla0 (no budget) H512', k0s, dict(NQ_TAP_MLA='0', NQ_TAP_H='512'), dict(seed=5, pin=True)),
         ('stub k0s lat=meas', k0s, dict(NQ_TAP_LAT='meas'), dict(seed=6, kv=True)),
         ('stub k0 lat=meas Ha2 q0.9 hl2 c0.5', k0, dict(NQ_TAP_LAT='meas', NQ_TAP_HA='2', NQ_TAP_LAT_Q='0.9', NQ_TAP_LAT_HL='2',
                                                    NQ_TAP_C='0.5'), dict(seed=8, pin=True)),
         ('stub k0s s3 arm c0 cema.2 lw.25', k0s, dict(NQ_TAP_C='0', NQ_S3_ADM_CEMA='0.2', NQ_S3_ADM_HL='64', NQ_S3_LW_POW='0.25'),
          dict(seed=9, sal=True, kv=True)),
         ('stub k0 s3 arm budget lw1 hl16', k0, dict(NQ_TAP_C='0', NQ_S3_ADM_CEMA='0.5', NQ_S3_ADM_HL='16', NQ_S3_LW_POW='1',
                                                 NQ_TAP_MLA='0.3'), dict(seed=10, sal=True, pin=True)),
         ('stub b s3 mul.5 + add.2 lw.5 hl256 jfmass', b, dict(NQ_TAP_C='0', NQ_S3_MUL_POW='0.5', NQ_S3_ADM_CEMA='0.2', NQ_S3_LW_POW='0.5',
                                                 NQ_S3_LW_HL='256', NQ_S3_MASS='jf', NQ_S3_TRACK='1'), dict(seed=11, sal=True, kv=True))]
    sel = os.environ.get('CASES')
    for i, (nm, cfg, env, kw) in enumerate(C):
        if sel and str(i) not in sel.split(','): continue
        ok &= tap_parity(nm, steps, lambda kw=kw: (StubS(kw['seed']), StubS(kw['seed'])), cfg, env, **kw)
    if os.environ.get('NQ_TEST_JF', '1') == '1' and (not sel or 'jf' in sel.split(',')):
        js = [x for x in steps if x[1] <= 16][:int(os.environ.get('JF_STEPS', '1500'))]
        ok &= tap_parity('jF joint (CPU) rec/replay, c1', js, jf_pair, k0, dict(NQ_TAP_C='1'), seed=7, kv=True)
    return ok


def bench():
    steps = [x for x in routing_steps(int(os.environ.get('NFILES', '60'))) if x[1] <= 16]; k0, _ = cfgs()
    for hl in ('py', 'cpp'):
        P = StubS(1, nan=False); C = Clock(0); S = mk(k0, P, hl, dict(NQ_TAP_C='1'), C); S.io_all = fake_io(0); X = LeaderX(0)
        ts = []; tr = []
        for c, ntok, tid, nr in steps:
            c = c.astype(np.float64); C.adv(ntok); r0 = S.stats['refreshes']
            t = time.perf_counter(); u, d = S.step(c, ntok, tid, nr); dt = time.perf_counter() - t
            (tr if S.stats['refreshes'] != r0 else ts).append(dt); X.apply(u, d); X.tick(S)
        a = np.array(ts + tr) * 1e3; ts = np.array(ts) * 1e3; tr = np.array(tr) * 1e3
        print(f'  tap {hl}: all steps mean {a.mean():.3f} p50 {np.median(a):.3f} p99 {np.percentile(a, 99):.3f} ms/iter; '
              f'refresh steps ({len(tr)}) mean {tr.mean():.3f} p99 {np.percentile(tr, 99):.3f}; other ({len(ts)}) mean {ts.mean():.3f} ms '
              f'(stub predictor ~{0:.0f}, io_all fake)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'bench': bench(); sys.exit(0)
    ok = main(); print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
