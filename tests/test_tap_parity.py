"""Tap scheduler refactor parity (clean-d): the standalone serve TapScheduler (streaming/scheduler_tap.py, no base class,
python host loop only) vs the pre-cleanup one (scheduler_tap.TapScheduler(scheduler.Scheduler) at git REF, default
4c9c8e0, hostloop='py'), extracted from git into a temp dir. Needs no GPU.
  python tests/test_tap_parity.py            all cases       (NFILES routing-log files, default 60)
Both run in lockstep on per-step routing counts from the GLM-5.3 routing logs (routing_steps), a deterministic fake
clock (tok/s with jitter), a fake io_all() (4 ranks, varying delivered rate / outstanding / slot wait / backlog,
sometimes empty or raising) and identical fake executors (LeaderX: landed / released / failed incl. read errors).
Per step: ups and downs identical (same order); score / state / want / hold / doomed bitwise; the lazy-eviction list
(doom), the no-slot queue (todo), the stats (minus the base class's deferred_steps, never incremented by tap) and the
rate state equal; kv_pressure results and level() equal. Predictors: stub score matrices (float32 ties, all-zero
layers, NaN entries, negative scores) and the jF joint predictor on CPU, recorded on the reference side and replayed
on the new side. Configs: the serve k0 layout (default env: the reference gets NQ_TAP_H=64, the new class its own
default, so the D default is checked too), a fixed set, slot-bound (todo queue + eager eviction), budget-bound (MLA),
adaptive H (HA), far=ema, lat=meas, pin and kv_pressure."""
import os, sys, glob, random, subprocess, tempfile, importlib, collections
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO + '/streaming')
import numpy as np
import scheduler_tap as TS

LAYERS = list(range(3, 78)); NE = 256
REF = os.environ.get('REF', '4c9c8e0')
TAPENV = ('NQ_TAP_C', 'NQ_TAP_H', 'NQ_TAP_HA', 'NQ_TAP_MLA', 'NQ_TAP_SVC_MS', 'NQ_TAP_FAR', 'NQ_TAP_TP', 'NQ_TAP_RATE_GBPS',
          'NQ_TAP_LAT', 'NQ_TAP_LAT_HL', 'NQ_TAP_LAT_Q', 'NQ_TAP_LAT_N', 'NQ_HOSTLOOP', 'NQ_PREDICTOR')


def ref_module():
    """the pre-cleanup scheduler_tap (+ its base scheduler.py) from git, imported under private names"""
    d = tempfile.mkdtemp(prefix='nq_tapref_')
    for f in ('scheduler.py', 'scheduler_tap.py'):
        open(f'{d}/{f}', 'wb').write(subprocess.check_output(['git', '-C', REPO, 'show', f'{REF}:streaming/{f}']))
    saved = sys.modules.pop('scheduler', None); sys.path.insert(0, d)
    try:
        sys.modules.pop('scheduler_tap', None)
        m = importlib.import_module('scheduler_tap'); m = sys.modules.pop('scheduler_tap'); sys.modules.pop('scheduler', None)
    finally:
        sys.path.remove(d)
        if saved is not None: sys.modules['scheduler'] = saved
        sys.modules['scheduler_tap'] = TS
    assert m.__file__.startswith(d) and TS.__file__.startswith(REPO), (m.__file__, TS.__file__)
    return m


def routing_steps(nfiles, start=0):
    """-> list of (counts [75, 256] uint16, ntok, token_ids, new_request) per model step"""
    fs = sorted(glob.glob('/data/Jarrel/routing_logs/glm5.3-arvq-v2/seg-*.npz'))[start:]
    out = []; last_req = None; run = min(20, nfiles)        # runs of 20 consecutive files at evenly spaced offsets over the logs
    fs = [f for o in np.linspace(0, len(fs) - run, max(1, nfiles // run)).astype(int) for f in fs[o:o + run]]
    for f in fs:
        z = np.load(f); ex = z['experts'].astype(np.int64); st = z['step']; tid = z['token_ids']; rq = z['req']
        for sv in np.unique(st):
            m = st == sv; e = ex[m][:, 3:78, :]; n = int(m.sum())
            c = np.bincount((e + (np.arange(75) * NE)[None, :, None]).ravel(), minlength=75 * NE).reshape(75, NE).astype(np.uint16)
            r = set(rq[m].tolist()); nr = last_req is not None and not r <= last_req; last_req = r
            out.append((c, n, tid[m].tolist(), nr))
    return out


class LeaderX:
    """fake executor: each step lands / fails a random subset of in-flight ups (some read errors), releases downs"""
    def __init__(s, seed): s.rng = random.Random(seed); s.upq = []; s.dq = []
    def apply(s, ups, downs): s.upq += ups; s.dq += downs
    def tick(s, S):
        r = s.rng; keep = []
        for k in s.upq:
            x = r.random()
            if x < 0.55: S.landed(*k)
            elif x < 0.57: S.failed(*k, read_error=True)
            elif x < 0.58: S.failed(*k)
            else: keep.append(k)
        s.upq = keep; keep = []
        for k in s.dq:
            if r.random() < 0.7: S.released(*k)
            else: keep.append(k)
        s.dq = keep


def cfgs():
    rng = np.random.default_rng(3)
    k0 = dict(fixed={L: [] for L in LAYERS}, nf=77, slots=6000)
    k0['dflt'] = {L: [int(x) for x in rng.permutation(NE)[:77]] for L in LAYERS}
    fx = {L: sorted(int(x) for x in rng.permutation(NE)[:26]) for L in LAYERS}
    b = dict(fixed=fx, nf=51, slots=3000)
    b['dflt'] = {L: [int(x) for x in rng.permutation(NE) if x not in fx[L]][:60] for L in LAYERS}
    return k0, b


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
    """reference side: real predictor, recording step() results and S"""
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


def with_env(env, f):
    old = {k: os.environ.get(k) for k in TAPENV}
    try:
        for k in TAPENV: os.environ.pop(k, None)
        os.environ.update(env); return f()
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def mk_ref(RM, cfg, P, env, clock):
    e = dict(env); e.setdefault('NQ_TAP_H', '64')       # reference default was 256; D serves 64 (= the new default)
    return with_env(e, lambda: RM.TapScheduler(LAYERS, cfg['fixed'], cfg['dflt'], 2560000, NE=NE, n_float=cfg['nf'], slots=cfg['slots'],
                                               cap_GBps=1e6, predictor=P, hostloop='py', clock=clock))


def mk_new(cfg, P, env, clock):
    return with_env(env, lambda: TS.TapScheduler(LAYERS, cfg['fixed'], cfg['dflt'], 2560000, NE=NE, n_float=cfg['nf'], slots=cfg['slots'],
                                                 predictor=P, clock=clock))


def stats(S): return {k: v for k, v in S.stats.items() if k != 'deferred_steps'}


def tap_parity(RM, name, steps, mkp, cfg, env, seed=0, kv=False, pin=False, io=True):
    pa, pb = mkp()
    CA, CB = Clock(seed), Clock(seed)
    A = mk_ref(RM, cfg, pa, env, CA); B = mk_new(cfg, pb, env, CB)
    assert A.core is None and A.tcore is None and type(B).__mro__[1] is object
    if A.tH != B.tH or A.tc != B.tc or A.wants_sal != B.wants_sal: print(f'  {name}: CONFIG DIFF tH {A.tH}/{B.tH} wants_sal {A.wants_sal}/{B.wants_sal}'); return False
    if io: A.io_all = fake_io(seed); B.io_all = fake_io(seed)
    XA, XB = LeaderX(seed), LeaderX(seed)
    init = [(L, E) for L in LAYERS for E in cfg['dflt'][L] if E not in cfg['fixed'][L]][:cfg['slots']]
    for S, X in ((A, XA), (B, XB)):
        for L, E in init: S.state[S.li[L], E] = 1
        X.apply(init, [])
    rng = random.Random(seed + 5); nops = 0; nref = 0
    for n, (c, ntok, tid, nr) in enumerate(steps):
        c = c.astype(np.float64); CA.adv(ntok); CB.adv(ntok); r0 = A.stats['refreshes']
        ua, da = A.step(c, ntok, tid, nr); ub, db = B.step(c, ntok, tid, nr)
        if (ua, da) != (ub, db):
            print(f'  {name}: OPS DIFF at step {n}: ups {len(ua)}/{len(ub)} downs {len(da)}/{len(db)} first ups {ua[:3]} {ub[:3]}'); return False
        for f in ('score', 'state', 'want', 'hold', 'doomed'):
            if not eq(getattr(A, f), getattr(B, f)): print(f'  {name}: {f} DIFF at step {n}'); return False
        if A.stats['refreshes'] != r0: nref += 1
        if list(A.doom.items()) != list(B.doom.items()) or list(A.todo) != list(B.todo):
            print(f'  {name}: doom/todo DIFF at step {n}: {len(A.doom)}/{len(B.doom)} {len(A.todo)}/{len(B.todo)}'); return False
        pk = (A.peak is None and B.peak is None) or (A.peak is not None and B.peak is not None and eq(A.peak, B.peak))
        if stats(A) != stats(B) or A.lat_ema != B.lat_ema or A.tok != B.tok or A.tps != B.tps or A.n_land != B.n_land or not pk:
            print(f'  {name}: scalar DIFF at step {n}: stats {stats(A)} {stats(B)}'); return False
        XA.apply(ua, da); XB.apply(ub, db); XA.tick(A); XB.tick(B); nops += len(ua) + len(da)
        if not eq(A.state, B.state) or not eq(A.hold, B.hold) or stats(A) != stats(B): print(f'  {name}: feedback DIFF at step {n}'); return False
        if kv and rng.random() < 0.01:
            k = rng.randint(1, 40)
            if A.kv_pressure(k) != B.kv_pressure(k): print(f'  {name}: kv_pressure DIFF at step {n}'); return False
        if n % 97 == 0 and not eq(A.level(), B.level()): print(f'  {name}: level() DIFF at step {n}'); return False
        if pin and n == len(steps) // 3:
            pm = np.random.default_rng(seed).random((75, NE)) < 0.05; A.pin = pm.copy(); B.pin = pm.copy()
        if pin and n == 2 * len(steps) // 3: A.pin = B.pin = None
    st = {k: A.stats[k] for k in ('ups', 'downs', 'refreshes', 'promotions', 'budget_cut', 'eager_evict', 'no_slot_skip', 'big_steps')}
    print(f'  {name:36s} IDENTICAL over {len(steps)} steps, {nref} refreshes, {nops} ops, {st}', flush=True)
    return nops > 0


def main():
    RM = ref_module()
    nf = int(os.environ.get('NFILES', '60'))
    steps = routing_steps(nf); print(f'tap refactor parity vs {REF}: {len(steps)} model steps, {sum(s[1] for s in steps)} tokens from {nf} routing-log files', flush=True)
    k0, b = cfgs()
    k0s = dict(k0, slots=5200)                       # fewer slots than 75 x 77: no-slot paths (todo, eager eviction)
    ok = True
    C = [('stub k0 default env (H=64)', k0, {}, dict(seed=1)),
         ('stub k0 slot-bound c1 kv+pin', k0s, dict(NQ_TAP_C='1'), dict(seed=2, kv=True, pin=True)),
         ('stub fixed+budget Ha2 mla0.5', b, dict(NQ_TAP_HA='2', NQ_TAP_MLA='0.5'), dict(seed=3, kv=True)),
         ('stub k0 far=ema c0.3 no-io', k0, dict(NQ_TAP_FAR='ema', NQ_TAP_C='0.3'), dict(seed=4, io=False)),
         ('stub k0s mla0 (no budget) H512', k0s, dict(NQ_TAP_MLA='0', NQ_TAP_H='512'), dict(seed=5, pin=True)),
         ('stub k0s lat=meas', k0s, dict(NQ_TAP_LAT='meas'), dict(seed=6, kv=True)),
         ('stub k0 lat=meas Ha2 q0.9 hl2 c0.5', k0, dict(NQ_TAP_LAT='meas', NQ_TAP_HA='2', NQ_TAP_LAT_Q='0.9', NQ_TAP_LAT_HL='2',
                                                    NQ_TAP_C='0.5'), dict(seed=8, pin=True))]
    sel = os.environ.get('CASES')
    for i, (nm, cfg, env, kw) in enumerate(C):
        if sel and str(i) not in sel.split(','): continue
        ok &= tap_parity(RM, nm, steps, lambda kw=kw: (StubS(kw['seed']), StubS(kw['seed'])), cfg, env, **kw)
    if os.environ.get('NQ_TEST_JF', '1') == '1' and (not sel or 'jf' in sel.split(',')):
        js = [x for x in steps if x[1] <= 16][:int(os.environ.get('JF_STEPS', '1500'))]
        ok &= tap_parity(RM, 'jF joint (CPU) rec/replay, c1 kv', js, jf_pair, k0, dict(NQ_TAP_C='1'), seed=7, kv=True)
    # followers construct the scheduler without a predictor and never step it
    F = mk_new(k0, None, {}, Clock(0)); ok &= F.P is None and not F.wants_sal and F.predictor_name == 'none' and F.tH == 64.
    print('  follower construction (predictor None): ok' if ok else '  follower construction: FAIL')
    return ok


if __name__ == '__main__':
    ok = main(); print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
