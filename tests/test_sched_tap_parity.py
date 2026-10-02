"""NQ_SCHED parity: unset NQ_SCHED must be the unchanged default scheduler.
1. make_scheduler() with NQ_SCHED unset / '' / 'default' returns exactly scheduler.Scheduler (not a subclass)
2. the nq_vllm selection expression with NQ_SCHED unset is scheduler.Scheduler itself
3. op-for-op identity: a direct Scheduler and the factory-built one, fed the same routing trace (ids-only stream) with
   the same executor feedback (land after 3 steps, release after 1), give identical ups/downs and state every step,
   with the EMA rule and with a predictor object
4. NQ_SCHED=tap constructs TapScheduler and runs on the same trace (smoke: ops issued, invariants hold)
  python tests/test_sched_tap_parity.py [trace.npz] [n_tokens]"""
import os, sys, collections
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R + '/streaming')
import scheduler as SC
import scheduler_tap as TS

trace = sys.argv[1] if len(sys.argv) > 1 else '/rawdata/Jarrel/nq-tfpred/ds/ids-embedding-drift-monitor.npz'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
z = np.load(trace); ex = np.asarray(z['ex'][:N]); NL, NE = 75, 256
L_ = list(range(3, 78)); fx = {L: [] for L in L_}
dflt = {L: list(range(77)) for L in L_}


class StubP:
    """deterministic predictor object (GBDTPredictor interface): score = EMA64 of counts, refresh every 16 tokens"""
    def __init__(s): s.E = np.zeros((NL, NE)); s.n = 0; s.S = None; s.nf = 77
    def step(s, c, ntok=1, token_ids=None, new_request=False):
        s.E = s.E * 0.5 ** (ntok / 64) + c; s.n += ntok
        if s.n % 16 == 0: s.S = s.E.astype(np.float32).copy(); return True
        return False
    def target(s, r):
        if s.S is None: return None
        v = np.where(r, s.S * 1.7, s.S); top = np.argsort(-v, 1, kind='stable')[:, :77]
        w = np.zeros((NL, NE), bool); np.put_along_axis(w, top, True, 1); return w
    def order_score(s, r): return None if s.S is None else np.where(r, s.S * 1.7, s.S)
    def close(s): pass


def run(S, n=N):
    q = collections.deque(); log = []
    for t in range(n):
        c = np.zeros((NL, NE)); np.add.at(c, (np.repeat(np.arange(NL), 8), ex[t].reshape(-1).astype(np.int64)), 1)
        ups, downs = S.step(c, 1)
        log.append((tuple(ups), tuple(downs), S.state.tobytes()))
        for L, e in ups: q.append((t + 3, 'land', L, e))
        for L, e in downs: q.append((t + 1, 'rel', L, e))
        while q and q[0][0] <= t:
            _, k, L, e = q.popleft(); (S.landed if k == 'land' else S.released)(L, e)
        q = collections.deque(sorted(q))
    return log


def mk(cls, pred):
    return cls(L_, fx, dflt, 9970000, NE=NE, n_float=77, slots=80 * 75, cap_GBps=1e6, predictor=pred)


ok = True
for v in (None, '', 'default'):
    if v is None: os.environ.pop('NQ_SCHED', None)
    else: os.environ['NQ_SCHED'] = v
    t = type(TS.make_scheduler(L_, fx, dflt, 9970000, NE=NE, n_float=77, slots=6000, predictor='ema'))
    print('make_scheduler NQ_SCHED=%r ->' % v, t.__name__); ok &= t is SC.Scheduler
os.environ.pop('NQ_SCHED', None)
sel = SC.Scheduler if not os.environ.get('NQ_SCHED') else __import__('scheduler_tap').make_scheduler   # nq_vllm.py expression
print('nq_vllm selection with NQ_SCHED unset is scheduler.Scheduler:', sel is SC.Scheduler); ok &= sel is SC.Scheduler
for name, pf in (('ema', lambda: 'ema'), ('predictor-object', StubP)):
    a = run(mk(SC.Scheduler, pf())); b = run(TS.make_scheduler(L_, fx, dflt, 9970000, NE=NE, n_float=77, slots=6000, cap_GBps=1e6, predictor=pf()))
    nd = sum(x != y for x, y in zip(a, b)); nops = sum(len(x[0]) + len(x[1]) for x in a)
    print(f'{name}: {N} steps, {nops} ops, mismatching steps {nd}'); ok &= nd == 0 and nops > 0
os.environ['NQ_SCHED'] = 'tap'
T = TS.make_scheduler(L_, fx, dflt, 9970000, NE=NE, n_float=77, slots=6000, cap_GBps=1e6, predictor=StubP(), clock=iter(np.arange(0, 1e6, 1 / 94.)).__next__)
lt = run(T); nu = sum(len(x[0]) for x in lt); occ = (((T.state == 1) | (T.state == 2)) & ~T.doomed).sum(1)
print(f'tap: {type(T).__name__}, ups {nu}, max per-layer occupancy {occ.max()} (<= 77), stats', {k: (round(v, 1) if isinstance(v, float) else v) for k, v in T.stats.items()})
ok &= type(T) is TS.TapScheduler and nu > 0 and occ.max() <= 77
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
