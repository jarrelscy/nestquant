"""nq-tapad offline sim (CPU only, no GPU): TapScheduler against a synthetic drive, NQ_TAP_ADAPT off (today) vs on.
Workload: 75 layers x 256 experts, top-8 routing per token per layer drawn from per-layer popularity logits that jump to a
new topic every TOPIC tokens (TOPIC_MIX of the logit mass re-drawn: a burst of swap demand) plus a slow drift; 1 token
per step at 94 tok/s (sim clock). Predictor: EMA (half-life 32 tok) of the routing counts, rescaled to 512 hits/layer,
refreshed every 16 tokens (jF-shaped S, lags a topic switch like a real predictor). k0 layout (no fixed set, nf 77,
80 slots/layer), the floating set starts landed. Drive (rank 0 only, TP 4 record size 2.56 MB): FIFO into qd 8 servers,
each at R(t)/qd, so throughput is R only with >= qd reads outstanding (queue-depth bound below that, like an NVMe);
a read lands (S.landed) at the first step boundary after it completes; downs release one step later.
Metrics (from token WARM): hit = share of routed calls served at level 4 (state 2), lat = issue->land tokens (mean /
p95), q = reads at the drive (p95 / max), GB/s delivered, est = the scheduler's rate (GB/s) at the end, conv = first
token after which the estimate stays within 20% of the true rate (adaptive arms), bp = backpressure cuts.
main port: NQ_SCHED=tap TapScheduler(scheduler.Scheduler); NQ_TAP_H=64 (prod) unless SIM_H; SIM_HL=cpp runs every arm on the
nqhost.TapCore path (NQ_BUILD: use a private dir, the default build dir is shared with prod).
  python tests/tap_adapt_sim.py [parity|b|c|d|e|all] [ntok]
parity: the branch's TapScheduler with NQ_TAP_ADAPT unset vs REF's (default origin/main) from git, lockstep on the same sim
(scenario b, c, d drives, lat model and lat meas): ups / downs (order) / state / stats / peak identical; on the py and the
cpp host loop; + ADAPT=1 py vs cpp lockstep (the TapCore refresh takes the adaptive budget bit for bit).
e: RAM tier / host-LRU hits / prefill-borrow landings (TIER_FRAC of the experts land one step after issue without the
drive): drive 2 GB/s, start 6.6; ADAPT with the executor's land_nd exclusion (S.xnd) vs without."""
import os, sys, subprocess, tempfile, importlib, collections, heapq, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO + '/streaming')
import numpy as np
import scheduler_tap as TS

NL, NE, K, TPS, QD = 75, 256, 8, 94.0, 8
RB, TP = 2560000 * 4, 4; RBR = RB / TP
LAYERS = list(range(3, 3 + NL)); NF = 77; SLOTS = 80 * NL
TOPIC, MIX, WARM = int(os.environ.get('TOPIC', '1000')), float(os.environ.get('TOPIC_MIX', '0.5')), 300
ENVK = [k for k in os.environ if k.startswith('NQ_TAP_')]
for k in ENVK: os.environ.pop(k)
os.environ['NQ_TAP_CTL'] = '/nonexistent/nq_tap_ctl'; os.environ['NQ_TAP_TP'] = str(TP); os.environ['NQ_TAP_H'] = os.environ.get('SIM_H', '64')
HL = os.environ.get('SIM_HL', 'py'); TIERF = float(os.environ.get('TIER_FRAC', '0.3'))


def routing(ntok, seed=0):
    """-> counts [ntok, NL, NE] uint8 (top-8 per layer per token)"""
    r = np.random.default_rng(seed); z = r.normal(0, 1.6, (NL, NE)); out = np.zeros((ntok, NL, NE), np.uint8)
    for t in range(ntok):
        if t and t % TOPIC == 0:
            m = r.random((NL, NE)) < MIX; z[m] = r.normal(0, 1.6, int(m.sum()))
        z += r.normal(0, 0.01, (NL, NE))
        g = z + r.gumbel(size=(NL, NE)); top = np.argpartition(-g, K, 1)[:, :K]
        np.put_along_axis(out[t], top, 1, 1)
    return out


class EmaP:
    def __init__(s): s.E = np.zeros((NL, NE)); s.n = 0; s.S = None; s.a = 0.5 ** (1 / 32)
    def step(s, c, ntok=1, token_ids=None, new_request=False):
        s.E = s.E * s.a ** ntok + c; s.n += ntok
        if s.n % 16: return False
        s.S = (s.E * (512.0 / np.maximum(s.E.sum(1, keepdims=True), 1e-9))).astype(np.float32); return True
    def close(s): pass


class Drive:
    """FIFO into QD servers at R(t)/QD each"""
    def __init__(s, R): s.R = R; s.q = collections.deque(); s.free = [0.0] * QD; s.fl = []; s.n = 0; s.bytes = 0
    def submit(s, now, key): s.q.append((now, key))
    def advance(s, now):
        while s.q:
            i = int(np.argmin(s.free)); t0, k = s.q[0]; st = max(s.free[i], t0)
            if st > now: break
            s.q.popleft(); fin = st + RBR / (s.R(st) / QD); s.free[i] = fin; heapq.heappush(s.fl, (fin, s.n, k)); s.n += 1
        done = []
        while s.fl and s.fl[0][0] <= now: done.append(heapq.heappop(s.fl)[2])
        s.bytes += len(done) * RBR; return done
    def outstanding(s): return len(s.q) + len(s.fl)


def mk(mod, env, clock, hl=None):
    old = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        dflt = {L: list(range(NF)) for L in LAYERS}
        S = mod.TapScheduler(LAYERS, {L: [] for L in LAYERS}, dflt, RB, NE=NE, n_float=NF, slots=SLOTS, predictor=EmaP(), clock=clock,
                             hostloop=hl or HL)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    S.state[:, :NF] = 2                                   # floating default landed at start
    return S


def run(R, ntok, env, rc, mod=TS, probe=None, trace=None, hl=None, tier=False, nd=True):
    """-> metrics dict; trace (list) gets per-step (ups, downs, state bytes, stats, peak) for parity; tier: TIER_FRAC of the
    experts land one step after issue without the drive (RAM tier / LRU / borrowed pool); nd: the executor reports them (S.xnd)"""
    T = [0.0]; S = mk(mod, env, lambda: T[0], hl); D = Drive(R); rel = []; iss = {}; fast = []
    if probe is not None: S.probe_rate(probe)
    if getattr(S, 'tad', False): S.xio = D.outstanding
    X = set() if tier and nd and getattr(S, 'tad', False) else None
    if X is not None: S.xnd = X
    ist = lambda k: tier and (k[0] * 7919 + k[1] * 104729) % 1000 < TIERF * 1000
    hit = tot = 0; lat = []; qs = []; est = []; cut0 = 0
    for t in range(ntok):
        T[0] = t / TPS; c = rc[t].astype(np.float64)
        if t >= WARM: hit += float((c * (S.state == 2)).sum()); tot += float(c.sum())
        ups, downs = S.step(c, 1)
        if trace is not None:
            trace.append((tuple(ups), tuple(downs), S.state.tobytes(), {k: v for k, v in S.stats.items()},
                          None if S.peak is None else S.peak.tobytes()))
        for k in fast:
            if X is not None: X.add(k)
            S.landed(*k); i0 = iss.pop(k, None)
            if i0 is not None and t >= WARM: lat.append(t + 1 - i0)
        fast = []
        for k in ups:
            iss[k] = t
            if ist(k): fast.append(k)
            else: D.submit(T[0], k)
        for k in rel: S.released(*k)
        rel = list(downs)
        for k in D.advance(T[0] + 1 / TPS):               # completed during this step -> applied at the next step boundary
            S.landed(*k); i0 = iss.pop(k, None)
            if i0 is not None and t >= WARM: lat.append(t + 1 - i0)
        if t >= WARM: qs.append(D.outstanding())
        est.append((float(S.peak.min()) if S.peak is not None else S.rate0) * RBR / 1e9)
    est = np.array(est); tru = np.array([R(t / TPS) for t in range(ntok)]) / 1e9
    bad = np.nonzero(np.abs(est - tru) > 0.2 * tru)[0]
    lat = np.array(lat or [np.nan]); qs = np.array(qs)
    return dict(hit=hit / max(tot, 1), lat=float(np.nanmean(lat)), lat95=float(np.nanpercentile(lat, 95)), q95=float(np.percentile(qs, 95)),
                qmax=int(qs.max()), gbps=D.bytes / (ntok / TPS) / 1e9, nd=int(S.stats.get('ad_nd', 0)), landed=int(D.bytes / RBR), est=float(est[-1]),
                conv=int(bad[-1] + 1) if len(bad) else 0, bp=int(S.stats.get('bp_cut', 0)), cut=int(S.stats['budget_cut']), est_tr=est)


def ref_module(ref):
    d = tempfile.mkdtemp(prefix='nq_tapad_ref_')
    open(f'{d}/scheduler_tap.py', 'wb').write(subprocess.check_output(['git', '-C', REPO, 'show', f'{ref}:streaming/scheduler_tap.py']))
    assert subprocess.check_output(['git', '-C', REPO, 'diff', ref, '--', 'streaming/scheduler.py']) == b'', 'scheduler.py differs from REF'
    sys.path.insert(0, d); saved = sys.modules.pop('scheduler_tap')
    try: m = importlib.import_module('scheduler_tap')
    finally: sys.path.remove(d); sys.modules['scheduler_tap'] = saved
    assert m.__file__.startswith(d); return m


def parity(ntok, rc):
    RM = ref_module(os.environ.get('REF', 'origin/main')); ok = True
    for hl in ('py', 'cpp'):
      for nm, R, env in (('b: 2 GB/s drive, start 6.6', lambda t: 2e9, dict(NQ_TAP_RATE_GBPS='6.6')),
                       ('c: 6 -> 2 GB/s, lat meas', lambda t: 6e9 if t < ntok / TPS / 2 else 2e9, dict(NQ_TAP_LAT='meas')),
                       ('d: 12 GB/s, ADAPT=0 explicit', lambda t: 12e9, dict(NQ_TAP_ADAPT='0'))):
        a, b = [], []
        ma = run(R, ntok, env, rc, RM, trace=a, hl=hl); mb = run(R, ntok, env, rc, TS, trace=b, hl=hl)
        nd = sum(x != y for x, y in zip(a, b)); nops = sum(len(x[0]) + len(x[1]) for x in a)
        print(f'  parity {hl:3s} REF vs ADAPT=0 {nm:32s}: {ntok} steps, {nops} ops, mismatching steps {nd}, hit {ma["hit"]:.4f}/{mb["hit"]:.4f}', flush=True)
        ok &= nd == 0 and nops > 0 and len(a) == len(b)
    for nm, R, env, kw in (('b: ADAPT=1 2 GB/s, start 6.6', lambda t: 2e9, dict(NQ_TAP_RATE_GBPS='6.6', NQ_TAP_ADAPT='1'), {}),
                           ('c: ADAPT=1 6 -> 2, lat meas', lambda t: 6e9 if t < ntok / TPS / 2 else 2e9, dict(NQ_TAP_ADAPT='1', NQ_TAP_LAT='meas'), {}),
                           ('e: ADAPT=1 tier hits excluded', lambda t: 2e9, dict(NQ_TAP_ADAPT='1'), dict(tier=True))):
        a, b = [], []
        ma = run(R, ntok, env, rc, TS, trace=a, hl='py', **kw); mb = run(R, ntok, env, rc, TS, trace=b, hl='cpp', **kw)
        cm = lambda x, y: x[:3] + (x[4], {k: v for k, v in x[3].items() if k in y[3]})   # py-only stats (q_st1 / q_x ...) aside
        nd = sum(cm(x, y) != cm(y, x) for x, y in zip(a, b)); nops = sum(len(x[0]) + len(x[1]) for x in a)
        print(f'  parity ADAPT=1 py vs cpp {nm:32s}: {ntok} steps, {nops} ops, mismatching steps {nd}, hit {ma["hit"]:.4f}/{mb["hit"]:.4f}, bp {ma["bp"]}/{mb["bp"]}', flush=True)
        ok &= nd == 0 and nops > 0 and len(a) == len(b)
    return ok


def fmt(nm, m): return (f'| {nm:34s} | {m["hit"]:.4f} | {m["lat"]:6.1f} | {m["lat95"]:6.0f} | {m["q95"]:6.0f} | {m["qmax"]:5d} | {m["gbps"]:5.2f} | '
                        f'{m["landed"]:6d} | {m["est"]:5.2f} | {m["conv"]:5d} | {m["bp"]:4d} |')


HDR = ('| arm | hit | lat tok | lat p95 | q p95 | q max | GB/s | landed | est GB/s | conv tok | bp |\n'
       '|---|---|---|---|---|---|---|---|---|---|---|')
AD = dict(NQ_TAP_ADAPT='1')


def scen(which, ntok, rc):
    rows = []
    if which == 'b':
        R = lambda t: 2e9; E0 = dict(NQ_TAP_RATE_GBPS='6.6')
        rows += [('today (peak-hold), start 6.6', run(R, ntok, E0, rc)),
                 ('ADAPT=1, start 6.6, no probe', run(R, ntok, dict(E0, **AD), rc)),
                 ('ADAPT=1 + probe (1.9 GB/s)', run(R, ntok, dict(E0, **AD), rc, probe=1.9)),
                 ('ADAPT=1 EWMA only (BP=0), start 6.6', run(R, ntok, dict(E0, NQ_TAP_BP='0', **AD), rc)),
                 ('reference: today, start 2.0 (oracle)', run(R, ntok, dict(NQ_TAP_RATE_GBPS='2'), rc))]
    elif which == 'c':
        R = lambda t: 6e9 if t < ntok / TPS / 2 else 2e9
        rows += [('today (peak-hold), start 6', run(R, ntok, {}, rc)),
                 ('ADAPT=1, start 6', run(R, ntok, dict(AD), rc)),
                 ('ADAPT=1 EWMA only (BP=0)', run(R, ntok, dict(NQ_TAP_BP='0', **AD), rc))]
    elif which == 'e':
        R = lambda t: 2e9; E0 = dict(NQ_TAP_RATE_GBPS='6.6')
        rows += [(f'today, {TIERF:.0%} fast landings', run(R, ntok, E0, rc, tier=True)),
                 ('ADAPT=1, fast landings counted', run(R, ntok, dict(E0, **AD), rc, tier=True, nd=False)),
                 ('ADAPT=1, excluded (land_nd)', run(R, ntok, dict(E0, **AD), rc, tier=True))]
    elif which == 'd':
        for g in (12, 6):
            R = lambda t, g=g: g * 1e9
            rows += [(f'{g} GB/s: today, start 6', run(R, ntok, {}, rc)),
                     (f'{g} GB/s: ADAPT=1, start 6', run(R, ntok, dict(AD), rc)),
                     (f'{g} GB/s: ADAPT=1 + probe ({g * 0.95:.1f})', run(R, ntok, dict(AD), rc, probe=g * 0.95))]
    return rows


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else 'all'; ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    t0 = time.time(); rc = routing(ntok); print(f'routing {ntok} tok in {time.time() - t0:.1f}s (topic every {TOPIC}, mix {MIX}), host loop {HL}, H {os.environ["NQ_TAP_H"]}', flush=True)
    ok = True
    if w in ('parity', 'all'): ok &= parity(min(ntok, 3000), rc)
    for s in ('b', 'c', 'd', 'e'):
        if w not in (s, 'all'): continue
        rows = scen(s, ntok, rc); print(f'\nscenario {s} ({ntok} tok, metrics from token {WARM})\n' + HDR)
        for nm, m in rows: print(fmt(nm, m), flush=True)
        if s == 'c':
            for nm, m in rows:
                e = m['est_tr']; h = ntok // 2; print(f'  {nm}: est GB/s at drop+0/94/188/470/940/end tok: ' +
                                                      ' '.join(f'{e[min(h + d, ntok - 1)]:.2f}' for d in (0, 94, 188, 470, 940, ntok)))
    if w in ('parity', 'all'): print('PARITY PASS' if ok else 'PARITY FAIL')
    sys.exit(0 if ok else 1)
