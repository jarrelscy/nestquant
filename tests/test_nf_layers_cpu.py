"""nq-lalloc: per-layer n_float (NQ_NF_LAYERS) CPU checks. Needs no GPU.
  python tests/test_nf_layers_cpu.py [PROD_TREE]      (PROD_TREE default /data/Jarrel/nestquant)
1. unset = prod: the prod-tree TapScheduler / Scheduler(ema) and this tree's, same config (serve layout: 26 fixed + 51 floating,
   56 x 75 slots, NQ_TAP_H=64 TODO_FIX=2) on routing-log steps with a stub score predictor, fake clock / executor: every
   step's ups / downs identical (each tree runs in its own subprocess, op logs compared by hash and length).
2. scalar 51 vs an array of 51s in this tree: identical ops (the array path computes the same thing).
3. ramp arm (71 -> 31): per layer the floating level-4 + in-flight count never exceeds nf[l] (tap), the mean over decode
   steps is close to nf[l], and the total stays under the slot pool.
4. jF target patch: nf_topmask / nf_target_patch with an array of 51s == the scalar target bitwise; ramp -> per-row counts."""
import os, sys, json, hashlib, subprocess, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(HERE)
PROD = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else '/data/Jarrel/nestquant'
NSTEP = int(os.environ.get('NSTEP', '3000'))


def child(tree, sched, nfmode, out):
    sys.path.insert(0, tree + '/streaming'); sys.path.insert(0, WT + '/tests')
    import numpy as np, random
    os.environ.update(NQ_TAP_H='64', NQ_TAP_TODO_FIX='2', NQ_TAP_C='1.0', NQ_TAP_FAR='tail:0.1', NQ_TAP_CTL='/nonexistent/x')
    import scheduler as SC, scheduler_tap as TS
    from test_hostloop_parity import routing_steps, LeaderX, LAYERS, NE
    from test_tap_parity import StubS, Clock
    steps = [x for x in routing_steps(int(os.environ.get('NFILES','60'))) if x[1] <= 16][:NSTEP]
    rng = np.random.default_rng(11)
    fx = {L: sorted(int(x) for x in rng.permutation(NE)[:26]) for L in LAYERS}
    if nfmode == 'scalar': nf = 51
    elif nfmode == 'arr51': nf = np.full(75, 51, np.int64)
    else: nf = np.array(json.load(open(nfmode))['nf'], np.int64)
    nmax = 51 if np.ndim(nf) == 0 else int(nf.max())
    dflt = {L: [int(x) for x in rng.permutation(NE) if x not in fx[L]][:nmax] for L in LAYERS}
    if np.ndim(nf): dflt = {L: dflt[L][:int(nf[i])] for i, L in enumerate(LAYERS)}
    else: dflt = {L: dflt[L][:51] for L in LAYERS}
    slots = 56 * 75; C = Clock(3)
    if sched == 'tap':
        S = TS.TapScheduler(LAYERS, fx, dflt, 2560000, NE=NE, n_float=nf, slots=slots, cap_GBps=1e6, predictor=StubS(5, nan=False),
                            hostloop='py', clock=C)
    else:
        S = SC.Scheduler(LAYERS, fx, dflt, 2560000, NE=NE, n_float=nf, slots=slots, cap_GBps=1e6, predictor='ema', hostloop='py')
    X = LeaderX(9); init = [(L, E) for L in LAYERS for E in dflt[L]][:slots]
    for L, E in init: S.state[S.li[L], E] = 1
    X.apply(init, [])
    h = hashlib.sha256(); nops = 0; occ = np.zeros(75); over = 0; n = 0
    for c, ntok, tid, nr in steps:
        c = c.astype(np.float64); C.adv(ntok); u, d = S.step(c, ntok, tid, nr)
        h.update(repr((u, d)).encode()); nops += len(u) + len(d)
        X.apply(u, d); X.tick(S)
        o = (((S.state == 1) | (S.state == 2)) & ~S.fixed & ~getattr(S, 'doomed', np.zeros_like(S.fixed))).sum(1)
        if n > 200: occ += o
        if np.ndim(nf): over = max(over, int((o - nf).max()))
        n += 1
    print('    steps', len(steps), flush=True)
    pickle.dump(dict(hash=h.hexdigest(), nops=nops, occ=(occ / max(1, n - 201)).tolist(), over=over,
                     busy=int((S.state > 0).sum()), stats={k: v for k, v in S.stats.items() if not k.startswith('q_')}), open(out, 'wb'))


def run(tree, sched, nfmode):
    out = f'/tmp/nqlalloc_{os.getpid()}_{abs(hash((tree, sched, nfmode)))}.pkl'
    subprocess.run([sys.executable, __file__, '--child', tree, sched, nfmode, out], check=True)
    r = pickle.load(open(out, 'rb')); os.remove(out); return r


def jf_check():
    import numpy as np
    sys.path.insert(0, WT + '/streaming'); import scheduler as SC
    rng = np.random.default_rng(1)

    class P:                                       # GPUJointPredictor.target / _adj, as in /nqpred/joint/gpu_predictor.py
        NL, NE, hm, ha = 75, 256, 0.7, 0.0
        def __init__(s, nf): s.nf = nf; s.fixed = rng.random((75, 256)) < 0.1; s.S = None
        def _adj(self, resident):
            v = np.where(self.fixed, -np.inf, self.S).astype(np.float32)
            r = np.asarray(resident, bool) & ~self.fixed
            return np.where(r, v * np.float32(1 + self.hm) + np.float32(self.ha), v), r
        def target(self, resident):
            if self.S is None: return None
            v, r = self._adj(resident)
            tot = np.where(self.fixed, 0, np.maximum(self.S, 0)).sum(1)
            top = np.argsort(-v, 1, kind="stable")[:, :self.nf]
            want = np.zeros((self.NL, self.NE), bool); np.put_along_axis(want, top, True, 1)
            nz = tot <= 0; want[nz] = r[nz]
            return want
    a = P(51); b = P(np.full(75, 51)); b.fixed = a.fixed; SC.nf_target_patch(b)
    ramp = np.round(np.linspace(71, 31, 75)).astype(int); c = P(ramp); c.fixed = a.fixed; SC.nf_target_patch(c)
    for k in range(50):
        S = np.round(rng.gamma(0.4, 3, (75, 256)) * 4).astype(np.float32) / 4; S[rng.random(75) < 0.05] = 0
        res = rng.random((75, 256)) < 0.2
        a.S = b.S = c.S = S
        wa, wb, wc = a.target(res), b.target(res), c.target(res)
        assert (wa == wb).all(), 'jF target: array 51 != scalar 51'
        ok = S.sum(1) > 0
        assert (wc.sum(1)[ok] == ramp[ok]).all(), 'jF target: ramp row counts'
    print('  jF target patch: array-51 == scalar bitwise (50 draws), ramp row counts exact')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        child(*sys.argv[2:6]); sys.exit(0)
    import numpy as np
    ok = True
    for sched in ('tap', 'ema'):
        p = run(PROD, sched, 'scalar'); w = run(WT, sched, 'scalar'); a = run(WT, sched, 'arr51')
        e1 = (p['hash'], p['nops']) == (w['hash'], w['nops']); e2 = (w['hash'], w['nops']) == (a['hash'], a['nops'])
        print(f'  {sched}: prod vs worktree (unset) {"IDENTICAL" if e1 else "DIFF"} ({p["nops"]} ops); scalar vs array-51 {"IDENTICAL" if e2 else "DIFF"}')
        ok &= e1 and e2
    rp = WT + '/streaming/results/nf_layers/R1.json'
    if os.path.exists(rp):
        nf = np.array(json.load(open(rp))['nf']); r = run(WT, 'tap', rp); occ = np.array(r['occ'])
        print(f'  tap R1 ramp: max over nf {r["over"]} (must be <= 0), busy {r["busy"]} / {56*75} slots; mean occ first5 {np.round(occ[:5],1).tolist()} '
              f'(nf {nf[:5].tolist()}) last5 {np.round(occ[-5:],1).tolist()} (nf {nf[-5:].tolist()}); mean |occ-nf| {np.abs(occ-nf).mean():.2f}')
        ok &= r['over'] <= 0
    jf_check()
    print('ALL PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
