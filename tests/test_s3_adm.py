"""Step 3 tap arm (NQ_S3*, streaming/scheduler_tap.py) vs the p-online reference (/data/Jarrel/nq-step3/p-online pols.Combo +
olib.budget_sim, the sim behind res_stack_<stream>.json). CPU only, needs the p-online data.
A live TapScheduler (hostloop py) is wrapped as an olib Policy: observe(t) = T.step(routed counts of row t, ntok=1,
new_request at request starts, sal = routed salience w^2 xn), its predictor replays the jF S p-online uses (S[n // 16]
at every 16 rows). At each refresh the live floating admission values (_value V x pair-gain weight lw, i.e. what the
global sort sees) must equal Combo.score(t) up to the common window factor (rel 1e-5), and budget_sim on the live keys
must reproduce p-online's count / salience hot % (res_stack, |d| <= 0.02).
  arms: env NQ_S3=1 (= g25.9.25+B5n22) with NQ_S3_RESET=0 (the sim streams run state across requests) -> res_stack;
        same, NQ_S3_RESET=1 (per-request reset, the live default) vs Combo + reset at new_request (numbers reported);
        safer arm via ctl only: env unset + NQ_S3_TRACK=1, ctl 's3=1 s3b=0.35 s3a=1 s3rst=0' -> res_stack g35.1.25+B5n22;
        env unset -> the plain jF S, no weight (step 2 rc).
  python tests/test_s3_adm.py [stream ...]   (tb-<task> | gen-<key>; default tb-cad-model gen-lean4_explain0; ROWS caps rows)"""
import os, sys, json, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../streaming'); sys.path.insert(0, HERE)
PO = '/data/Jarrel/nq-step3/p-online'; sys.path.insert(0, PO)
import numpy as np
import scheduler_tap as TS
from test_hostloop_parity import cfgs, LAYERS, NE
import olib as OL, pols as PL
NL = 75
ARM = dict(b=.25, a=.9, mass='512', wL=.5, kL=.5, nb=22, p=.25)            # p-online t_stack 'g25.9.25+B5n22'
SAFE = dict(b=.35, a=1.0, mass='512', wL=.5, kL=.5, nb=22, p=.25)          # 'g35.1.25+B5n22'
S3ENV = ('NQ_S3', 'NQ_S3_B', 'NQ_S3_BHL', 'NQ_S3_A', 'NQ_S3_AHL', 'NQ_S3_P', 'NQ_S3_BAND', 'NQ_S3_BW', 'NQ_S3_BK', 'NQ_S3_RESET',
         'NQ_S3_TRACK', 'NQ_TAP_CTL', 'NQ_TAP_C')


class ReplayS:
    """predictor stub: refresh every 16 rows with jF S[n // 16] (n = rows seen) = p-online's S[t // G] at refresh row t"""
    bs = None
    def __init__(s, S): s.SS = S; s.n = 0; s.S = None
    def step(s, c, ntok, tid=None, nr=False, sal=None):
        s.n += ntok
        if s.n % 16 or s.n // 16 >= len(s.SS): return False
        s.S = np.asarray(s.SS[s.n // 16], np.float32); return True
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


def boot(SS, env):
    k0, _ = cfgs(); old = {k: os.environ.get(k) for k in S3ENV}
    try:
        for k in S3ENV: os.environ.pop(k, None)
        os.environ['NQ_TAP_CTL'] = '/nonexistent/nq_tap_ctl'; os.environ.update(env)
        return TS.TapScheduler(LAYERS, k0['fixed'], k0['dflt'], 2560000, NE=NE, n_float=k0['nf'], slots=k0['slots'], cap_GBps=k0['cap'],
                               predictor=ReplayS(SS), hostloop='py')
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


class ComboR(PL.Combo):
    """Combo with all state reset at request start (the live per-request default)"""
    def __init__(s, st, **kw): s._kw = kw; s._st = st; super().__init__(st, **kw)
    def new_request(s, t, st, r): PL.Combo.__init__(s, s._st, **s._kw)


class Live(OL.Policy):
    """the live scheduler as a budget_sim policy; checks its keys against a reference Combo at every refresh"""
    def __init__(s, T, ref):
        s.T = T; s.ref = ref; s.nr = False; s.cap = None; s.worst = 0.0; s.n = 0; s.bad = 0
        v0 = T._value
        def cap(lat, H):
            V, V0 = v0(lat, H); lw = T._lw(); s.cap = (T._srcS(), V if lw is None else V * lw[:, None]); return V, V0
        T._value = cap
    def new_request(s, t, st, r): s.nr = True; s.ref.new_request(t, st, r)
    def observe(s, t, idr, sr):
        c = np.zeros((NL, NE)); c[OL.ar, idr] = 1.0; sl = np.zeros((NL, NE)); sl[OL.ar, idr] = sr
        s.T.step(c, 1, None, s.nr, sal=sl); s.nr = False; s.ref.observe(t, idr, sr)
    def score(s, t):
        Sx, Vw = s.cap; s.cap = None
        v, _ = s.ref.score(t); fl = ~s.T.fixed & (v > 1e-30)
        r = Vw[fl] / v[fl]; k = np.median(r); d = float(np.abs(r / k - 1).max()) if r.size else 0.0
        s.worst = max(s.worst, d); s.n += 1; s.bad += d > 1e-5
        return Sx.astype(np.float64), Sx.astype(np.float64)     # Sx = v cast to float32 (lw x _value's normalisation = 1)


def load(name):
    if name.startswith('tb-'): return OL.load_tb(name[3:])
    return OL.load_gen(name[4:])


def run(st, T, ref, rows):
    if rows: st = dict(st, ids=st['ids'][:rows], sal=st['sal'][:rows], think=st['think'][:rows], rstart=st['rstart'][st['rstart'] < rows])
    L = Live(T, ref); t0 = time.time(); H, u = OL.budget_sim(st, L); r = OL.report(st, H)['all']
    return r, L, time.time() - t0


def main():
    names = sys.argv[1:] or ['tb-cad-model', 'gen-lean4_explain0']; rows = int(os.environ.get('ROWS', '0')) or None
    ok = True
    for nm in names:
        st = load(nm); SS = st['S']; res = json.load(open(f'{PO}/res_stack_{st["name"]}.json'))
        ctl = tempfile.NamedTemporaryFile('w', suffix='.nq_tap_ctl', delete=False); ctl.write('s3=1 s3b=0.35 s3a=1 s3rst=0\n'); ctl.close()
        cases = [('g25.9.25+B5n22 env NQ_S3=1 rst0', boot(SS, dict(NQ_S3='1', NQ_S3_RESET='0')), PL.Combo(st, **ARM), 'g25.9.25+B5n22'),
                 ('g35.1.25+B5n22 ctl (env unset, TRACK)', boot(SS, dict(NQ_S3_TRACK='1', NQ_TAP_CTL=ctl.name)), PL.Combo(st, **SAFE), 'g35.1.25+B5n22'),
                 ('g25.9.25+B5n22 env NQ_S3=1 (reset)', boot(SS, dict(NQ_S3='1')), ComboR(st, **ARM), None)]
        U = boot(SS, {}); U.P.S = np.asarray(SS[1], np.float32)
        off = not U.s3on and not U.s3any and U.s3st is None and U._srcS() is U.P.S and U._lw() is None and not U.s3_sal
        print(f'{st["name"]}: {len(st["ids"]) if rows is None else rows} rows, {len(st["rstart"])} requests; env unset = jF S / no weight / no state: {off}; '
              f'p-online jF {res["jF"]["all"][0]:.2f}/{res["jF"]["all"][1]:.2f}', flush=True)
        ok &= off
        for cn, T, ref, key in cases:
            r, L, dt = run(st, T, ref, rows)
            line = f'  {cn:40s} live {r[0]:.2f}/{r[1]:.2f}  keys vs Combo: {L.n} refreshes, max rel {L.worst:.1e} ({L.bad} > 1e-5)'
            good = L.bad == 0 and L.n > 0
            if key and not rows:
                e = res[key]['all']; dd = (r[0] - e[0], r[1] - e[1]); good &= max(abs(dd[0]), abs(dd[1])) <= 0.02
                line += f'; p-online {e[0]:.2f}/{e[1]:.2f} (d {dd[0]:+.3f}/{dd[1]:+.3f})'
            line += f'; vs jF {r[0] - res["jF"]["all"][0]:+.2f}/{r[1] - res["jF"]["all"][1]:+.2f}  [{dt:.0f}s] ' + ('ok' if good else 'FAIL')
            print(line, flush=True); ok &= good
        os.unlink(ctl.name)
    return ok


if __name__ == '__main__':
    ok = main(); print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
