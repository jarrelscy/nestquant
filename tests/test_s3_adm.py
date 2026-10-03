"""Step 3 tap arm (NQ_S3_ADM_CEMA / NQ_S3_ADM_HL / NQ_S3_LW_POW, streaming/scheduler_tap.py) vs the p-policy sim
(/data/Jarrel/nq-step3/p-policy/psim.py, arm src='mix:0.8:cema64', lw=1, lwp=0.25). CPU only.
A TapScheduler (hostloop py) is driven row by row (ntok=1) with a short stream's routed counts + salience and the jF S
the sim uses (replayed every 16 rows: S[b] = info of rows < 16b); at every refresh the admission source (_srcS: blended
S, float32) must equal the sim's srcval('mix:0.8:cema64') cast to float32 bitwise, and the layer weight (_lw) the sim's
lwv (rtol 1e-12: the sim sums salience per expert first). Stream: tb task cap (psim.load_tb, default cad-model, first
ROWS=1024 rows) when the p-policy data exist, else a synthetic stream (sim formulas copied below).
Runtime switch: a second scheduler booted with the env unset + NQ_S3_TRACK=1 and the arm written to its NQ_TAP_CTL file
before the first refresh must give the same keys (same-boot A/B); and with the ctl file absent / env unset the source
is the plain jF S, no layer weight, c = 1 (step 2 rc).
  python tests/test_s3_adm.py [task]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../streaming'); sys.path.insert(0, HERE)
import numpy as np
import scheduler_tap as TS
from test_hostloop_parity import cfgs, LAYERS, NE
PP = '/data/Jarrel/nq-step3/p-policy'
NL = 75; ENV = dict(NQ_TAP_C='0', NQ_S3_ADM_CEMA='0.2', NQ_S3_ADM_HL='64', NQ_S3_LW_POW='0.25'); A_MIX = 0.8; LWP = 0.25


# ---- sim reference (verbatim formulas of psim.py: Ema(64, sal=False), norm512, mix:<a>:cema64, lw/lwp)
def norm512(V):
    m = V.sum(1, keepdims=True); return np.where(m > 0, V * 512.0 / np.where(m > 0, m, 1), 0)
class Ema:
    def __init__(s, hl): s.d = 0.5 ** (1 / hl); s.E = np.zeros(NL * NE)
    def add(s, flat): s.E *= s.d; np.add.at(s.E, flat, 1.0)


def stream(task, rows):
    if os.path.exists(PP + '/psim.py'):
        sys.path.insert(0, PP); import psim as P
        tr = P.load_tb(task); N = min(rows, tr.N)
        return task, tr.ids[:N], tr.sal[:N].astype(np.float32), tr.S, P
    rng = np.random.default_rng(0); N = rows                                  # synthetic: zipf-ish routing, gamma salience
    pr = rng.dirichlet(np.full(NE, 0.3), NL)
    ids = np.stack([np.stack([rng.choice(NE, 8, replace=False, p=pr[l]) for l in range(NL)]) for _ in range(N)])
    sal = (rng.gamma(1.0, 1.0, (N, NL, 8)) * rng.gamma(2.0, 1.0, (1, NL, 1))).astype(np.float32)
    S = (rng.gamma(0.5, 2.0, (N // 16 + 1, NL, NE)) - 0.3).astype(np.float32)
    return 'synthetic', ids, sal, S, None


class ReplayS:
    """predictor stub: refresh every 16 rows with the sim's jF S[b] (b = rows seen // 16)"""
    bs = None
    def __init__(s, S): s.SS = S; s.n = 0; s.S = None
    def step(s, c, ntok, tid=None, nr=False, sal=None):
        s.n += ntok
        if s.n % 16: return False
        s.S = np.asarray(s.SS[s.n // 16], np.float32); return True
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


def boot(SS, env):
    k0, _ = cfgs(); keys = list(ENV) + ['NQ_S3_TRACK', 'NQ_TAP_CTL', 'NQ_S3_MUL_POW', 'NQ_S3_MUL_HL', 'NQ_S3_LW_HL', 'NQ_S3_MASS']
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys: os.environ.pop(k, None)
        os.environ['NQ_TAP_CTL'] = '/nonexistent/nq_tap_ctl'; os.environ.update(env)
        return TS.TapScheduler(LAYERS, k0['fixed'], k0['dflt'], 2560000, NE=NE, n_float=k0['nf'], slots=k0['slots'], cap_GBps=k0['cap'],
                               predictor=ReplayS(SS), hostloop='py')
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else 'cad-model'
    name, ids, sal, SS, P = stream(task, int(os.environ.get('ROWS', '1024')))
    N = len(ids); ids = ids.astype(np.int64)
    import tempfile
    ctl = tempfile.NamedTemporaryFile('w', suffix='.nq_tap_ctl', delete=False); ctl.write('c=0 s3a=0.2 s3hl=64 s3q=0.25\n'); ctl.close()
    T = boot(SS, ENV); T2 = boot(SS, dict(NQ_S3_TRACK='1', NQ_TAP_CTL=ctl.name))
    assert T.s3 == dict(a=0.2, hl=64.0, p=0.0, phl=256.0, q=0.25, lhl=0.0, mass='512') and T.tc == 0.0 and T.s3_sal
    assert T2.tc == 1.0 and not T2.s3on and T2.s3_sal
    got = {}
    v0 = T._value
    def cap(lat, H):
        got[T.P.n // 16] = (np.array(T._srcS(), copy=True), T._lw()); return v0(lat, H)
    T._value = cap
    got2 = {}; v2 = T2._value
    def cap2(lat, H):
        got2[T2.P.n // 16] = (np.array(T2._srcS(), copy=True), T2._lw()); return v2(lat, H)
    T2._value = cap2
    em = Ema(64); cnt_hit = np.zeros(NL * NE); sal_hit = np.zeros(NL * NE); ref = {}
    AR = np.arange(NL)[:, None]; nb = 0
    for t in range(N):
        if t > 0 and t % 16 == 0:                                   # sim refresh at the start of row t (rows < t)
            b = t // 16; Sj = np.asarray(SS[b], np.float32).astype(np.float64)
            R = A_MIX * norm512(np.maximum(Sj, 0)) + (1 - A_MIX) * norm512(em.E.reshape(NL, NE).copy())
            lm = sal_hit.reshape(NL, NE).sum(1) / np.maximum(cnt_hit.reshape(NL, NE).sum(1), 1)
            ref[b] = (R, (lm / max(lm.mean(), 1e-12)) ** LWP)
        flat = (ids[t] + AR * NE).ravel()
        c = np.zeros((NL, NE)); c[AR, ids[t]] = 1.0
        s_ = np.zeros((NL, NE)); s_[AR, ids[t]] = sal[t].astype(np.float64)
        T.step(c, 1, sal=s_); T2.step(c, 1, sal=s_)
        em.add(flat); np.add.at(cnt_hit, flat, 1.0); np.add.at(sal_hit, flat, sal[t].ravel())
    bad = 0; worst = 0.0
    for b in sorted(ref):
        if b not in got: continue
        nb += 1; R, lw = ref[b]; g, lwg = got[b]
        if not (g.dtype == np.float32 and g.tobytes() == R.astype(np.float32).tobytes()):
            bad += 1; print(f'  refresh {b}: admission key DIFF, {int((g != R.astype(np.float32)).sum())} entries, max |d| {np.abs(g - R).max():.3g}')
        if lwg is None or not np.allclose(lwg, lw, rtol=1e-12, atol=0):
            bad += 1; print(f'  refresh {b}: layer weight DIFF {None if lwg is None else np.abs(lwg / lw - 1).max()}')
        else: worst = max(worst, float(np.abs(lwg / lw - 1).max()))
    os.unlink(ctl.name)
    same = T2.tc == 0.0 and T2.s3 == T.s3 and sorted(got2) == sorted(got) and all(
        got2[b][0].tobytes() == got[b][0].tobytes() and np.array_equal(got2[b][1], got[b][1]) for b in got)
    # env unset, no ctl file: the plain jF S (step 2 rc)
    U = boot(SS, {}); U.P.S = np.asarray(SS[1], np.float32)
    off = (not U.s3on and not U.s3any and U.s3Ea is None and U.s3sal is None and U._srcS() is U.P.S and U._lw() is None
           and U.tc == 1.0 and not U.s3_sal)
    ok = bad == 0 and nb == len(ref) and nb > 0 and off and same
    print(f's3 adm ({name}, {N} rows): {nb}/{len(ref)} refreshes, admission keys bitwise = sim mix:0.8:cema64, layer weight '
          f'max rel diff {worst:.2g} vs sim lw.25; ctl-switched same boot (TRACK=1) identical: {same}; env unset = jF S / no weight: {off}')
    return ok


if __name__ == '__main__':
    ok = main(); print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
