"""nq-tfpred arms on the nq-algo simulator (nqsim.py, unchanged).  Results -> /data/Jarrel/nq-tfpred/nqalgo/runs/<cfg>.<task>.json
  ext.py CFG TASK        CFG = <policy>@<env>; env strings as nq-algo run.py (e.g. prB6.co, shB11.co)
policies (besides every run.py policy):
  tap-<fc>-c<c>[-H<H>|-Ha<k>][-ml<N>|-mla<k>][-cs][-g][-ls<name>]
      ValuePolicy + (a) issue budget sized to the MEASURED slowest-rank throughput (rank_state achieved_Bps peak-hold,
      falls back to the drive's processor-sharing rate before the first read) and kept under max landing time
      ml tokens (-mla<k>: ml = k x the refresh-to-refresh value window, i.e. adaptive), (b) adaptive horizon
      -Ha<k>: H = clamp(k x est_land_tok, 64, 1024), (c) -ls<name>: per-layer slot counts from slots_<name>.json
      (same total), (d) -cs cancel queued reads whose value fell below the weakest resident's.
forecasts (besides run.py's): win:<name> (WindowFC), shift1:<name> (the same file one block stale = the value of 16
  tokens fresher routing; bounds the MTP-draft lookahead gain)."""
import os, sys, json, re, time, collections
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/data/Jarrel/nq-algo')
import numpy as np
import nqsim as Q
import run as RN

OUT = '/data/Jarrel/nq-tfpred/nqalgo/runs'
NL, NE, NX, RB = Q.NL, Q.NE, Q.NX, Q.RB


class ShiftFC(Q.WindowFC):
    """forecast made one block earlier (row b-1) for the same absolute tokens: value(b, a, h) = F[b-1] over [a+16, a+16+h)"""
    def __init__(s, name):
        super().__init__(name); s.name = 'shift1:' + name
    def reset(s, tr):
        p = f'{Q.FC}/{tr.task}.{s.name[7:]}.npy'; s.path = p; super().reset(tr)
    def value(s, b, a, h):
        return Q.WindowFC.value(s, max(b - 1, 0), a + Q.G, h)


class EMAScore(Q.EMAFC):
    """EMA as a 64-token score source (score = 64 x rate), for tests of the score->value mapping without a file"""
    def score(s, b): return s.rate(b) * 64.0


class AnsFC:
    """answer-phase fix (expert-stability: think experts linger ~256 tok after </think>): for K tokens after a
    think->answer switch, blend the base rate with a bias-corrected fast EMA (half-life hl) of counts since </think>,
    weight 1 -> 0 linearly over K.  -> base rate elsewhere.  rate = hits/token; score = 64 x rate."""
    def __init__(s, base, K=256, hl=32): s.base = base; s.K = K; s.d = 0.5 ** (Q.G / hl); s.name = f'{base.name}+ans{K}h{hl}'
    def reset(s, tr):
        s.base.reset(tr); s.tr = tr; s.b = -1; s.E = np.zeros(NX); s.W = 0.0; s.since = None; s.n = len(tr.ans)
    def _adv(s, b):
        G = Q.G; A = s.tr.ans
        while s.b < b:
            s.b += 1; i0 = s.b * G; i1 = min(i0 + G, s.n) - 1
            if i0 >= s.n: continue
            a0 = bool(A[i0 - 1]) if i0 > 0 else False; a1 = bool(A[i1])
            if a1 and (not a0 or not bool(A[i0])):          # </think> inside / at the start of this block
                s.E[:] = 0; s.W = 0.0; s.since = 0
            if not a1: s.since = None
            if s.since is not None:
                s.E = s.E * s.d + s.tr.C[s.b].ravel(); s.W = s.W * s.d + G; s.since += G
    def rate(s, b):
        r = s.base.rate(b); s._adv(b)
        if s.since is not None and s.since < s.K and s.W > 0:
            w = 1.0 - s.since / s.K; r = ((1 - w) * r + w * s.E / s.W).astype(np.float32)
        return r
    def score(s, b): return s.rate(b) * 64.0


def _src(n):
    if n.endswith('a') and n[:-1] in ('jf', 'gbdt'): return AnsFC(_src(n[:-1]))
    if n in ('jf', 'gbdt'): return Q.ScoreFC(n)
    m = re.fullmatch(r'ema(\d+)', n)
    if m: return EMAScore(int(m[1]))
    raise KeyError(n)


class JFLandFC(Q.Forecast):
    """jF's single score S (predicted hits in the next span=64 tokens) mapped to hits after landing, no retraining.
    Same arithmetic as the serve TapScheduler._value (streaming/scheduler_tap.py):
      near: rate S/span inside [0, span);  far (beyond span): jfw -> tail x near rate; jfe<hl> -> EMA<hl> rate rescaled
      per layer to jF's per-layer mass;  V *= 512 / sum_e S[l]  (per-layer normalisation to hit units, so c is in hits
      whatever the score's scale: counts in the sim, salience in the serve)"""
    def __init__(s, src='jf', far=None, tail=0.1, span=64):
        s.J = _src(src); s.far = Q.EMAFC(far) if far else None; s.tail = tail; s.span = span
        s.name = f'{src}' + (f'+far:ema{far}' if far else f'+tail{tail}') + ',norm'; s._b = None
    def reset(s, tr):
        s.J.reset(tr); s._b = None
        if s.far: s.far.reset(tr)
    def _rates(s, b):
        if s._b != b:
            S = np.maximum(s.J.rate(b).reshape(NL, NE) * 64.0, 0); m = S.sum(1); ok = m > 0; mm = np.where(ok, m, 1.0)   # rate(): GBDT forced rows decoded (nqsim fix)
            rn = S / s.span
            if s.far:
                rf = s.far.rate(b).reshape(NL, NE); rs = rf.sum(1); rf = rf * np.where(rs > 0, (mm / s.span) / np.where(rs > 0, rs, 1), 0)[:, None]
            else:
                rf = s.tail * rn
            nrm = np.where(ok, 512.0 / mm, 0)[:, None]
            s._c = ((rn * nrm).ravel().astype(np.float32), (rf * nrm).ravel().astype(np.float32)); s._b = b
        return s._c
    def value(s, b, a, h):
        rn, rf = s._rates(b); sp = s.span; lo, hi = a, a + h
        near = max(0.0, min(hi, sp) - lo); farlen = max(0.0, hi - max(lo, sp))
        return rn * near + rf * farlen


class _FakeP:
    """predictor stub for the serve TapScheduler in the sim: S = the forecast file's score row of the block just closed"""
    def __init__(s, src): s.src = src; s.S = None; s.n = 0
    def step(s, c, ntok, *a, **k):
        s.n += ntok
        if s.n >= 16:
            s.n -= 16; s.S = (s.src.rate(s.b) * 64.0).reshape(NL, NE); return True
        return False
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


class SrvTapPolicy(Q.Policy):
    """the SERVE class streaming/scheduler_tap.TapScheduler driven inside the sim (sim clock; sim.rank_state() as
    io_all; executor feedback from sim state transitions) -> checks the deployable code against ext.py tap-jfe512."""
    def __init__(s, src, c=0.5, H=256, mla=1.0, svc_ms=None, name=None, far='ema'):
        s.far = far; s.src = _src(src); s.c = c; s.H = H; s.mla = mla; s.svc = svc_ms; s.name = name or f'srvtap[{src},c{c},H{H},mla{mla}]'
        s.stats = collections.Counter()
    def reset(s, sim):
        sys.path.insert(0, '/data/Jarrel/nq-tfpred/nq-src/streaming')
        os.environ.update(NQ_TAP_FAR=s.far, NQ_TAP_C=str(s.c), NQ_TAP_H=str(s.H), NQ_TAP_MLA=str(s.mla), NQ_TAP_TP='4',
                          NQ_TAP_SVC_MS=str(s.svc if s.svc is not None else (RB / sim._drv[0].B * 1e3 if sim._drv[0].B else 0.0) + sim.c['ovh_s'] * 1e3 + sim.c['fixed_lat_tok'] * sim.dt * 1e3))
        import scheduler_tap as TS
        s.src.reset(sim.tr); s.P = _FakeP(s.src); s.t = 0
        L_ = list(range(Q.L0, Q.L0 + NL)); fx = {L: [] for L in L_}; dflt = {L: [] for L in L_}
        s.S = TS.TapScheduler(L_, fx, dflt, RB * 4, NE=NE, n_float=Q.NF, slots=Q.NS, predictor=s.P, clock=lambda: s.t * sim.dt)
        s.S.state[:] = sim.st.reshape(NL, NE)
        def io_all():
            return {r: dict(delivered_GBps=d['achieved_Bps'] / 1e9, ops_outstanding=d['queued_ops'], slot_wait=d['slot_waits'],
                            backlog=d['follower_backlog']) for r, d in enumerate(sim.rank_state())}
        s.S.io_all = io_all
    def step(s, sim, t):
        S = s.S; st = sim.st.reshape(NL, NE); ss = S.state; s.t = t
        land = (ss == 1) & (st == 2)
        for l, e in zip(*np.nonzero(land)): S.landed(S.layers[l], int(e))
        rel = (ss == 3) & (st == 0); ss[rel] = 0
        fail = (ss == 1) & (st == 0); ss[fail] = 0
        b = (t + 1) // Q.G - 1; s.P.b = b
        c = sim.tr.C[b].reshape(NL, NE).astype(np.float64) if (t + 1) % Q.G == 0 else np.zeros((NL, NE))
        ups, downs = S.step(c, 1)
        return [S.li[L] * NE + e for L, e in ups], [S.li[L] * NE + e for L, e in downs], ()


def fcs(n):
    if n.startswith('shift1:'):
        return ShiftFC(n[7:])
    m = re.fullmatch(r'(jfa?|gbdta?|ema\d+?)(?:w|e(\d+))', n)
    if m:
        return JFLandFC(m[1], far=int(m[2]) if m[2] else None)
    return RN.fcs(n)


class TapPolicy(Q.ValuePolicy):
    def __init__(s, fc, c=0.5, H=256, Ha=None, ml=None, mla=None, cs=False, glob=False, ls=None, name=None):
        super().__init__(fc, H=H, c=c, max_lat_tok=None, cancel_stale=cs, glob=glob, name=name or 'tap')
        s.Ha = Ha; s.mlx = ml; s.mla = mla; s.peak = None; s.ls = ls
        if ls is not None:
            s.nfl = np.array(json.load(open(f'/data/Jarrel/nq-tfpred/nqalgo/slots_{ls}.json'))['nf'], int)
            assert len(s.nfl) == NL
    def reset(s, sim):
        super().reset(sim); s.peak = np.zeros(sim.R); s.stats = collections.Counter()
    def step(s, sim, t):
        if (t + 1) % s.R == 0:
            lat = sim.est_land_tok()
            if s.Ha is not None:
                s.H = float(np.clip(s.Ha * lat, 64, 1024))
            s.stats['H_sum'] += s.H
            # measured per-rank read rate (peak-hold of the ~1 s achieved EMA, decaying slowly so a slower drive is seen)
            rs = sim.rank_state()
            ach = np.array([r['achieved_Bps'] for r in rs])
            s.peak = np.maximum(s.peak * 0.999, ach)
            ml = s.mlx if s.mla is None else s.mla * s.H
            s._ml_now = ml
        return super().step(sim, t)
    # the budget: override ValuePolicy's use of the configured drive rate with the measured one
    def _budget(s, sim):
        if s._ml_now is None:
            return 10 ** 9
        el = sim.est_land_tok(); per = RB * (sim.nshare[0] if sim.mirror else 1)
        r = float(s.peak.min()) if s.peak is not None and s.peak.min() > 0 else None
        if r is None:
            d = sim._drv[0]; r = (d.B / max(1, d.n_active() + (d.live[0] == 0))) if d.B else None
        return 10 ** 9 if r is None else max(0, int((s._ml_now - el) * sim.dt * r / per))
    def _pairs(s, V, cand, res, occ):
        out = s._pairs_ls(V, cand, res, occ) if s.ls is not None else super()._pairs(V, cand, res, occ)
        # apply the measured-rate budget by truncating the gain-sorted pair list (ValuePolicy's own ml budget is off)
        b = s._budget(s._sim)
        if len(out) > b:
            s.stats['budget_cut'] += len(out) - b; out = out[:b]
        return out
    def _pairs_ls(s, V, cand, res, occ):
        out = []
        Vl = V.reshape(NL, NE); cl = cand.reshape(NL, NE); rl = res.reshape(NL, NE); oc = occ.reshape(NL, NE).sum(1)
        ce = np.argsort(-np.where(cl, Vl, -np.inf), 1, kind='stable'); rv = np.argsort(np.where(rl, Vl, np.inf), 1, kind='stable')
        ncand = cl.sum(1); nres = rl.sum(1)
        for l in range(NL):
            free = int(s.nfl[l]) - int(oc[l]); k = 0; base = l * NE
            while k < ncand[l]:
                e = ce[l, k]; ve = Vl[l, e]
                if k < free:
                    if ve > s.cc: out.append((ve, base + e, -1)); k += 1; continue
                    break
                j = k - max(free, 0)
                if j >= nres[l]: break
                v = rv[l, j]; g = ve - Vl[l, v]
                if g <= s.cc: break
                out.append((g, base + e, base + v)); k += 1
        out.sort(key=lambda z: -z[0]); return out


class _SimRef:
    """ValuePolicy._pairs has no sim argument: stash it per step"""


class LandWinFC:
    """TF window forecast as the serve TFGPUPredictor scores it: S = expected hits in [lat, lat+H) after the block
    (NQ_TF_LAT / NQ_TF_H), fed to the prod cur scheduler (SchedPolicy) -> candidate row on the identical setup"""
    def __init__(s, name, lat=64, H=256): s.W = Q.WindowFC(name); s.lat = lat; s.H = H; s.name = f'{name}[{lat},+{H}]'
    def reset(s, tr): s.W.reset(tr)
    def score(s, b): return s.W.value(b, s.lat, s.H)
    def value(s, b, a, h): return s.W.value(b, a, h)


def pol(p):
    m = re.fullmatch(r'curwin-(\w+)-a(\d+)-h(\d+)', p)
    if m: return Q.SchedPolicy(LandWinFC(m[1], int(m[2]), int(m[3])), name='curwin:' + p)
    if p in ('cur-jfa', 'cur-gbdta'): return Q.SchedPolicy(_src(p[4:]))
    m = re.fullmatch(r'srvtap-(jfa?|gbdt|ema\d+?)(w?)-c([\d.]+)-H(\d+)-mla([\d.]+)', p)
    if m:
        return SrvTapPolicy(m[1], c=float(m[3]), H=int(m[4]), mla=float(m[5]), name='srvtap:' + p, far='tail:0.1' if m[2] else 'ema')
    m = re.fullmatch(r'tap-(.+?)-c([\d.]+)((?:-H\d+|-Ha[\d.]+)?)((?:-ml\d+|-mla[\d.]+)?)((?:-cs)?)((?:-g)?)((?:-ls\w+)?)', p)
    if not m:
        return RN.pol(p) if not p.startswith(('val-shift1', 'cur-shift1')) else None
    H = 256; Ha = None
    if m[3].startswith('-Ha'): Ha = float(m[3][3:])
    elif m[3]: H = int(m[3][2:])
    ml = mla = None
    if m[4].startswith('-mla'): mla = float(m[4][4:])
    elif m[4]: ml = int(m[4][3:])
    P = TapPolicy(fcs(m[1]), c=float(m[2]), H=H, Ha=Ha, ml=ml, mla=mla, cs=bool(m[5]), glob=bool(m[6]),
                  ls=m[7][3:] if m[7] else None, name='tap:' + p)
    return P


def main():
    cfg, task = sys.argv[1], sys.argv[2]; mt = int(sys.argv[3]) if len(sys.argv) > 3 else None
    p, e = cfg.split('@'); os.makedirs(OUT, exist_ok=True)
    out = f'{OUT}/{cfg}.{task}{"." + str(mt) if mt else ""}.json'
    if os.path.exists(out): return
    tr = Q.Trace(task, max_tokens=mt); a = time.time()
    m = re.fullmatch(r'(val|cur)-(shift1:[^-]+)(.*)', p)
    if m:                                   # run.py policy grammar with a shift1 forecast
        RN_fcs = RN.fcs; RN.fcs = fcs
        try: P = RN.pol(m[1] + '-' + m[2] + m[3])
        finally: RN.fcs = RN_fcs
    else:
        P = pol(p)
    kw = RN.env(e)
    S = Q.Sim(tr, P, **kw)
    if isinstance(P, TapPolicy):
        P._sim = S; P._ml_now = None
    r = S.run(); r.update(cfg=cfg, policy_name=P.name, env=kw, wall_s=time.time() - a, pstats=dict(getattr(P, 'stats', {})))
    json.dump(r, open(out + '.tmp', 'w'), default=float); os.replace(out + '.tmp', out)
    print(cfg, task, 'done', round(r['wall_s']), 's share_all', round(r['share_all'], 4), flush=True)


if __name__ == '__main__':
    main()
