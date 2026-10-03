"""NQ_SCHED=tap: throughput-aware value scheduling of the floating set (nq-tfpred, thread 36), same predictor (jF).
Only the candidate ordering / budgeting changes; the predictor, executor, level ops and feedback calls are those of
scheduler.Scheduler (unset NQ_SCHED = the base class, untouched).  Leader-only decisions (wall-clock inputs): use with
NQ_LEADER=1 (TP>1) or TP=1.

Every predictor refresh (16 tokens, P.step() -> True):
  lat  = estimated landing time of a read issued now, in tokens, on the slowest rank:
         rank r: queued_r / rate_r + svc, queued_r = reads outstanding + waiting for a slot (+ follower backlog);
         rate_r = peak-hold of the delivered read rate (records/s); rank 0 is measured live here (landed() callbacks
         and in-flight count), followers from Scheduler.io_all() (nq-io stats API; {} -> rank 0 only)
  V(x) = expected hits in [lat, lat + H) after the end of the block, two horizons, normalised per layer to hit units:
         near: jF score S (predicted hits in the next 64 tokens) as a flat rate inside [0, 64)
         far : the scheduler's EMA512 rate, rescaled per layer to jF's mass, beyond 64 tokens
               (NQ_TAP_FAR=tail:<x>: x times the jF rate instead; sim tap-jfw = tail:0.1)
         V *= 512 / sum_e S[l]   (jF mass of a layer = 64 tok x top-8)
  V0(v)= the victim's expected hits during the flight [0, lat)
  Per layer, pair the best idle candidates with the weakest landed residents while V(e) - V(v) > c (one read per
  promotion; the layer keeps at most n_float occupied). Lazy eviction: v stays at 4-bit until e lands; with no free slot
  v is downed now only if V(e) - V(v) - V0(v) > c, and e is issued when a slot frees.
  Issue budget: only as many reads as land within mla x H tokens at the measured rate (no backlog); the sorted pair
  list is truncated (highest gain first).
Env: NQ_TAP_C (1.0 hits) NQ_TAP_H (256 tok) NQ_TAP_HA (0 = fixed H; k -> H = clip(k x lat, 64, 1024))
     NQ_TAP_MLA (1.0; 0 = no budget) NQ_TAP_RATE_GBPS (6, per rank, before the first measurement) NQ_TAP_SVC_MS (2)
     NQ_TAP_FAR (tail:0.1 default | ema) NQ_TAP_TP (ranks, for the per-rank record size when io_all is empty; default 4)
     NQ_TAP_LAT (model default | meas): landing latency per rank. model = queued reads / peak rate + NQ_TAP_SVC_MS.
       meas (nq-tapc adaptive lookahead) = max(model, not-yet-issued reads / peak rate + MEASURED issue->land latency):
       rank 0 from its own landings (step() emits the up -> landed(); NQ_TAP_LAT_Q quantile of the last NQ_TAP_LAT_N=256),
       followers from io_all op_p50_ms (engine issue->land) + (backlog + slot_wait) / peak; so a fixed per-read latency
       (SSD service, overheads, h2d) the queue model does not know shifts the value window to where reads really land.
       With meas, NQ_TAP_HA > 0 uses H = clip(HA x lat_ema, NQ_TAP_H, 1024) (lat_ema: EWMA over refreshes with half-life
       NQ_TAP_LAT_HL refreshes, default 4; floor NQ_TAP_H instead of 64 so a short lat never shrinks the window).
       The issue budget keeps using the queue-model latency (only queueing consumes the window; a measured fixed
       latency is pipelined), so a large measured latency never stalls issuing.
Step 3 policy arm (p-online Combo g25.9.25+B5n22; default OFF -> bit-identical to step 2 rc). Env at boot and/or NQ_TAP_CTL
keys at runtime (same-boot A/B). All state is per request (reset at new_request) and counts only decode steps (ntok <= 16):
     s3     (NQ_S3, 0)          master switch; 0 = exact step 2 rc
     s3b    (NQ_S3_B, .25)      count tilt   s = share(share(max(jF,0)) * (Cb + 1e-3)^b), Cb = count EMA hl s3bhl (0 off)
     s3bhl  (NQ_S3_BHL, 256)
     s3a    (NQ_S3_A, .9)       count mix    s = a s + (1-a) share(Ca), Ca = count EMA hl s3ahl (1 off)
     s3ahl  (NQ_S3_AHL, 64)
     s3p    (NQ_S3_P, .25)      layer weight v = 512 s (lm_l / mean lm)^p, lm = cumulative routed salience / (8 rows) since
                                request start (skipped until every layer has lm > 0; needs the rank-0 salience export)
     s3band (NQ_S3_BAND, 22)    band over layer idx < n (L3..L24; 0 off): v[l] = m_l ((1-bw) share(v[l]) + bw share(C64[l] + 1e-6)),
     s3bw   (NQ_S3_BW, .5)        m_l = sum_e v[l], C64 = count EMA hl 64 (band layers only)
     s3bk   (NQ_S3_BK, .5)        then v[l] *= bk
     s3rst  (NQ_S3_RESET, 1)    per-request reset (0 = state runs across requests, = the p-online sim streams)
  external score (model hook, default off; applied as pols.Combo e1, after the count mix, before x 512):
     s3x    (NQ_S3_X, 0)        1 = s = (1-xw) s + xw share(max(X, 0)), X = fn(ctx) [NL, NE] at each refresh (None = skip)
     s3xw   (NQ_S3_XW, 0.15)
     s3xf   (NQ_S3_EXT, '')     /path/module.py:fn (loaded on first use; a ctl s3xf= loads another one at runtime)
     s3xpos (NQ_S3_XPOS, 0)     0 = blend as above (Combo e1); 1 = at the end on the finished v (after p / band / bk), layer mass
                                kept: v[l] = m_l ((1-xw) share(v[l]) + xw share(max(X[l], 0)))  (= (nr(v) + beta nr(X)) m with xw = beta/(1+beta))
     s3xrst (NQ_S3_XRESET, 1)   0 = the history ring is not reset at request start (rolling across requests; rows / new_request still per request)
     NQ_S3_XRING (1024)         rows of per-request decode history kept for fn (kept whenever an fn path is set)
     fn(ctx) gets a TapExtCtx: ids [n, NL, 8] int16 / sal [n, NL, 8] float32 = the last n <= ring routed rows of the current
     request, oldest first (ids ascending within a row; a step with ntok > 1 tokens is split into ntok rows by
     multiplicity), nnew (rows pushed since the previous call), rows (decode rows since request start), trows (rows pushed since the ring was made), new_request (a request started since the previous call),
     S (jF, float32), Ca / Cb / C64 (count EMAs), tok, layers, fixed, state (dict kept for fn across calls). An
     exception disables the hook until the next s3xf / s3x ctl change (logged); cost: s.s3xt = [calls, seconds].
  Live mapping: the shape of v goes in as the admission S (float32; _value normalises it per layer over the floating
  experts as before) and the layer scale as a pair-gain weight lw_l = floating mass of v_l / 512, so the floating
  values the global sort / budget cut / eager test see are v itself (x the common H window factor). Tap c is a ctl key
  too (c=0 with the arm). NQ_S3_TRACK=1 keeps the state running while s3=0 (and turns the salience export on at boot,
  which is decided at boot: s3=1 with s3p>0 at boot does too). Safer arm via ctl only: s3=1 s3b=0.35 s3a=1 (s3band=0
  for plain g35.1.25). tests/test_s3_adm.py: live keys = p-online pols.Combo, budget_sim numbers reproduced.
Mirror of nq-tfpred/nqalgo ext.py tap-jfe512-c<c>-H<H>-mla<k> (JFLandFC normalised two-horizon value)."""
import os, time, collections
import numpy as np
from scheduler import Scheduler


class TapExtCtx:
    """argument of the NQ_S3_EXT hook (fields: see the module doc)"""
    __slots__ = ('ids', 'sal', 'nnew', 'rows', 'trows', 'new_request', 'S', 'Ca', 'Cb', 'C64', 'tok', 'layers', 'fixed', 'state')


class TapScheduler(Scheduler):
    def __init__(s, *a, clock=None, **k):
        super().__init__(*a, **k)
        # NQ_HOSTLOOP=cpp (or hostloop='cpp'): the array work runs in nqhost.TapCore (nq-tapc, bit-exact vs this Python
        # step: tests/test_tap_parity.py); s.core (SchedCore) is kept for the score decay only
        s.tcore = None
        if s.core is not None:
            import hostcore; s.tcore = hostcore.mod().TapCore(s.layers, s.NE)
        else:
            s.core = None
        e = os.environ.get
        s.tc = float(e('NQ_TAP_C', '1.0')); s.tH = float(e('NQ_TAP_H', '256')); s.tHa = float(e('NQ_TAP_HA', '0'))
        s.tmla = float(e('NQ_TAP_MLA', '1.0')); s.tsvc = float(e('NQ_TAP_SVC_MS', '2')) / 1e3
        fr = e('NQ_TAP_FAR', 'tail:0.1'); s.tfar = float(fr.split(':')[1]) if fr.startswith('tail:') else None   # far window: EMA512 | tail:<x> = x * jF rate
        s.tp_n = int(e('NQ_TAP_TP', '4')); s.rb_rank = s.rb / max(1, s.tp_n)
        s.rate0 = float(e('NQ_TAP_RATE_GBPS', '6')) * 1e9 / s.rb_rank      # records/s per rank before measurement
        # nq-kld (default off): NQ_TAP_RATE_FLOOR_GBPS per rank = lower bound of the peak-held rate (the 0.999 decay can't pull
        # it under); in-boot override of floor / mla / lat via /dev/shm/nq_tap_ctl ("floor=<GB/s> mla=<x> lat=model|meas"), re-read on change
        s.tfloor = float(e('NQ_TAP_RATE_FLOOR_GBPS', '0')) * 1e9 / s.rb_rank; s.tctl = e('NQ_TAP_CTL', '/dev/shm/nq_tap_ctl'); s.tctl_m = None
        # nq-kld (default off): NQ_TAP_PEAK_GBPS = TOTAL (all ranks) cap on the held peak rate (and rate0) the issue budget and
        # landing-latency model use (nq-io STAGE.md option #1, mirror of stage_sim +cap<GBps>); ctl key cap=<GB/s total>, 0 = off
        s.tcap = 0.0; s._setcap(float(e('NQ_TAP_PEAK_GBPS', '0')))
        # nq-kld (default off): NQ_TAP_QREAL=1 / ctl qreal=1: the latency model's rank-0 queue = the executor's real outstanding
        # upgrades (s.xq(), set by nq_vllm: engine ops + slot waiters) + todo, instead of the count of state-1 experts (which
        # can include ups whose landing was never reported, e.g. superseded hseq)
        s.tqreal = e('NQ_TAP_QREAL', '0') == '1'; s.xq = None
        s.clock = clock or time.monotonic
        s.tlat = e('NQ_TAP_LAT', 'model'); assert s.tlat in ('model', 'meas'), s.tlat
        s.tlat_q = float(e('NQ_TAP_LAT_Q', '0.5')); s.tlat_hl = float(e('NQ_TAP_LAT_HL', '4'))
        s.t_iss = None; s.lat0 = None; s.lat_ema = None
        if s.tlat == 'meas':
            s.t_iss = np.full((len(s.layers), s.NE), np.nan); s.lat0 = collections.deque(maxlen=int(e('NQ_TAP_LAT_N', '256')))
        s.span = 64.0; s.NL = len(s.layers)
        s.doom = {}; s.doomed = np.zeros((s.NL, s.NE), bool); s.todo = collections.deque()
        s.peak = None; s.t_last = None; s.tps = None; s.n_land = 0; s.t_rate = None; s.n_land0 = 0
        s.stats.update(refreshes=0, promotions=0, budget_cut=0, eager_evict=0, no_slot_skip=0, lat_tok_sum=0.0)
        if s.tlat == 'meas': s.stats.update(lat_model_tok_sum=0.0, H_sum=0.0, lat0_n=0)
        s.sched_name = 'tap'
        # step 3 policy arm (default off; see module doc)
        s.s3 = dict(on=int(float(e('NQ_S3', '0') or 0)), b=float(e('NQ_S3_B', '0.25') or 0), bhl=float(e('NQ_S3_BHL', '256') or 256),
                    a=float(e('NQ_S3_A', '0.9') or 0), ahl=float(e('NQ_S3_AHL', '64') or 64), p=float(e('NQ_S3_P', '0.25') or 0),
                    band=int(float(e('NQ_S3_BAND', '22') or 0)), bw=float(e('NQ_S3_BW', '0.5') or 0), bk=float(e('NQ_S3_BK', '0.5') or 0),
                    rst=int(float(e('NQ_S3_RESET', '1') or 0)), x=int(float(e('NQ_S3_X', '0') or 0)), xw=float(e('NQ_S3_XW', '0.15') or 0),
                    xf=e('NQ_S3_EXT', '') or '', xpos=int(float(e('NQ_S3_XPOS', '0') or 0)), xrst=int(float(e('NQ_S3_XRESET', '1') or 0)))
        s.s3xring = int(e('NQ_S3_XRING', '1024') or 1024); s.s3xfn = None; s.s3xf0 = None; s.s3xt = [0, 0.0]; s.s3xbad = 0; s.s3xl8 = np.repeat(np.arange(s.NL), 8)
        s.s3track = e('NQ_S3_TRACK', '0') == '1'
        s.s3_sal = s.s3track or (s.s3['on'] == 1 and s.s3['p'] > 0)   # nq_vllm: rank 0 exports decode salience (step(..., sal=))
        s.s3st = None; s.s3gen = 0; s.s3c = None; s._s3cfg()

    def _setcap(s, g):
        if not hasattr(s, 'rate00'): s.rate00 = s.rate0
        s.tcap = g * 1e9 / max(1, s.tp_n) / s.rb_rank if g > 0 else 0.0
        s.rate0 = min(s.rate00, s.tcap) if s.tcap > 0 else s.rate00
        if s.tcap > 0 and getattr(s, 'peak', None) is not None: np.minimum(s.peak, s.tcap, out=s.peak)

    # ---- executor feedback (counts landings for the live rank-0 rate)
    def landed(s, L, e):
        i = s.li[L]
        if s.state[i, e] == 1:
            s.n_land += 1
            if s.lat0 is not None and s.t_iss[i, e] == s.t_iss[i, e]:      # not for ups set in flight outside step()
                s.lat0.append(s.clock() - s.t_iss[i, e]); s.t_iss[i, e] = np.nan; s.stats['lat0_n'] += 1
        super().landed(L, e)

    # ---- live I/O state
    def _plan(s, now, q0, nt=0):
        """refresh scalars: -> (lat tokens, H tokens, issue budget reads); q0 = rank-0 reads queued (in flight + todo), nt = todo only"""
        tps = s.tps or 94.0
        s._ctl()
        if s.tlat == 'meas':
            mo, me = s._rates(now, q0, nt)
            lm = max(mo) * tps; lat = max(lm, max(me) * tps)
            a = 0.5 ** (1.0 / max(s.tlat_hl, 1e-9)); s.lat_ema = lat if s.lat_ema is None else a * s.lat_ema + (1 - a) * lat
            H = float(np.clip(s.tHa * s.lat_ema, s.tH, 1024)) if s.tHa > 0 else s.tH
            s.stats['lat_model_tok_sum'] += lm; s.stats['H_sum'] += H
        else:
            lat = max(s._rates(now, q0)) * tps
            H = float(np.clip(s.tHa * lat, 64, 1024)) if s.tHa > 0 else s.tH
        s.stats['refreshes'] += 1; s.stats['lat_tok_sum'] += lat
        budget = 10 ** 9
        if s.tmla > 0:
            el = lm if s.tlat == 'meas' else lat     # meas: a fixed per-read latency shifts the window, costs no bandwidth
            rate = float(s.peak.min()) if s.peak is not None else s.rate0
            budget = max(0, int((s.tmla * H - el) / tps * rate))
        return lat, H, budget

    def _ctl(s):
        try: m = os.stat(s.tctl).st_mtime_ns
        except OSError: return
        if m == s.tctl_m: return
        s.tctl_m = m
        try: kv = dict(x.split('=', 1) for x in open(s.tctl).read().split() if '=' in x)
        except (OSError, ValueError): return
        if 'floor' in kv: s.tfloor = float(kv['floor']) * 1e9 / s.rb_rank
        if 'mla' in kv: s.tmla = float(kv['mla'])
        if 'cap' in kv: s._setcap(float(kv['cap']))
        if 'qreal' in kv: s.tqreal = kv['qreal'] == '1'
        if 'c' in kv: s.tc = float(kv['c'])
        k3 = {'s3': 'on', 's3b': 'b', 's3bhl': 'bhl', 's3a': 'a', 's3ahl': 'ahl', 's3p': 'p', 's3band': 'band', 's3bw': 'bw', 's3bk': 'bk', 's3rst': 'rst',
              's3x': 'x', 's3xw': 'xw', 's3xf': 'xf', 's3xpos': 'xpos', 's3xrst': 'xrst'}
        if any(k in kv for k in k3):
            for k, n in k3.items():
                if k in kv: s.s3[n] = kv[k] if n == 'xf' else int(float(kv[k])) if n in ('on', 'band', 'rst', 'x', 'xpos', 'xrst') else float(kv[k])
            if s.s3xfn is None or s.s3['xf'] != s.s3xf0: s.s3xf0 = None   # (re)load on a new s3xf path / re-enable after an error (else fn + state kept)
            s._s3cfg()
        if kv.get('lat') in ('model', 'meas') and kv['lat'] != s.tlat:
            s.tlat = kv['lat']
            if s.tlat == 'meas' and s.t_iss is None:
                s.t_iss = np.full((len(s.layers), s.NE), np.nan); s.lat0 = collections.deque(maxlen=int(os.environ.get('NQ_TAP_LAT_N', '256')))
                s.stats.update(lat_model_tok_sum=0.0, H_sum=0.0, lat0_n=0)
            elif s.tlat == 'model': s.t_iss = None; s.lat0 = None
        import logging; logging.getLogger('vllm.nestquant').warning('NestQuant tap ctl: floor %.2f GB/s/rank, mla %s, lat %s, cap %.2f GB/s total, qreal %s, c %s, s3 %s (sal %s)', s.tfloor * s.rb_rank / 1e9, s.tmla, s.tlat, s.tcap * s.rb_rank * max(1, s.tp_n) / 1e9, s.tqreal, s.tc, s.s3, s.s3_sal)

    def _rates(s, now, q0, nt=None):
        """-> (lat_s per rank list). Rank 0 live; followers from io_all() if available.
        nt (todo length) given: -> (model list, measured list): measured = pre-issue queue / peak + measured issue->land latency (model
        value where no measurement exists)"""
        if s.t_rate is None: s.t_rate = now; s.n_land0 = s.n_land
        dt = now - s.t_rate
        out = []
        if dt >= 0.25:                                   # rank-0 delivered rate over >= 250 ms
            r0 = (s.n_land - s.n_land0) / dt; s.t_rate = now; s.n_land0 = s.n_land
        else:
            r0 = None
        io = {}
        f = getattr(s, 'io_all', None)
        if f is not None:
            try: io = f() or {}
            except Exception: io = {}
        nr = max(len(io), 1)
        if s.peak is None or len(s.peak) != nr: s.peak = np.full(nr, s.rate0)
        s.peak *= 0.999
        if s.tfloor > 0: np.maximum(s.peak, s.tfloor, out=s.peak)
        if s.tcap > 0: np.minimum(s.peak, s.tcap, out=s.peak)
        if r0 is not None: s.peak[0] = max(s.peak[0], r0)
        out.append(q0 / max(s.peak[0], 1.0) + s.tsvc); me = []
        meas = nt is not None
        if meas:
            m0 = float(np.quantile(np.fromiter(s.lat0, float, len(s.lat0)), s.tlat_q)) if len(s.lat0) >= 8 else None
            me.append(out[0] if m0 is None else nt / max(s.peak[0], 1.0) + min(max(m0, 0.0), 30.0))
        for r in sorted(io):
            if r == 0 or r >= nr: continue
            d = io[r]; rbr = s.rb / nr
            g = d.get('delivered_GBps') or 0.0
            s.peak[r] = max(s.peak[r], g * 1e9 / rbr)
            pre = (d.get('slot_wait') or 0) + (d.get('backlog') or 0)
            q = (d.get('ops_outstanding') or 0) + pre
            out.append(q / max(s.peak[r], 1.0) + s.tsvc)
            if meas:
                p = d.get('op_p50_ms')
                me.append(out[-1] if not isinstance(p, (int, float)) or not p == p else
                          pre / max(s.peak[r], 1.0) + min(max(p / 1e3, 0.0), 30.0))
        if s.tcap > 0: np.minimum(s.peak, s.tcap, out=s.peak)
        return (out, me) if meas else out

    # ---- step 3 policy arms (default off)
    def _s3cfg(s):
        """(re)derive the arm state after an env / ctl change. State (float32 count EMAs as pols.Combo, float64 salience sums)
        is kept while the arm is on or NQ_S3_TRACK=1, dropped when off and untracked (restarts from zero); an EMA restarts
        when its half-life changes."""
        P = s.s3; s.s3on = P['on'] == 1; s.s3any = s.s3on or s.s3track; s.s3gen += 1
        if not s.s3any: s.s3st = None; return
        if s.s3st is None or s.s3st['bhl'] != P['bhl'] or s.s3st['ahl'] != P['ahl']: s._s3reset()
        if P['xf'] and s.s3st.get('xi') is None: s._s3xreset()

    def _s3reset(s):
        P = s.s3; z = lambda: np.zeros((s.NL, s.NE), np.float32)
        s.s3st = dict(bhl=P['bhl'], ahl=P['ahl'], db=np.float32(0.5 ** (1 / P['bhl'])), da=np.float32(0.5 ** (1 / P['ahl'])),
                      d64=np.float32(0.5 ** (1 / 64)), Cb=z(), Ca=z(), C64=z(), ls=np.zeros(s.NL), lc=np.zeros(s.NL), xi=None)
        if P['xf']: s._s3xreset()
        s.s3gen += 1

    def _s3xreset(s):
        R = max(16, s.s3xring); Z = s.s3st
        Z.update(xi=np.zeros((2 * R, s.NL, 8), np.int16), xs=np.zeros((2 * R, s.NL, 8), np.float32), xp=0, xn=0, xnn=0, xrows=0, xtot=0, xnew=True)

    def _s3xpush(s, c, ntok, sal):
        """append this step's routed rows to the per-request history (linear buffer of 2 ring, compacted when full)"""
        Z = s.s3st; c = np.asarray(c); f = np.flatnonzero(c.ravel() != 0); l = f // s.NE; e = f - l * s.NE   # (np.nonzero on float is ~10x slower)
        if ntok == 1 and len(l) == 8 * s.NL and np.array_equal(l, s.s3xl8) and (c[l, e] == 1).all():   # fast path: one decode row
            ids = e.reshape(1, s.NL, 8); sv = np.asarray(sal)[l, e].reshape(1, s.NL, 8) if sal is not None else 0.0
        else:
            k = c[l, e].astype(np.int64)
            if len(k) == 0 or (np.bincount(l, weights=k, minlength=s.NL) != 8 * ntok).any(): s.s3xbad += 1; return
            ids = np.repeat(e, k).reshape(s.NL, 8, ntok).transpose(2, 0, 1)        # duplicates of an expert go to different rows
            sv = np.repeat(np.asarray(sal, np.float64)[l, e] / k, k).reshape(s.NL, 8, ntok).transpose(2, 0, 1) if sal is not None else 0.0
        R = Z['xi'].shape[0] // 2
        if Z['xp'] + ntok > 2 * R:
            m = min(Z['xn'], R); Z['xi'][:m] = Z['xi'][Z['xp'] - m:Z['xp']]; Z['xs'][:m] = Z['xs'][Z['xp'] - m:Z['xp']]; Z['xp'] = m
        p = Z['xp']; Z['xi'][p:p + ntok] = ids; Z['xs'][p:p + ntok] = sv; Z['xp'] = p + ntok
        Z['xn'] = min(Z['xn'] + ntok, R); Z['xrows'] += ntok; Z['xnn'] += ntok; Z['xtot'] += ntok

    def _s3xcall(s):
        """the external score X [NL, NE] (float32) or None"""
        P = s.s3; Z = s.s3st
        if s.s3xf0 != P['xf']:
            s.s3xf0 = P['xf']; s.s3xfn = None
            try:
                import importlib.util; path, _, fn = P['xf'].rpartition(':')
                sp = importlib.util.spec_from_file_location('nq_s3ext_%d' % s.s3gen, path); m = importlib.util.module_from_spec(sp)
                sp.loader.exec_module(m); s.s3xfn = getattr(m, fn); s.s3xstate = {}
            except Exception as ex:
                import logging; logging.getLogger('vllm.nestquant').warning('NestQuant tap s3 ext: load %r failed: %r', P['xf'], ex)
        if s.s3xfn is None or Z.get('xi') is None: return None
        cx = TapExtCtx(); p = Z['xp']; n = Z['xn']
        cx.ids = Z['xi'][p - n:p]; cx.sal = Z['xs'][p - n:p]; cx.rows = Z['xrows']; cx.trows = Z['xtot']; cx.new_request = Z['xnew']; Z['xnew'] = False
        cx.nnew = min(Z['xnn'], n); Z['xnn'] = 0
        cx.S = s.P.S; cx.Ca = Z['Ca']; cx.Cb = Z['Cb']; cx.C64 = Z['C64']; cx.tok = s.tok; cx.layers = s.layers; cx.fixed = s.fixed
        cx.state = s.s3xstate
        t0 = time.perf_counter()
        try: X = s.s3xfn(cx)
        except Exception as ex:
            import logging; logging.getLogger('vllm.nestquant').warning('NestQuant tap s3 ext: %r raised %r, hook off until the next ctl s3x/s3xf', P['xf'], ex)
            s.s3xfn = None; X = None
        s.s3xt[0] += 1; s.s3xt[1] += time.perf_counter() - t0
        return None if X is None else np.asarray(X, np.float32).reshape(s.NL, s.NE)

    def _s3_add(s, c, ntok, sal, new_request=False):
        """per-request state update from this step's routed counts (+ salience); pols.Combo.observe per row"""
        if new_request and s.s3['rst']:
            X = s.s3st.get('xi') is not None and not s.s3['xrst'] and {k: s.s3st[k] for k in ('xi', 'xs', 'xp', 'xn', 'xnn', 'xtot')}
            s._s3reset()
            if X: s.s3st.update(X)
        if new_request and s.s3st.get('xi') is not None: s.s3st['xrows'] = 0; s.s3st['xnew'] = True
        if ntok > 16: return
        Z = s.s3st; cf = np.asarray(c, np.float32)
        if Z.get('xi') is not None: s._s3xpush(c, ntok, sal)
        for k, d in (('Cb', 'db'), ('Ca', 'da'), ('C64', 'd64')):
            E = Z[k]; E *= Z[d] if ntok == 1 else np.float32(Z[d] ** ntok); E += cf
        if sal is not None: Z['ls'] += np.asarray(sal, np.float64).sum(1); Z['lc'] += np.asarray(c, np.float64).sum(1)

    @staticmethod
    def _sh(x):
        m = x.sum(1, keepdims=True); return x / np.where(m > 0, m, 1)

    def _s3v(s):
        """the arm's admission values v [NL, NE] (float64), op for op pols.Combo.score (mass='512', lwsrc='cum')"""
        P = s.s3; Z = s.s3st; sh = s._sh
        j = np.maximum(np.nan_to_num(np.asarray(s.P.S, np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0); v = sh(j)
        if P['b']: v = sh(v * (Z['Cb'] + 1e-3) ** P['b'])
        if P['a'] < 1: v = P['a'] * v + (1 - P['a']) * sh(Z['Ca'])
        if P['x'] and P['xw'] and P['xf'] and not P['xpos']:
            X = s._s3xcall()
            if X is not None: v = (1 - P['xw']) * v + P['xw'] * sh(np.maximum(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0))
        v = v * 512.0
        if P['p']:
            lm = Z['ls'] / np.maximum(Z['lc'], 1e-9)
            if (lm > 0).all(): v = v * ((lm / lm.mean()) ** P['p'])[:, None]
        n = min(P['band'], s.NL)
        if n > 0 and P['bw']:
            v = v.copy(); m = v[:n].sum(1, keepdims=True)
            v[:n] = m * ((1 - P['bw']) * sh(v[:n]) + P['bw'] * sh(Z['C64'][:n] + 1e-6))
        if n > 0 and P['bk'] != 1: v = v.copy(); v[:n] *= P['bk']
        if P['x'] and P['xw'] and P['xf'] and P['xpos']:
            X = s._s3xcall()
            if X is not None:
                m = v.sum(1, keepdims=True); v = m * ((1 - P['xw']) * sh(v) + P['xw'] * sh(np.maximum(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0)))
        return v

    def _s3val(s):
        """(admission S float32, pair-gain layer weight float64), cached per refresh"""
        key = (s.tok, s.s3gen, id(s.P.S))
        if s.s3c is None or s.s3c[0] != key:
            S = s._s3v().astype(np.float32)
            fm = np.where(s.fixed, 0, np.maximum(S, 0)).sum(1).astype(np.float32)   # = _value's mass (floating experts)
            s.s3c = (key, S, fm.astype(np.float64) / 512.0)
        return s.s3c[1], s.s3c[2]

    def _srcS(s):
        """admission source for _value: jF S (arm off), else the arm's v (float32)"""
        return s._s3val()[0] if s.s3on else s.P.S

    def _lw(s):
        """per-layer pair-gain weight (arm on), else None"""
        return s._s3val()[1] if s.s3on else None

    def _value(s, lat, H):
        S = np.asarray(s._srcS(), np.float32); S = np.where(s.fixed, 0, np.maximum(S, 0))
        mass = S.sum(1).astype(np.float32); ok = mass > 0; m = np.where(ok, mass, 1.0)
        rn = S / s.span
        if s.tfar is not None:
            rf = s.tfar * rn
        else:
            rf = (s.score * (1 - s.a)).astype(np.float32); rf = np.where(s.fixed, 0, rf)
            rs = rf.sum(1); rf = rf * np.where(rs > 0, (m / s.span) / np.where(rs > 0, rs, 1), 0)[:, None]
        def v(lo, h):
            hi = lo + h; near = max(0.0, min(hi, s.span) - lo); far = max(0.0, hi - max(lo, s.span))
            return rn * near + rf * far
        nrm = (512.0 / m)[:, None]
        V = v(lat, H) * nrm; V0 = v(0.0, lat) * nrm if lat >= 1 else np.zeros_like(V)
        V[~ok] = 0; V0[~ok] = 0
        return V, V0

    def _pairs(s, V, cand, res, occ, lw=None):
        out = []; c = s.tc
        ce = np.argsort(-np.where(cand, V, -np.inf), 1, kind='stable'); rv = np.argsort(np.where(res, V, np.inf), 1, kind='stable')
        ncand = cand.sum(1); nres = res.sum(1); oc = occ.sum(1)
        for l in range(s.NL):
            free = s.nf - int(oc[l]); k = 0
            while k < ncand[l]:
                e = ce[l, k]; ve = V[l, e]
                if k < free:
                    if ve > c: out.append((float(ve), l, int(e), -1)); k += 1; continue
                    break
                j = k - max(free, 0)
                if j >= nres[l]: break
                v = rv[l, j]; g = ve - V[l, v]
                if g <= c: break
                out.append((float(g), l, int(e), int(v))); k += 1
        if lw is not None: out = [(float(g * lw[l]), l, e, v) for g, l, e, v in out]
        out.sort(key=lambda z: -z[0]); return out

    def _step_cpp(s, counts, ntok, token_ids, new_request, sal):
        """step() with the array work in nqhost.TapCore (same ops, same order, same float32 value bits)"""
        now = s.clock(); K = s.core; T = s.tcore
        c = np.ascontiguousarray(counts, np.float64)
        for n, dt in (('score', np.float64), ('state', np.int8), ('hold', np.float64), ('want', bool), ('doomed', bool)):
            a = getattr(s, n)
            if a.dtype != dt or not a.flags.c_contiguous or not a.flags.writeable: setattr(s, n, np.ascontiguousarray(a, dt).copy())
        K.decay(s.score, c, s.a ** ntok); s.tok += ntok
        if s.s3any: s._s3_add(c, ntok, sal, new_request)
        if s.t_last is not None and ntok <= 16:
            dt = now - s.t_last
            if dt > 0: x = ntok / dt; s.tps = x if s.tps is None else 0.9 * s.tps + 0.1 * x
        s.t_last = now
        ref = False
        if s.P is not None:
            ref = s.P.step(c, ntok, token_ids, new_request, sal=sal) if s.wants_sal else s.P.step(c, ntok, token_ids, new_request)
        st = s.state
        big = T.pre(st, s.doomed, c, -1 if s.slots is None else int(s.slots), float(s.big * s.NE))
        if big: s.stats['big_steps'] += 1
        if ref and not big and s.P is not None and getattr(s.P, 'S', None) is not None:
            lat, H, budget = s._plan(now, T.queued(st), T.todo_len())
            pin = s.pin if s.pin is not None else None
            if pin is not None and (pin.dtype != bool or not pin.flags.c_contiguous): pin = np.ascontiguousarray(pin, bool)
            # the window lengths exactly as _value's v() computes them; numpy scalars (strong) make _value float64 math
            def win(lo, h):
                hi = lo + h; near = max(0.0, min(hi, s.span) - lo); far = max(0.0, hi - max(lo, s.span))
                return float(near), type(near) is not float, float(far), type(far) is not float
            w = win(lat, H) + win(0.0, lat) + (bool(lat >= 1),)
            lw = s._lw() if s.s3on else None
            k, cut, ee, sk = T.refresh(np.ascontiguousarray(s._srcS(), np.float32), s.score, s.fixed, st, s.doomed, s.hold, pin,
                                       float(s.tok), int(s.nf), 1 - s.a, -1.0 if s.tfar is None else float(s.tfar), s.span,
                                       w, s.tc, int(min(budget, 2 ** 62)), lw)
            s.stats['budget_cut'] += cut; s.stats['eager_evict'] += ee; s.stats['no_slot_skip'] += sk; s.stats['promotions'] += k
        if not (s.want.dtype == bool and s.want.flags.c_contiguous and s.want.flags.writeable): s.want = np.zeros(st.shape, bool)
        ups, downs = T.finish(st, s.fixed, s.want)
        if s.t_iss is not None and ups: s._mark_issue(now, ups)
        s.stats['ups'] += len(ups); s.stats['downs'] += len(downs); s.stats['bytes'] += len(ups) * s.rb
        return ups, downs

    def step(s, counts, ntok=1, token_ids=None, new_request=False, sal=None):
        if s.tcore is not None: return s._step_cpp(counts, ntok, token_ids, new_request, sal)
        now = s.clock()
        c = np.asarray(counts, np.float64)
        s.score = s.score * s.a ** ntok + c; s.tok += ntok
        if s.s3any: s._s3_add(c, ntok, sal, new_request)
        if s.t_last is not None and ntok <= 16:
            dt = now - s.t_last
            if dt > 0: x = ntok / dt; s.tps = x if s.tps is None else 0.9 * s.tps + 0.1 * x
        s.t_last = now
        ref = False
        if s.P is not None:
            ref = s.P.step(c, ntok, token_ids, new_request, sal=sal) if s.wants_sal else s.P.step(c, ntok, token_ids, new_request)
        st = s.state; ups = []; downs = []
        for (i, e), (j, v) in list(s.doom.items()):       # lazy evictions: e landed -> release v; e gone -> undoom v
            if st[i, e] == 2:
                del s.doom[i, e]; s.doomed[j, v] = False
                if st[j, v] == 2: downs.append((s.layers[j], v)); st[j, v] = 3
            elif st[i, e] == 0:
                del s.doom[i, e]; s.doomed[j, v] = False
        if s.slots is not None:                          # the pool shrank under the set (nq-prefill reclaims its borrowed slots):
            over = int(((st == 1) | (st == 2)).sum()) - s.slots   # tap has no want-driven downs, so drop the lowest-score
            if over > 0:                                  # residents now, else the over-issued ups wait for a slot forever
                r = (st == 2) & ~s.fixed & ~s.doomed; i, e = np.nonzero(r)
                for k in np.argsort(s.score[i, e], kind='stable')[:over]:
                    downs.append((s.layers[i[k]], int(e[k]))); st[i[k], e[k]] = 3
                s.stats['shrink_evict'] = s.stats.get('shrink_evict', 0) + min(over, len(i))
        nfree = (s.slots - int((st > 0).sum())) if s.slots is not None else 10 ** 9
        while s.todo and nfree > 0:
            i, e = s.todo.popleft()
            if st[i, e] == 0: ups.append((s.layers[i], e)); st[i, e] = 1; nfree -= 1
        big = ((c > 0).sum(1) > s.big * s.NE).any()
        if big: s.stats['big_steps'] += 1
        if ref and not big and s.P is not None and getattr(s.P, 'S', None) is not None:
            n1 = int((st == 1).sum()); nx = s.xq() if s.xq is not None else -1
            s.stats['q_st1'] = n1; s.stats['q_todo'] = len(s.todo); s.stats['q_x'] = nx   # nq-kld: queue-model inputs (last refresh)
            s.stats['q_st1_sum'] = s.stats.get('q_st1_sum', 0) + n1; s.stats['q_x_sum'] = s.stats.get('q_x_sum', 0) + nx
            q0 = (nx if (s.tqreal and nx >= 0) else n1) + len(s.todo)
            lat, H, budget = s._plan(now, q0, len(s.todo))
            V, V0 = s._value(lat, H)
            pin = s.pin & ~s.fixed if s.pin is not None else None
            if pin is not None: V = np.where(pin, np.float32(1e9), V)
            res = (st == 2) & ~s.doomed; occ = ((st == 1) | (st == 2)) & ~s.doomed
            cand = (st == 0) & ~s.doomed & ~s.fixed & (s.hold <= s.tok)
            pairs = s._pairs(V, cand, res & ~s.fixed, occ & ~s.fixed, s._lw() if s.s3on else None)
            k = 0
            for g, l, e, v in pairs:
                if k >= budget: s.stats['budget_cut'] += 1; continue
                if v < 0:
                    if nfree > 0: ups.append((s.layers[l], e)); st[l, e] = 1; nfree -= 1
                    else: s.todo.append((l, e))
                elif nfree > 0:
                    ups.append((s.layers[l], e)); st[l, e] = 1; nfree -= 1; s.doom[l, e] = (l, v); s.doomed[l, v] = True
                elif g - V0[l, v] > s.tc:
                    downs.append((s.layers[l], v)); st[l, v] = 3; s.todo.append((l, e)); s.stats['eager_evict'] += 1
                else:
                    s.stats['no_slot_skip'] += 1; continue
                k += 1
            s.stats['promotions'] += k
        s.want = ((st == 1) | (st == 2)) & ~s.fixed          # compatibility (kv_pressure / session restore read want)
        s.stats['ups'] += len(ups); s.stats['downs'] += len(downs); s.stats['bytes'] += len(ups) * s.rb
        if s.t_iss is not None and ups: s._mark_issue(now, ups)
        return ups, downs

    def _mark_issue(s, now, ups):
        """NQ_TAP_LAT=meas: issue time of each emitted up (landed() turns it into an issue->land latency sample)"""
        li = s.li; s.t_iss[[li[L] for L, _ in ups], [e for _, e in ups]] = now


def make_scheduler(*a, **k):
    """NQ_SCHED unset/'' -> scheduler.Scheduler (the default path, unchanged); 'tap' -> TapScheduler"""
    m = os.environ.get('NQ_SCHED', '')
    if m in ('', 'default'): return Scheduler(*a, **k)
    if m == 'tap': return TapScheduler(*a, **k)
    raise ValueError(f'unknown NQ_SCHED {m!r}')
