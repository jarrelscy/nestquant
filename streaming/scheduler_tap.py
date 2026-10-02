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
Mirror of nq-tfpred/nqalgo ext.py tap-jfe512-c<c>-H<H>-mla<k> (JFLandFC normalised two-horizon value)."""
import os, time, collections
import numpy as np
from scheduler import Scheduler


class TapScheduler(Scheduler):
    def __init__(s, *a, clock=None, **k):
        super().__init__(*a, **k)
        s.core = None                                   # python step only (the C++ host loop mirrors the base policy)
        e = os.environ.get
        s.tc = float(e('NQ_TAP_C', '1.0')); s.tH = float(e('NQ_TAP_H', '256')); s.tHa = float(e('NQ_TAP_HA', '0'))
        s.tmla = float(e('NQ_TAP_MLA', '1.0')); s.tsvc = float(e('NQ_TAP_SVC_MS', '2')) / 1e3
        fr = e('NQ_TAP_FAR', 'tail:0.1'); s.tfar = float(fr.split(':')[1]) if fr.startswith('tail:') else None   # far window: EMA512 | tail:<x> = x * jF rate
        s.tp_n = int(e('NQ_TAP_TP', '4')); s.rb_rank = s.rb / max(1, s.tp_n)
        s.rate0 = float(e('NQ_TAP_RATE_GBPS', '6')) * 1e9 / s.rb_rank      # records/s per rank before measurement
        s.clock = clock or time.monotonic
        s.span = 64.0; s.NL = len(s.layers)
        s.doom = {}; s.doomed = np.zeros((s.NL, s.NE), bool); s.todo = collections.deque()
        s.peak = None; s.t_last = None; s.tps = None; s.n_land = 0; s.t_rate = None; s.n_land0 = 0
        s.stats.update(refreshes=0, promotions=0, budget_cut=0, eager_evict=0, no_slot_skip=0, lat_tok_sum=0.0)
        s.sched_name = 'tap'

    # ---- executor feedback (counts landings for the live rank-0 rate)
    def landed(s, L, e):
        i = s.li[L]
        if s.state[i, e] == 1: s.n_land += 1
        super().landed(L, e)

    # ---- live I/O state
    def _rates(s, now):
        """-> (lat_s per rank list). Rank 0 live; followers from io_all() if available."""
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
        if r0 is not None: s.peak[0] = max(s.peak[0], r0)
        q0 = int((s.state == 1).sum()) + len(s.todo)
        out.append(q0 / max(s.peak[0], 1.0) + s.tsvc)
        for r in sorted(io):
            if r == 0 or r >= nr: continue
            d = io[r]; rbr = s.rb / nr
            g = d.get('delivered_GBps') or 0.0
            s.peak[r] = max(s.peak[r], g * 1e9 / rbr)
            q = (d.get('ops_outstanding') or 0) + (d.get('slot_wait') or 0) + (d.get('backlog') or 0)
            out.append(q / max(s.peak[r], 1.0) + s.tsvc)
        return out

    def _value(s, lat, H):
        S = np.asarray(s.P.S, np.float32); S = np.where(s.fixed, 0, np.maximum(S, 0))
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

    def _pairs(s, V, cand, res, occ):
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
        out.sort(key=lambda z: -z[0]); return out

    def step(s, counts, ntok=1, token_ids=None, new_request=False, sal=None):
        now = s.clock()
        c = np.asarray(counts, np.float64)
        s.score = s.score * s.a ** ntok + c; s.tok += ntok
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
        nfree = (s.slots - int((st > 0).sum())) if s.slots is not None else 10 ** 9
        while s.todo and nfree > 0:
            i, e = s.todo.popleft()
            if st[i, e] == 0: ups.append((s.layers[i], e)); st[i, e] = 1; nfree -= 1
        big = ((c > 0).sum(1) > s.big * s.NE).any()
        if big: s.stats['big_steps'] += 1
        if ref and not big and s.P is not None and getattr(s.P, 'S', None) is not None:
            tps = s.tps or 94.0
            lat = max(s._rates(now)) * tps
            H = float(np.clip(s.tHa * lat, 64, 1024)) if s.tHa > 0 else s.tH
            s.stats['refreshes'] += 1; s.stats['lat_tok_sum'] += lat
            V, V0 = s._value(lat, H)
            pin = s.pin & ~s.fixed if s.pin is not None else None
            if pin is not None: V = np.where(pin, np.float32(1e9), V)
            res = (st == 2) & ~s.doomed; occ = ((st == 1) | (st == 2)) & ~s.doomed
            cand = (st == 0) & ~s.doomed & ~s.fixed & (s.hold <= s.tok)
            pairs = s._pairs(V, cand, res & ~s.fixed, occ & ~s.fixed)
            budget = 10 ** 9
            if s.tmla > 0:
                el = lat; rate = float(s.peak.min()) if s.peak is not None else s.rate0
                budget = max(0, int((s.tmla * H - el) / tps * rate))
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
        return ups, downs


def make_scheduler(*a, **k):
    """NQ_SCHED unset/'' -> scheduler.Scheduler (the default path, unchanged); 'tap' -> TapScheduler"""
    m = os.environ.get('NQ_SCHED', '')
    if m in ('', 'default'): return Scheduler(*a, **k)
    if m == 'tap': return TapScheduler(*a, **k)
    raise ValueError(f'unknown NQ_SCHED {m!r}')
