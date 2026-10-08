"""Bounded delivery scheduling for jT's unchanged desired expert pool.

No extra buffers. Only admitted replacements evict residents. Read cancellations
are best-effort; executor callbacks, never cancellation requests, release slots.
"""
import math
import time
import numpy as np


class ThroughputPrefetch:
    def __init__(self, pool, executor, budgets, priority, *, max_pending=64,
                 min_pending=8, lookahead_seconds=.1, horizon_tokens=16,
                 clock=time.monotonic):
        if not 1 <= min_pending <= max_pending or lookahead_seconds <= 0:
            raise ValueError('Invalid prefetch limits')
        self.pool, self.ex, self.priority = pool, executor, priority
        self.budgets = np.asarray([budgets[L] for L in pool.layers])
        self.maximum, self.minimum = max_pending, min_pending
        self.lookahead, self.horizon = lookahead_seconds, horizon_tokens
        self.clock = clock
        self.rate = None
        self.tps = 25.
        self.limit = min_pending
        self.next_poll = 0.
        self.next_measure = 0.
        self.cancel_sent = set()
        self.issued = self.demotions = self.cancellations = 0
        self.peak_pending = 0
        self.ex.io_stats('prefetch')  # prime the interval, excluding startup bytes

    def observe_decode(self, committed, elapsed):
        if committed > 0 and elapsed > 0:
            self.tps = .9*self.tps + .1*min(committed/elapsed, 1000.)

    def measure(self, now):
        d = self.ex.io_stats('prefetch')
        # Do not interpret idle/under-fed throughput as the SSD's capacity.
        delivered = max(0., float(d.get('delivered_GBps', 0.)))*1e9/self.ex.rb
        if delivered > 0 and (self.rate is None or self.peak_pending >= self.minimum):
            self.rate = delivered if self.rate is None else .75*self.rate+.25*delivered
        if self.rate is not None:
            window = min(self.lookahead, self.horizon/max(self.tps, 1.))
            self.limit = max(self.minimum, min(self.maximum, math.ceil(self.rate*window)))
        self.peak_pending = 0
        self.next_measure = now+.25

    def pump(self, *, force=False, bootstrap=False):
        now = self.clock()
        if not force and now < self.next_poll:
            return
        self.next_poll = now+.01
        p = self.pool
        # A stale request can return to desired before its cancellation lands.
        # We still wait for its callback; if cancelled it is eligible to retry.
        self.cancel_sent = {k for k in self.cancel_sent if p.state[p.li[k[0]], k[1]] == 1}
        for i,e in np.argwhere((p.state == 1) & ~p.wanted):
            key=(p.layers[int(i)],int(e))
            if key not in self.cancel_sent:
                if self.ex.cancel_up(*key, sched=p):
                    self.cancel_sent.add(key)
                    self.cancellations += 1
        if now >= self.next_measure and not bootstrap:
            self.measure(now)
        limit = self.maximum if bootstrap else self.limit
        pending = int(np.count_nonzero((p.state == 1) | (p.state == 3)))
        self.peak_pending = max(self.peak_pending, pending)
        room = limit-pending
        if room <= 0:
            return
        candidates = np.argwhere((p.state == 0) & p.wanted)
        if not len(candidates):
            return
        score=np.asarray(self.priority(), np.float64)
        if score.shape != p.state.shape or not np.isfinite(score).all():
            raise ValueError('Invalid committed jT loading priorities')
        order=np.lexsort((candidates[:,1], candidates[:,0], -score[tuple(candidates.T)]))
        occupied=np.count_nonzero(p.state, axis=1)
        free=len(self.ex.free)
        ups,downs=[],[]
        for i,e in candidates[order]:
            if not room:
                break
            i,e=int(i),int(e)
            if occupied[i] >= self.budgets[i]:
                # Keep useful old residents until an admission is possible.
                # Do not demote a whole layer while its replacements queue.
                victims=np.flatnonzero((p.state[i] == 2) & ~p.wanted[i])
                already=int(np.count_nonzero(p.state[i] == 3))
                if not len(victims) or already:
                    continue
                victim=int(victims[np.argmin(score[i,victims])])
                p.state[i,victim]=3
                downs.append((p.layers[i],victim))
                room-=1
            elif free:
                p.state[i,e]=1
                ups.append((p.layers[i],e))
                occupied[i]+=1;free-=1;room-=1
        if ups or downs:
            # No executor slot-wait queue: every admitted read owns a free slot.
            self.ex.apply(ups,downs,p)
            self.issued+=len(ups);self.demotions+=len(downs)
            self.peak_pending=max(self.peak_pending,
                int(np.count_nonzero((p.state == 1) | (p.state == 3))))

    def stats(self):
        p=self.pool
        return dict(mode='throughput', pending_limit=self.limit,
                    maximum_pending=self.maximum, estimated_records_per_second=self.rate,
                    estimated_committed_tps=self.tps, issued=self.issued,
                    demotions=self.demotions, stale_cancel_requests=self.cancellations,
                    pending=int(np.count_nonzero((p.state == 1) | (p.state == 3))),
                    desired_not_yet_admitted=int(np.count_nonzero((p.state == 0) & p.wanted)))
