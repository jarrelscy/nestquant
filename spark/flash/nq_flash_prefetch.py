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
        self.last_io = {}
        self.ex.io_stats('prefetch')  # prime the interval, excluding startup bytes

    def observe_decode(self, committed, elapsed):
        if committed > 0 and elapsed > 0:
            self.tps = .9*self.tps + .1*min(committed/elapsed, 1000.)

    def measure(self, now):
        d = self.ex.io_stats('prefetch')
        self.last_io = d
        delivered = max(0., float(d.get('delivered_GBps', 0.)))*1e9/self.ex.rb
        # Delivered bandwidth is a lower bound when admission/acknowledgments
        # starve the drive. Keep the observed peak rather than feeding our own
        # under-admission back into a shrinking capacity estimate.
        if delivered > 0:
            self.rate = max(self.rate or 0., delivered)
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
        # An I/O completion awaiting a layer mailbox is not an outstanding
        # read. Demotions contain no SSD read at all. Bound these separately.
        reads = sum(op[2] == 4 for op in self.ex.ops.values())
        room = min(limit-reads, self.maximum-pending)
        down_room = min(self.minimum-int(np.count_nonzero(p.state == 3)),
                        self.maximum-pending)
        self.peak_pending = max(self.peak_pending, reads)
        if room <= 0 and down_room <= 0:
            return
        occupied = np.count_nonzero(p.state, axis=1)
        free = len(self.ex.free)
        demoting = np.any(p.state == 3, axis=1)
        victim_mask = (p.state == 2) & ~p.wanted
        has_victim = np.any(victim_mask, axis=1)
        can_load = (occupied < self.budgets) & (free > 0) & (room > 0)
        can_demote = ((occupied >= self.budgets) & ~demoting & has_victim
                      & (down_room > 0))
        # Filter once per layer. A pending mailbox can block hundreds of
        # candidates; rescanning that layer for every expert starves launches.
        candidates = np.argwhere((p.state == 0) & p.wanted
                                 & (can_load | can_demote)[:, None])
        if not len(candidates):
            return
        score=np.asarray(self.priority(), np.float64)
        if score.shape != p.state.shape or not np.isfinite(score).all():
            raise ValueError('Invalid committed jT loading priorities')
        order=np.lexsort((candidates[:,1], candidates[:,0], -score[tuple(candidates.T)]))
        victims = np.argmin(np.where(victim_mask, score, np.inf), axis=1)
        ups,downs=[],[]
        for i,e in candidates[order]:
            if room <= 0 and down_room <= 0:
                break
            i,e=int(i),int(e)
            if occupied[i] >= self.budgets[i]:
                # Keep useful old residents until an admission is possible.
                # Do not demote a whole layer while its replacements queue.
                if not has_victim[i] or demoting[i] or down_room <= 0 or pending >= self.maximum:
                    continue
                victim=int(victims[i])
                p.state[i,victim]=3
                demoting[i]=True
                downs.append((p.layers[i],victim))
                down_room-=1;pending+=1
                room=min(room,self.maximum-pending)
            elif free and room > 0 and pending < self.maximum:
                p.state[i,e]=1
                ups.append((p.layers[i],e))
                occupied[i]+=1;free-=1;room-=1;pending+=1
                down_room=min(down_room,self.maximum-pending)
        if ups or downs:
            # No executor slot-wait queue: every admitted read owns a free slot.
            self.ex.apply(ups,downs,p)
            self.issued+=len(ups);self.demotions+=len(downs)
            self.peak_pending=max(self.peak_pending,
                int(np.count_nonzero((p.state == 1) | (p.state == 3))))

    def stats(self):
        p=self.pool
        return dict(mode='throughput', pending_limit=self.limit,
                    maximum_pending=self.maximum, outstanding_reads=sum(op[2] == 4 for op in self.ex.ops.values()),
                    mailbox_pending=len(self.ex.wait_apply),
                    pending_demotions=int(np.count_nonzero(p.state == 3)),
                    estimated_records_per_second=self.rate,
                    estimated_committed_tps=self.tps, issued=self.issued,
                    demotions=self.demotions, stale_cancel_requests=self.cancellations,
                    pending=int(np.count_nonzero((p.state == 1) | (p.state == 3))),
                    desired_not_yet_admitted=int(np.count_nonzero((p.state == 0) & p.wanted)))
