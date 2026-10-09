"""Experimental fixed subset inside an unchanged total per-layer slot budget.

EMA, causal inputs, refresh cadence and hysteresis follow shipped jT. Only the
candidate set changes: pinned IDs are always wanted, and the other slots select
from non-pinned experts. This is not the original zero-fixed research policy.
"""
import numpy as np


class FixedSetPolicy:
    def __init__(self, policy_cls, total, default, fixed):
        ids = np.asarray(fixed, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= 288)):
            raise ValueError('Invalid fixed expert IDs')
        if not 0 <= len(ids) < total <= 288:
            raise ValueError('Fixed experts must leave at least one floating slot')
        self.inner = policy_cls(total, default)
        self.fixed = np.zeros(288, dtype=bool)
        self.fixed[ids] = True
        if not np.all(self.inner.cur[self.fixed]):
            raise ValueError('Initial pool must contain all fixed experts')

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def before_row(self, block_prev):
        p = self.inner
        if not self.fixed.any():
            return p.before_row(block_prev)
        load = []
        if p.t and p.t % p.G == 0:
            score = (1 - p.mix) * p.state / max(p.state.sum(), 1e-30) + p.mix * np.asarray(block_prev, np.float64)
            priority = score * np.where(p.cur, 1 + p.hm, 1.0)
            # Preserve reference tie order; exclude fixed IDs before selecting.
            ordered = np.lexsort((~p.cur, -priority))
            candidates = ordered[~self.fixed[ordered]]
            new = self.fixed.copy()
            new[candidates[:p.nf-int(self.fixed.sum())]] = True
            load = np.flatnonzero(new & ~p.cur).tolist()
            p.cur = new
        return p.cur.copy(), load

    def after_row(self, ids, weights, xn):
        self.inner.after_row(ids, weights, xn)
