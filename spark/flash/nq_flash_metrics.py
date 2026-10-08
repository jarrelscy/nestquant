"""Aggregate actual device-table routing snapshots, without retaining token data."""
import json
import os
import time
from pathlib import Path
import numpy as np


class RoutingMetrics:
    def __init__(self, layers, path):
        self.layers = list(layers)
        self.path = Path(path)
        self.started = time.time()
        self.requests = 0
        self.total = {}
        self.current = {}
        self.last_write = 0.
        self.io = {}

    def new_request(self):
        self.requests += 1
        self.current = {}

    def add(self, hot, active, committed, phase, desired=None, salience=None):
        hot, active = np.asarray(hot, bool), np.asarray(active, bool)
        if hot.shape != active.shape or hot.ndim != 3 or hot.shape[1:] != (len(self.layers), 8):
            raise ValueError('Expected [tokens, layers, 8] actual routing levels')
        desired = hot.copy() if desired is None else np.asarray(desired, bool)
        if desired.shape != hot.shape:
            raise ValueError('Desired mask must match routing shape')
        if salience is not None:
            salience = np.asarray(salience, np.float64)
            if salience.shape != hot.shape or not np.isfinite(salience).all() or (salience < 0).any():
                raise ValueError('Expected finite nonnegative salience matching routing shape')
            salience = np.where(active, salience, 0.)
        if not 0 <= committed <= len(hot):
            raise ValueError('Invalid committed routing prefix')
        for scope, stop in (('executed', len(hot)), ('committed', committed)):
            h = (hot[:stop] & active[:stop]).sum(-1)
            a = active[:stop].sum(-1)
            wanted = (desired[:stop] & active[:stop]).sum(-1)
            late = (desired[:stop] & ~hot[:stop] & active[:stop]).sum(-1)
            unselected_cold = (~desired[:stop] & ~hot[:stop] & active[:stop]).sum(-1)
            for store in (self.total, self.current):
                key = phase + '_' + scope
                if key not in store:
                    store[key] = dict(rows=0, hot=np.zeros(len(self.layers), np.int64),
                                      active=np.zeros(len(self.layers), np.int64),
                                      histogram=np.zeros(9, np.int64),
                                      desired=np.zeros(len(self.layers), np.int64),
                                      late=np.zeros(len(self.layers), np.int64),
                                      unselected_cold=np.zeros(len(self.layers), np.int64))
                c = store[key]
                if salience is not None:
                    if 'salience' not in c:
                        c['salience'] = {k: np.zeros(len(self.layers), np.float64) for k in ('total', 'hot', 'desired', 'late', 'unselected_cold')}
                        c['salience_rows'] = 0
                    c['salience_rows'] += stop
                    masks = dict(total=np.ones_like(hot[:stop]), hot=hot[:stop], desired=desired[:stop],
                                 late=desired[:stop] & ~hot[:stop], unselected_cold=~desired[:stop] & ~hot[:stop])
                    for name, mask in masks.items():
                        c['salience'][name] += (salience[:stop] * mask).sum(axis=(0, 2))
                c['rows'] += stop
                c['hot'] += h.sum(0).astype(np.int64)
                c['active'] += a.sum(0).astype(np.int64)
                c['desired'] += wanted.sum(0).astype(np.int64)
                c['late'] += late.sum(0).astype(np.int64)
                c['unselected_cold'] += unselected_cold.sum(0).astype(np.int64)
                c['histogram'] += np.bincount(h.ravel().astype(np.int64), minlength=9)

    def summarize(self, store):
        result = {}
        for key, c in store.items():
            hot, active = int(c['hot'].sum()), int(c['active'].sum())
            observations = c['rows'] * len(self.layers)
            result[key] = dict(token_rows=c['rows'], expert_activations=active,
                hot_expert_activations=hot, hit_rate=hot/active if active else None,
                mean_hot_of_8=hot/observations if observations else None,
                desired_pool_hit_rate=int(c['desired'].sum())/active if active else None,
                desired_but_cold=int(c['late'].sum()),
                outside_desired_and_cold=int(c['unselected_cold'].sum()),
                desired_but_cold_rate=int(c['late'].sum())/active if active else None,
                outside_desired_and_cold_rate=int(c['unselected_cold'].sum())/active if active else None,
                hot_count_histogram=c['histogram'].tolist(),
                layers={str(L):dict(hot=int(c['hot'][i]), active=int(c['active'][i]),
                    desired=int(c['desired'][i]), desired_but_cold=int(c['late'][i]),
                    outside_desired_and_cold=int(c['unselected_cold'][i]),
                    mean_hot_of_8=float(c['hot'][i]/c['rows']) if c['rows'] else None)
                    for i,L in enumerate(self.layers)})
            if 'salience' in c:
                sums = c['salience']
                def weighted(i=None):
                    values = {k: float(v.sum() if i is None else v[i]) for k,v in sums.items()}
                    den = values['total']
                    return dict(salience_sum=den, hot_salience_sum=values['hot'],
                        desired_salience_sum=values['desired'],
                        hot_salience_coverage=values['hot']/den if den else None,
                        desired_salience_coverage=values['desired']/den if den else None,
                        desired_but_cold_salience_fraction=values['late']/den if den else None,
                        outside_desired_and_cold_salience_fraction=values['unselected_cold']/den if den else None)
                result[key].update(weighted(), salience_token_rows=c['salience_rows'])
                for i,L in enumerate(self.layers):result[key]['layers'][str(L)].update(weighted(i))
        return result

    def export(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 2:
            return
        obj = dict(schema='nq-flash-routing-v3', started_at=self.started,
                   updated_at=time.time(), requests=self.requests, io=self.io,
                   definition='hot=actual level 4 at MoE execution; active=FP16 routing weight nonzero; target layers only; salience=FP32 routing weight squared times normalized MoE input squared norm, active routes only',
                   total=self.summarize(self.total), current_request=self.summarize(self.current))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, self.path)
        self.last_write = now
