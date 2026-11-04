from collections import deque

import numpy as np


def top_mask(scores, budgets):
    mask = np.zeros(scores.shape, dtype=bool)
    for layer, budget in enumerate(budgets):
        mask[layer, np.argsort(-scores[layer], kind='stable')[:budget]] = True
    return mask


class PredictorMetrics:
    def __init__(self, budgets, defaults, fixed_ids):
        self.budgets = np.asarray(budgets)
        self.static = np.zeros((42, 288), dtype=bool)
        self.fixed = self.static.copy()
        for layer, (default, fixed) in enumerate(zip(defaults, fixed_ids)):
            self.static[layer, default] = True
            self.fixed[layer, fixed] = True
        self.steps = 0
        self.counts = {name: np.zeros(42, dtype=np.int64) for name in
                       ('wanted', 'ready', 'wanted_ready', 'late', 'cached_outside_wanted', 'fixed')}
        self.coverage = {name: np.zeros(42) for name in
                         ('wanted', 'ready', 'fixed', 'jt_top_budget', 'ema_top_budget', 'static')}
        self.predictions = deque(maxlen=16)
        self.shares = deque(maxlen=16)
        self.windows = 0
        self.tv = 0.0
        self.cross_entropy = 0.0
        self.predicted_mass = 0.0
        self.observed_mass = 0.0
        self.bin_edges = np.asarray([0, .001, .005, .01, .025, .05, .1, .25, .5, 1.000001])
        self.bin_count = np.zeros(9, dtype=np.int64)
        self.bin_predicted = np.zeros(9)
        self.bin_observed = np.zeros(9)

    def observe(self, wanted, ready, selected, weights, block, ema):
        if block is None or not np.isfinite(block).all() or np.any(block < 0) or not np.allclose(block.sum(-1), 1, atol=1e-5):
            raise ValueError('Expected normalized causal jT block predictions')
        shares = np.square(weights.astype(np.float64))
        shares /= np.maximum(shares.sum(-1, keepdims=True), 1e-30)
        masks = dict(wanted=wanted, ready=ready, wanted_ready=wanted & ready,
                     late=wanted & ~ready, cached_outside_wanted=ready & ~wanted, fixed=self.fixed,
                     jt_top_budget=top_mask(block, self.budgets), ema_top_budget=top_mask(ema, self.budgets),
                     static=self.static)
        for name, mask in masks.items():
            hits = np.take_along_axis(mask, selected, axis=1)
            if name in self.counts:
                self.counts[name] += hits.sum(-1)
            if name in self.coverage:
                self.coverage[name] += (hits * shares).sum(-1)
        dense = np.zeros((42, 288))
        np.put_along_axis(dense, selected, shares, axis=1)
        self.predictions.append(block.astype(np.float64).copy())
        self.shares.append(dense)
        self.steps += 1
        if len(self.shares) == 16:
            predicted = self.predictions[0]
            observed = np.mean(self.shares, axis=0)
            self.windows += 1
            self.tv += float(np.abs(predicted - observed).sum(-1).mean() / 2)
            self.cross_entropy += float(-(observed * np.log(np.maximum(predicted, 1e-30))).sum(-1).mean())
            mask = top_mask(predicted, self.budgets)
            self.predicted_mass += float((predicted * mask).sum(-1).mean())
            self.observed_mass += float((observed * mask).sum(-1).mean())
            bins = np.clip(np.searchsorted(self.bin_edges, predicted, side='right') - 1, 0, 8).ravel()
            self.bin_count += np.bincount(bins, minlength=9)
            self.bin_predicted += np.bincount(bins, weights=predicted.ravel(), minlength=9)
            self.bin_observed += np.bincount(bins, weights=observed.ravel(), minlength=9)

    def report(self):
        routed = self.steps * 42 * 8
        counts = {name: int(value.sum()) for name, value in self.counts.items()}
        calibration = dict(full_windows=self.windows, excluded_tail_predictions=min(self.steps, 15),
                           target='Mean per-token normalized squared routing weights over t..t+15, matching shipped train.py block_loss',
                           scope='Decode-only overlapping windows on observed mixed-precision routing; not independent samples or output-quality validation')
        if self.windows:
            populated = self.bin_count > 0
            calibration.update(
                mean_total_variation=self.tv / self.windows,
                mean_cross_entropy_nats=self.cross_entropy / self.windows,
                uniform_cross_entropy_nats=float(np.log(288)),
                top_budget_predicted_salience_share=self.predicted_mass / self.windows,
                top_budget_observed_salience_share=self.observed_mass / self.windows,
                soft_share_bin_error=float(np.abs(self.bin_predicted - self.bin_observed).sum() / self.bin_count.sum()),
                bins=[dict(lower=float(self.bin_edges[i]), upper=min(1.0, float(self.bin_edges[i + 1])),
                           count=int(self.bin_count[i]), predicted_mean=float(self.bin_predicted[i] / self.bin_count[i]),
                           observed_mean=float(self.bin_observed[i] / self.bin_count[i])) for i in np.flatnonzero(populated)],
            )
        return dict(
            decode_steps=self.steps, routed_calls=routed, allocated_slots=int(self.budgets.sum()),
            allocated_fraction=float(self.budgets.sum() / 12096), counts=counts,
            call_fractions={name: value / routed if routed else None for name, value in counts.items()},
            wanted_delivery_fraction=counts['wanted_ready'] / counts['wanted'] if counts['wanted'] else None,
            mean_salience_coverage={name: float(value.sum() / (42 * self.steps)) if self.steps else None
                                    for name, value in self.coverage.items()},
            per_layer=[dict(layer=i + 3, slots=int(self.budgets[i]), routed_calls=self.steps * 8,
                            counts={name: int(value[i]) for name, value in self.counts.items()},
                            mean_salience_coverage={name: float(value[i] / self.steps) if self.steps else None
                                                    for name, value in self.coverage.items()}) for i in range(42)],
            block_calibration=calibration,
            limitations=['Top-budget jT, EMA and static comparisons rescore the same observed routing, not alternative model executions',
                         'Salience coverage is not answer accuracy', 'No predictor parameters fitted on this request'],
        )
