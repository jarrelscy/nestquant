"""Use the existing committed jT predictor with persistent native refinement slots."""

import hashlib
import json
import os
import time

import mlx.core as mx
import numpy as np
import torch

from experts import ExpertStore, RefinementReader
from packed_expert_bank import PackedRefinementBank
from predictor import configure_slots, predictor_memory, staged_predictor
from predictor_metrics import PredictorMetrics
from prefetch import BoundedReader


COLD_SLOT = 0xffffffff
REFINEMENT_LIMIT = 4 * 1024**3


class NativeHotRefinements:
    def __init__(self, root, fraction, prefill_rows):
        os.environ.setdefault('NQ_FLASH_JT_PARAMS', json.dumps(dict(mix=0.75, hm=4.0, G=1, half_life=64.0)))
        module, self.staging = staged_predictor()
        self.predictor = module.CommittedPredictor(
            root, preset='spark_256K_U_2630_flat50', device='cpu', dtype=torch.float32, fixed_fraction=0.2)
        configure_slots(self.predictor, module, root, fraction)
        self.reader = RefinementReader(root)
        self.worker = BoundedReader(self.reader.read, 64 * 1024**2)
        self.prefill_rows = prefill_rows
        self.prefill_max_bytes = 0
        self.banks = {}
        self.position = 0
        self.reset()

    def attach(self, layer, base):
        rows = self.predictor.budgets[layer]
        proposed = PackedRefinementBank.bytes_for(base, rows)
        scratch = max(self.prefill_max_bytes, PackedRefinementBank.bytes_for(base, self.prefill_rows))
        if proposed + self.reserved_bytes() + scratch + self.worker.budget_bytes > REFINEMENT_LIMIT:
            raise MemoryError('Native decode, prefill, and staging banks exceed the 4 GiB refinement budget')
        self.banks[layer] = PackedRefinementBank(base, rows)
        self.prefill_max_bytes = scratch

    def reserved_bytes(self):
        return sum(bank.bytes for bank in self.banks.values())

    def reset(self):
        self.worker.discard()
        self.predictor.reset()
        self.position = 0
        self.wanted = np.zeros((42, 288), dtype=bool)
        self.ready = np.zeros((42, 288), dtype=bool)
        self.order = []
        self.phase = 'prefill'
        self.prefill_layer = None
        self.prefill_bank = None
        self.prefill_peak_bytes = 0
        self.chunk_index = 0
        self.traces = {}
        self.seconds = 0.0
        self.reads = dict(prefill_bytes=0, decode_bytes=0, installed=0, discarded=0, decode_worker_seconds=0.0)
        self.metrics = PredictorMetrics([self.predictor.budgets[layer] for layer in range(3, 45)],
                                        self.predictor.defaults, self.predictor.fixed_ids)

    def _queue(self):
        for key in self.order:
            layer, expert = key
            if expert in self.banks[layer].slots or key in self.worker.pending:
                continue
            size = ExpertStore.refinement_bytes(self.reader, layer, expert)
            if not self.worker.submit(key, size):
                break

    def begin_decode(self, token):
        self.phase = 'decode'
        self.prefill_bank = None
        self.chunk_index = 0
        self.ready.fill(False)
        started = time.perf_counter()
        self.wanted = self.predictor.preforward(token, self.position)
        self.seconds += time.perf_counter() - started
        priority = self.predictor.loading_priority()
        self.order = sorted(((int(layer) + 3, int(expert)) for layer, expert in np.argwhere(self.wanted)),
                            key=lambda key: (-float(priority[key[0] - 3, key[1]]), key))
        self._queue()

    def route(self, layer, base, routed_ids, weights, norms):
        unique = tuple(dict.fromkeys(routed_ids))
        trace = self.traces.setdefault(self.chunk_index, {})
        trace[layer] = (routed_ids, weights, norms)
        if self.phase == 'prefill':
            if self.prefill_layer != layer:
                if self.prefill_bank is not None:
                    self.prefill_bank.finish()
                    self.prefill_bank = None
                self.prefill_bank = PackedRefinementBank(base, self.prefill_rows)
                self.prefill_layer = layer
                self.prefill_peak_bytes = max(self.prefill_peak_bytes, self.prefill_bank.bytes)
            bank = self.prefill_bank
            required = set(unique)
            for expert in unique:
                if expert in bank.slots:
                    continue
                payload, count, _ = self.reader.read(layer, expert)
                self.reads['prefill_bytes'] += count
                if not bank.install(expert, payload, required):
                    raise RuntimeError('Required prefill refinement could not be installed')
        else:
            def publish(key, value):
                target, expert = key
                payload, count, elapsed = value
                self.reads['decode_bytes'] += count
                self.reads['decode_worker_seconds'] += elapsed
                wanted = set(np.flatnonzero(self.wanted[target - 3]))
                if expert in wanted and self.banks[target].install(expert, payload, wanted):
                    self.reads['installed'] += 1
                else:
                    self.reads['discarded'] += 1

            self.worker.poll(publish)
            self._queue()
            bank = self.banks[layer]
            for expert in unique:
                self.ready[layer - 3, expert] = expert in bank.slots
        slots = mx.array([bank.slots.get(expert, COLD_SLOT) for expert in routed_ids], dtype=mx.uint32)
        return bank, slots

    def commit(self, tokens, chunk_index=0):
        trace = self.traces.pop(chunk_index)
        if set(trace) != set(range(3, 45)):
            raise RuntimeError('A native token did not visit every routed layer')
        ordered = [trace[layer] for layer in range(3, 45)]
        selected = np.array([ids for ids, _, _ in ordered]).reshape(42, len(tokens), 8).transpose(1, 0, 2)
        weights = np.asarray(mx.stack([weights.reshape(len(tokens), 8).astype(mx.float32)
                                      for _, weights, _ in ordered])).transpose(1, 0, 2)
        norms = np.asarray(mx.stack([norms.reshape(len(tokens)) for _, _, norms in ordered])).T
        if self.phase == 'decode':
            self.metrics.observe(self.wanted, self.ready, selected[0], weights[0], self.predictor.block,
                                 np.stack([policy.state for policy in self.predictor.policies]))
        started = time.perf_counter()
        self.predictor.commit([(self.position + i, token, selected[i], weights[i], norms[i])
                               for i, token in enumerate(tokens)])
        self.seconds += time.perf_counter() - started
        self.position += len(tokens)

    def report(self):
        completed = dict(reads=0, bytes=0, seconds=0.0)

        def count(value):
            _, size, seconds = value
            completed['reads'] += 1
            completed['bytes'] += size
            completed['seconds'] += seconds

        self.worker.discard(count)
        return dict(bytes_reserved=self.reserved_bytes(), prefill_rows=self.prefill_rows,
                    prefill_max_bytes=self.prefill_max_bytes, budget_bytes=REFINEMENT_LIMIT,
                    prefill_staging_peak_bytes=self.prefill_peak_bytes,
                    staging_peak_bytes=self.worker.peak_reserved_bytes,
                    predictor_seconds=self.seconds, predictor_memory=predictor_memory(self.predictor),
                    **self.reads, unpublished_at_end=completed, coverage=self.metrics.report())

    def readiness_digest(self):
        return hashlib.sha256(self.ready.tobytes()).hexdigest()

    def prefill_state_digest(self):
        if self.phase != 'prefill' or self.traces:
            raise RuntimeError('Expected a committed native prefill')
        predictor = self.predictor
        digest = hashlib.sha256()
        clocks = (self.position, predictor.t, predictor.prepared_t, predictor.net.position,
                  *(policy.t for policy in predictor.policies))
        digest.update(np.asarray(clocks, dtype=np.int64).tobytes())
        arrays = [predictor.previous_ids, predictor.previous_q, predictor.block,
                  *(policy.state for policy in predictor.policies), *(policy.cur for policy in predictor.policies),
                  *(value.numpy() for pair in predictor.net.cache if pair is not None for value in pair)]
        for value in arrays:
            digest.update(value.tobytes())
        return digest.hexdigest()

    def close(self):
        try:
            self.worker.close()
        finally:
            self.reader.close()
            self.staging.cleanup()
