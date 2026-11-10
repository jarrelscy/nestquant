"""Stage predicted base rows without changing resident slots or model precision."""

import time

import numpy as np

from prefetch import BoundedReader


class NativeBasePrefetch:
    def __init__(self, banks, candidates=2, lookahead=4):
        if candidates not in (1, 2, 4) or not 1 <= lookahead <= 8:
            raise ValueError('Unsupported base prediction window')
        self.banks = {bank.layer: bank for bank in banks}
        if set(self.banks) != set(range(3, 45)) or any(bank.fully_resident for bank in banks):
            raise ValueError('Base read-ahead requires all 42 paged banks')
        if any(3 * (bank.bytes // bank.rows) + 1024**2 > 128 * 1024**2 for bank in banks):
            raise ValueError('One base row exceeds the read-ahead staging budget')
        self.candidates = candidates
        self.lookahead = lookahead
        self.worker = BoundedReader(self._read, 128 * 1024**2, max_pending=8)
        self.order = {}
        self.submitted = 0
        self.submit_refused = 0
        self.drain_seconds = 0.0
        for bank in banks:
            bank.prefetch = self

    def _read(self, layer, expert):
        return self.banks[layer].read_payload(expert)

    def _record(self, layer, value, discarded=False):
        _, size, seconds = value
        stats = self.banks[layer].stats
        stats['prefetch_read_bytes'] += size
        stats['prefetch_read_seconds'] += seconds
        stats['prefetch_discarded'] += int(discarded)

    def _error(self, key, failure):
        if not isinstance(failure, (OSError, EOFError)):
            raise failure
        self.banks[key[0]].stats['prefetch_errors'] += 1

    def _cancel(self, key):
        self.banks[key[0]].stats['prefetch_cancelled'] += 1

    def begin(self, priority):
        self.drain()
        if priority.shape != (42, 288) or not np.isfinite(priority).all():
            raise ValueError('Expected finite committed jT/EMA priorities for every expert')
        for layer, bank in self.banks.items():
            ranked = np.argsort(-priority[layer - 3], kind='stable')
            self.order[layer] = [int(expert) for expert in ranked
                                 if priority[layer - 3, expert] > 0 and int(expert) not in bank.slots][:self.candidates]
        self.advance(2)

    def advance(self, completed_layer):
        self.worker.retire(lambda key: key[0] <= completed_layer,
                           lambda key, value: self._record(key[0], value, discarded=True), self._error, self._cancel)
        for layer in range(completed_layer + 1, min(45, completed_layer + self.lookahead + 1)):
            for expert in self.order.get(layer, ()):
                key = layer, expert
                if key in self.worker.pending or expert in self.banks[layer].slots:
                    continue
                if not self.worker.submit(key, self.banks[layer].bytes // self.banks[layer].rows):
                    self.submit_refused += 1
                    return
                self.submitted += 1

    def consume(self, layer, expert, slot):
        stats = self.banks[layer].stats
        started = time.perf_counter()
        copy_before = stats['prefetch_copy_seconds']

        def install(value):
            self._record(layer, value)
            copy_start = time.perf_counter()
            self.banks[layer]._install_payload(slot, value[0])
            stats['prefetch_copy_seconds'] += time.perf_counter() - copy_start

        found = self.worker.consume((layer, expert), install, self._error, self._cancel)
        stats['prefetch_wait_seconds'] += time.perf_counter() - started - (stats['prefetch_copy_seconds'] - copy_before)
        stats['prefetch_hits'] += int(found)
        return found

    def drain(self):
        started = time.perf_counter()
        try:
            for key in list(self.worker.pending):
                self.worker.consume(key, lambda value, layer=key[0]: self._record(layer, value, discarded=True),
                                    self._error, self._cancel)
        finally:
            self.order.clear()
            self.drain_seconds += time.perf_counter() - started

    def report(self):
        return dict(candidates_per_layer=self.candidates, lookahead_layers=self.lookahead,
                    staging_limit_bytes=self.worker.budget_bytes, staging_peak_reserved_bytes=self.worker.peak_reserved_bytes,
                    max_pending=self.worker.max_pending, submit_refused=self.submit_refused,
                    drain_seconds=self.drain_seconds, cross_token_lookahead=False,
                    submitted=self.submitted, policy='Committed jT/EMA priority; publish only required base rows')

    def close(self):
        try:
            self.drain()
        finally:
            try:
                self.worker.close()
            finally:
                for bank in self.banks.values():
                    bank.prefetch = None
