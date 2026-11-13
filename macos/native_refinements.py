"""Use the existing committed jT predictor with persistent native refinement slots."""

import hashlib
import json
import os
import threading
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


class LockedRefinementReader(RefinementReader):
    """Serialize the lazily filled handle and manifest caches so several reader threads can share them."""

    def __init__(self, root):
        super().__init__(root)
        self.lock = threading.Lock()

    def _manifest(self, layer):
        with self.lock:
            return RefinementReader._manifest(self, layer)

    def _handle(self, layer, shard):
        with self.lock:
            return RefinementReader._handle(self, layer, shard)


class NativeHotRefinements:
    def __init__(self, root, fraction, prefill_rows, budget_gib=4, readers=1, pending=2, staging_mib=64,
                 predictor_backend=None):
        # 'torch' (default) runs jT on the CPU with PyTorch; 'coreml' (or 'coreml:cpu', 'coreml:all') runs it
        # through Core ML, by default on the CPU and Neural Engine. NQ_PREDICTOR_BACKEND sets the default.
        predictor_backend = predictor_backend or os.environ.get('NQ_PREDICTOR_BACKEND', 'torch')
        backend, _, units = predictor_backend.partition(':')
        if backend not in ('torch', 'coreml') or (units and backend != 'coreml'):
            raise ValueError(f'Unknown predictor backend {predictor_backend!r}')
        if type(budget_gib) is not int or not 4 <= budget_gib <= 32:
            raise ValueError('Use a refinement budget between 4 and 32 GiB')
        if type(readers) is not int or not 1 <= readers <= 8 or type(pending) is not int or not readers <= pending <= 64:
            raise ValueError('Use one to eight readers and at least one pending read per reader, at most 64')
        self.budget_bytes = budget_gib * 1024**3
        os.environ.setdefault('NQ_FLASH_JT_PARAMS', json.dumps(dict(mix=0.75, hm=4.0, G=1, half_life=64.0)))
        module, self.staging = staged_predictor()
        self.predictor = module.CommittedPredictor(
            root, preset='spark_256K_U_2630_flat50', device='cpu', dtype=torch.float32, fixed_fraction=0.2)
        configure_slots(self.predictor, module, root, fraction)
        self.predictor_backend = predictor_backend
        if backend == 'coreml':
            from coreml_predictor import install
            install(self.predictor, root / 'serving/predictor/jt', compute_units=units or 'ane')
        record = os.environ.get('NQ_PREDICTOR_RECORD')
        self.record = open(record, 'ab') if record else None
        self.reader = LockedRefinementReader(root) if readers > 1 else RefinementReader(root)
        self.worker = BoundedReader(self.reader.read, staging_mib * 1024**2, pending, readers)
        self.prefill_rows = prefill_rows
        self.draft_rows_base = False
        self.prefill_max_bytes = 0
        self.banks = {}
        self.position = 0
        self.reset()

    def attach(self, layer, base):
        rows = self.predictor.budgets[layer]
        proposed = PackedRefinementBank.bytes_for(base, rows)
        scratch = max(self.prefill_max_bytes, PackedRefinementBank.bytes_for(base, self.prefill_rows))
        if proposed + self.reserved_bytes() + scratch + self.worker.budget_bytes > self.budget_bytes:
            raise MemoryError('Native decode, prefill, and staging banks exceed the refinement budget')
        self.banks[layer] = PackedRefinementBank(base, rows)
        self.prefill_max_bytes = scratch

    def warm(self, readers=4, check=None):
        """Fill each layer's refinement bank with the predictor's initial pool before the first request.

        Spark starts from the same pools. Without this, every slot starts cold and a 2,630-slot
        allocation needs about 20 GiB of reads during the first decode steps.
        """
        from concurrent.futures import ThreadPoolExecutor

        keys = [(layer, int(expert)) for layer, pool in zip(self.predictor.layers, self.predictor.defaults)
                for expert in pool if int(expert) not in self.banks[layer].slots]
        pools = {layer: {int(expert) for expert in pool}
                 for layer, pool in zip(self.predictor.layers, self.predictor.defaults)}
        reader = self.reader if readers == 1 else LockedRefinementReader(self.reader.root)
        started = time.perf_counter()
        installed = read_bytes = 0
        try:
            with ThreadPoolExecutor(max_workers=readers, thread_name_prefix='nestquant-warm') as executor:
                window = 2 * readers
                futures = {}
                position = 0
                while position < len(keys) or futures:
                    while position < len(keys) and len(futures) < window:
                        futures[keys[position]] = executor.submit(reader.read, *keys[position])
                        position += 1
                    key = next(iter(futures))
                    payload, count, _ = futures.pop(key).result()
                    if not self.banks[key[0]].install(key[1], payload, pools[key[0]]):
                        raise RuntimeError(f'Initial refinement pool does not fit layer {key[0]}')
                    installed += 1
                    read_bytes += count
                    del payload
                    if check is not None:
                        check()
        finally:
            if reader is not self.reader:
                reader.close()
        return dict(installed=installed, read_bytes=read_bytes, seconds=time.perf_counter() - started)

    def reserved_bytes(self):
        return sum(bank.bytes for bank in self.banks.values())

    def _record(self, *event):
        if getattr(self, 'record', None) is not None:
            import pickle
            pickle.dump(event, self.record, protocol=5)

    def reset(self):
        self.worker.discard()
        self.predictor.reset()
        self._record('reset')
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
        self.cpu_seconds = 0.0
        self.reads = dict(prefill_bytes=0, decode_bytes=0, installed=0, discarded=0, decode_worker_seconds=0.0,
                          prefill_reader_seconds=0.0, prefill_install_seconds=0.0, decode_publication_seconds=0.0)
        self.metrics = PredictorMetrics([self.predictor.budgets[layer] for layer in range(3, 45)],
                                        self.predictor.defaults, self.predictor.fixed_ids)

    def _read_many(self, layer, experts):
        # Parallel reads need the locked reader, whose handle and manifest caches are thread-safe.
        if len(experts) < 2 or not isinstance(self.reader, LockedRefinementReader):
            return (self.reader.read(layer, expert) for expert in experts)
        if getattr(self, '_prefill_pool', None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._prefill_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='nestquant-prefill')
        return self._prefill_pool.map(lambda expert: self.reader.read(layer, expert), experts)

    def _size(self, layer, expert):
        sizes = self.__dict__.setdefault('_sizes', {})
        key = layer, expert
        if key not in sizes:
            sizes[key] = ExpertStore.refinement_bytes(self.reader, layer, expert)
        return sizes[key]

    def _queue(self):
        for key in self.order:
            layer, expert = key
            if expert in self.banks[layer].slots or key in self.worker.pending:
                continue
            if not self.worker.submit(key, self._size(layer, expert)):
                break

    def _publish(self, key, value):
        target, expert = key
        payload, count, elapsed = value
        self.reads['decode_bytes'] += count
        self.reads['decode_worker_seconds'] += elapsed
        started = time.perf_counter()
        installed = bool(self.wanted[target - 3, expert]) and self.banks[target].install(
            expert, payload, set(np.flatnonzero(self.wanted[target - 3])))
        self.reads['decode_publication_seconds'] += time.perf_counter() - started
        if installed:
            self.reads['installed'] += 1
        else:
            self.reads['discarded'] += 1

    def begin_decode(self, token):
        self.phase = 'decode'
        self.prefill_bank = None
        self.chunk_index = 0
        self.ready.fill(False)
        self._record('preforward', int(token), self.position)
        started = time.perf_counter()
        cpu = time.process_time()
        self.wanted = self.predictor.preforward(token, self.position)
        self.seconds += time.perf_counter() - started
        self.cpu_seconds = getattr(self, "cpu_seconds", 0.0) + time.process_time() - cpu
        # Publish completed reads at the token boundary. The previous token has been evaluated, so slot
        # buffers and the device slot tables can change without waiting on queued GPU work.
        self.worker.poll(self._publish)
        self._plan_reads()

    def _plan_reads(self):
        # Queue only wanted experts that are not already resident, highest loading priority first.
        missing = self.wanted.copy()
        for layer, bank in self.banks.items():
            resident = bank.table != COLD_SLOT
            missing[layer - 3, resident] = False
        layers, experts = np.nonzero(missing)
        priority = self.predictor.loading_priority()[layers, experts]
        order = np.lexsort((experts, layers, -priority))
        self.order = list(zip((layers[order] + 3).tolist(), experts[order].tolist()))
        self._queue()

    def publish(self):
        """Install completed reads. Call only while no queued GPU work uses the refinement banks."""
        self.worker.poll(self._publish)

    def plan(self, token):
        """Advance the predictor to the next target token and queue reads, without touching slot buffers.

        This only changes host state, so it can run while the GPU evaluates the current token.
        """
        self._record('preforward', int(token), self.position)
        started = time.perf_counter()
        cpu = time.process_time()
        self.wanted = self.predictor.preforward(token, self.position)
        self.seconds += time.perf_counter() - started
        self.cpu_seconds = getattr(self, "cpu_seconds", 0.0) + time.process_time() - cpu
        self._plan_reads()

    def start_pipelined_decode(self):
        self.phase = 'decode'
        self.prefill_bank = None
        self.chunk_index = 0
        self.ready.fill(False)

    def take_trace(self, chunk_index=0):
        return self.traces.pop(chunk_index)

    def route_device(self, layer, routed_ids, weights, norms):
        """Route decode refinements on the device without reading expert IDs back to the host.

        Missing refinements fall back to the 1.5-bit base inside the kernel. Readiness and predictor
        inputs are read once per token in commit().
        """
        if self.phase != 'decode':
            raise RuntimeError('Device refinement routing is limited to decode')
        bank = self.banks[layer]
        slots = bank.device_table()[routed_ids]
        if getattr(self, "draft_rows_base", False) and routed_ids.size > 8:
            # Speculative verify: only the first token keeps its 4-bit refinements; draft rows decode the
            # 1.5-bit base, which roughly halves the extra expert work those rows add.
            first = mx.arange(routed_ids.size, dtype=mx.uint32) < 8
            slots = mx.where(first, slots, mx.array(COLD_SLOT, dtype=mx.uint32))
        self.traces.setdefault(self.chunk_index, {})[layer] = (routed_ids, weights.astype(mx.float32), norms, slots)
        return bank, slots

    def route(self, layer, base, routed_ids, weights, norms):
        unique = tuple(dict.fromkeys(routed_ids))
        trace = self.traces.setdefault(self.chunk_index, {})
        trace[layer] = (routed_ids, weights if weights is None else weights.astype(mx.float32), norms)
        if self.phase == 'prefill':
            if self.prefill_layer != layer:
                if self.prefill_bank is not None:
                    self.prefill_bank.finish()
                    self.prefill_bank = None
                self.prefill_bank = PackedRefinementBank(base, self.prefill_rows)
                self.prefill_layer = layer
                self.prefill_peak_bytes = max(self.prefill_peak_bytes, self.prefill_bank.bytes)
            bank = self.prefill_bank
            decode = self.banks.get(layer)
            if decode is not None and all(expert in decode.slots for expert in unique):
                # Every routed expert already has its refinement in the decode bank.
                bank = decode
                self.reads['prefill_decode_bank_calls'] = self.reads.get('prefill_decode_bank_calls', 0) + 1
            else:
                required = set(unique)
                missing = []
                for expert in unique:
                    if expert in bank.slots:
                        continue
                    if decode is not None and expert in decode.slots:
                        started = time.perf_counter()
                        if not bank.copy_from(decode, expert, required):
                            raise RuntimeError('Required prefill refinement could not be installed')
                        self.reads['prefill_install_seconds'] += time.perf_counter() - started
                        self.reads['prefill_copied'] = self.reads.get('prefill_copied', 0) + 1
                    else:
                        missing.append(expert)
                for expert, (payload, count, seconds) in zip(missing, self._read_many(layer, missing)):
                    self.reads['prefill_bytes'] += count
                    self.reads['prefill_reader_seconds'] += seconds
                    started = time.perf_counter()
                    installed = bank.install(expert, payload, required)
                    self.reads['prefill_install_seconds'] += time.perf_counter() - started
                    if not installed:
                        raise RuntimeError('Required prefill refinement could not be installed')
        else:
            self.worker.poll(self._publish)
            self._queue()
            bank = self.banks[layer]
            for expert in unique:
                self.ready[layer - 3, expert] = expert in bank.slots
        slots = mx.array([bank.slots.get(expert, COLD_SLOT) for expert in routed_ids], dtype=mx.uint32)
        return bank, slots

    def commit(self, tokens, chunk_index=0, trace=None, rows=None):
        """Commit routing for `tokens`. `rows` limits the trace to its leading token rows, for
        speculative verification where only accepted rows may reach the predictor."""
        if trace is None:
            trace = self.traces.pop(chunk_index)
        if rows is not None:
            if len(tokens) != rows:
                raise ValueError('Commit one token per accepted row')
            trace = {layer: (np.asarray(entry[0]).reshape(-1)[:rows * 8], np.asarray(entry[1]).reshape(-1, 8)[:rows],
                             np.asarray(entry[2]).reshape(-1)[:rows],
                             *[np.asarray(value).reshape(-1)[:rows * 8] for value in entry[3:]])
                     for layer, entry in trace.items()}
        if set(trace) != set(range(3, 45)):
            raise RuntimeError('A native token did not visit every routed layer')
        ordered = [trace[layer] for layer in range(3, 45)]
        # np.asarray copies arrays that the forward already evaluated. Building new MLX ops here would
        # queue them behind the next token's forward on the GPU stream and serialize the pipeline.
        if len(ordered[0]) == 4:
            selected = np.stack([np.asarray(entry[0]).reshape(-1) for entry in ordered]).astype(np.int64)
            ready = np.stack([np.asarray(entry[3]).reshape(-1) for entry in ordered]) != COLD_SLOT
            self.ready.fill(False)
            for row, (ids_row, ready_row) in enumerate(zip(selected, ready)):
                self.ready[row, ids_row[ready_row]] = True
            selected = selected.reshape(42, len(tokens), 8).transpose(1, 0, 2)
        else:
            selected = np.array([entry[0] for entry in ordered]).reshape(42, len(tokens), 8).transpose(1, 0, 2)
        weights = np.stack([np.asarray(entry[1]).astype(np.float32).reshape(len(tokens), 8)
                            for entry in ordered]).transpose(1, 0, 2)
        norms = np.stack([np.asarray(entry[2]).astype(np.float32).reshape(len(tokens)) for entry in ordered]).T
        if self.phase == 'decode':
            self.metrics.observe(self.wanted, self.ready, selected[0], weights[0], self.predictor.block,
                                 np.stack([policy.state for policy in self.predictor.policies]))
        rows = [(self.position + i, int(token), selected[i], weights[i], norms[i]) for i, token in enumerate(tokens)]
        self._record('commit', rows)
        started = time.perf_counter()
        cpu = time.process_time()
        self.predictor.commit(rows)
        self.seconds += time.perf_counter() - started
        self.cpu_seconds = getattr(self, "cpu_seconds", 0.0) + time.process_time() - cpu
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
                    prefill_max_bytes=self.prefill_max_bytes, budget_bytes=self.budget_bytes,
                    prefill_staging_peak_bytes=self.prefill_peak_bytes,
                    staging_peak_bytes=self.worker.peak_reserved_bytes,
                    predictor_seconds=self.seconds, predictor_process_cpu_seconds=self.cpu_seconds,
                    predictor_backend=self.predictor_backend, predictor_memory=predictor_memory(self.predictor),
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
        if getattr(self, '_prefill_pool', None) is not None:
            self._prefill_pool.shutdown(wait=True)
        if getattr(self, 'record', None) is not None:
            self.record.close()
            self.record = None
        try:
            self.worker.close()
        finally:
            self.reader.close()
            self.staging.cleanup()
