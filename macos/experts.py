import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from decoder import projection_matvec, projection_matmul
from prefetch import BoundedReader
from positional_reader import PositionalTensorReader


class ExpertStore:
    def __init__(self, root: Path, cache_gib: float = 4.0, resident_base: bool = False, async_refinements: bool = False,
                 base_cache_gib: float | None = None):
        if cache_gib < 0:
            raise ValueError('Expert cache size must be nonnegative')
        self.root = root
        self.memory_check = None
        self.cache_budget = int(cache_gib * 1024**3)
        self.staging_budget = 64 * 1024**2 if async_refinements else 0
        if async_refinements and (not resident_base or self.cache_budget <= self.staging_budget):
            raise ValueError('Asynchronous refinements require resident base and more than 64 MiB of cache')
        self.cache_limit = self.cache_budget - self.staging_budget
        self.reader = RefinementReader(root, use_preadv=True) if async_refinements else None
        self.prefetch = BoundedReader(self.reader.read, self.staging_budget) if self.reader else None
        self.async_targets = set()
        self.async_order = []
        self.async_priorities = {}
        self.async_budgets = {}
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.resident_base = resident_base
        if base_cache_gib is not None and (not resident_base or not math.isfinite(base_cache_gib) or base_cache_gib <= 0):
            raise ValueError('A bounded base cache requires resident-base mode and a positive finite budget')
        self.base_cache_limit = int(base_cache_gib * 1024**3) if base_cache_gib is not None else None
        self.base_cache = OrderedDict()
        self.base_sizes = {}
        self.base_bytes = 0
        self.handles = {}
        self.manifests = {}
        self.metrics = dict(hits=0, misses=0, load_seconds=0.0, expert_forward_seconds=0.0,
                            base_load_seconds=0.0,
                            base_tensor_bytes=0, refinement_tensor_bytes=0, hot_sync_seconds=0.0,
                            base_hits=0, base_misses=0, refinement_hits=0, refinement_misses=0,
                            async_load_seconds=0.0, async_loaded=0, async_discarded=0,
                            async_fallbacks=0, async_install_seconds=0.0, base_evictions=0)

    def _manifest(self, layer):
        if layer not in self.manifests:
            path = self.root / 'layers' / f'L{layer}' / 'manifest.json'
            self.manifests[layer] = json.loads(path.read_text())
        return self.manifests[layer]

    def refinement_bytes(self, layer, expert):
        manifest = self._manifest(layer)
        size = 0
        for projection, meta in manifest['proj_meta'].items():
            units = meta['tk'] * meta['tn']
            rank = manifest['per_expert'][str(expert)]['lr_rank'][projection]
            size += units * (8 * int(meta['res_rule']['K'] * 32) + 4)
            size += 2 * (meta['k'] + meta['n']) + 2 * rank * meta['n']
        return size

    def check_hot_capacity(self, budgets):
        if not self.resident_base:
            raise ValueError('Hot-set residency requires resident-base mode')
        required = sum(sum(sorted((self.refinement_bytes(layer, expert) for expert in range(288)),
                                  reverse=True)[:count]) for layer, count in budgets.items())
        if required > self.cache_limit:
            minimum_gib = math.ceil((required + self.staging_budget) / 1024**3 * 1000) / 1000
            raise ValueError(f'Hot-set residency requires at least {minimum_gib:.3f} GiB of refinement cache')
        return required

    def prepare_hot(self, experts):
        if not self.resident_base:
            raise ValueError('Hot-set residency requires resident-base mode')
        wanted = {(layer, expert, 4) for layer, expert in experts}
        if any(not 3 <= layer <= 44 or not 0 <= expert < 288 for layer, expert, _ in wanted):
            raise ValueError('Invalid Flash expert coordinates')
        required = sum(self.refinement_bytes(layer, expert) for layer, expert, _ in wanted)
        if required > self.cache_limit:
            raise ValueError('The predicted hot set exceeds the refinement cache budget')
        started = time.perf_counter()
        for key in list(self.cache):
            if key not in wanted:
                _, removed = self.cache.pop(key)
                self.cache_bytes -= removed
        for key in sorted(wanted - self.cache.keys()):
            self.get(*key)
        if set(self.cache) != wanted or self.cache_bytes != required:
            raise RuntimeError('Resident hot-set storage does not match the requested allocation')
        self.metrics['hot_sync_seconds'] += time.perf_counter() - started

    def _handle(self, layer: int, shard: int):
        key = layer, shard
        if key not in self.handles:
            path = self.root / 'layers' / f'L{layer}' / f'tp{shard}.safetensors'
            handle = safe_open(path, framework='numpy', backend='pread')
            if handle.metadata().get('format') != 'nestquant-v1-tp-safetensors':
                raise ValueError(f'Unsupported checkpoint container: {path}')
            self.handles[key] = handle, set(handle.keys())
        return self.handles[key]

    def get(self, layer: int, expert: int, level: int):
        if self.memory_check is not None:
            self.memory_check()
        if not 3 <= layer <= 44 or not 0 <= expert < 288 or level not in (2, 4):
            raise ValueError('Invalid Flash expert coordinates')
        key = layer, expert, level
        if self.resident_base and level == 2 and (layer, expert) in self.base_cache:
            self.base_cache.move_to_end((layer, expert))
            self.metrics['hits'] += 1
            self.metrics['base_hits'] += 1
            return self.base_cache[layer, expert]
        if key in self.cache:
            if self.resident_base:
                self.base_cache.move_to_end((layer, expert))
            self.metrics['hits'] += 1
            self.metrics['refinement_hits' if level == 4 else 'base_hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.metrics['misses'] += 1
        self.metrics['refinement_misses' if level == 4 else 'base_misses'] += 1
        if self.resident_base and level == 4:
            base = self.base_cache.get((layer, expert))
            if base is None:
                base = self.get(layer, expert, 2)
            else:
                self.base_cache.move_to_end((layer, expert))
            started = time.perf_counter()
            result = self._upgrade(layer, expert, base)
            size = sum(value.nbytes for projection, (planes, _) in result.items()
                       for name, value in planes.items() if value is not base[projection][0].get(name))
        else:
            started = time.perf_counter()
            result = self._load(layer, expert, level)
            size = sum(value.nbytes for planes, _ in result.values() for value in planes.values())
        elapsed = time.perf_counter() - started
        self.metrics['load_seconds'] += elapsed
        if level == 2:
            self.metrics['base_load_seconds'] += elapsed
        if self.resident_base and level == 2:
            if self.base_cache_limit is not None:
                if size > self.base_cache_limit:
                    raise ValueError('One base expert exceeds the base cache budget')
                while self.base_cache and self.base_bytes + size > self.base_cache_limit:
                    removed_key, _ = self.base_cache.popitem(last=False)
                    self.base_bytes -= self.base_sizes.pop(removed_key)
                    refinement = self.cache.pop((*removed_key, 4), None)
                    if refinement is not None:
                        self.cache_bytes -= refinement[1]
                    self.metrics['base_evictions'] += 1
            self.base_cache[layer, expert] = result
            self.base_sizes[layer, expert] = size
            self.base_bytes += size
            return result
        while self.cache and self.cache_bytes + size > self.cache_limit:
            _, (_, removed) = self.cache.popitem(last=False)
            self.cache_bytes -= removed
        if size <= self.cache_limit:
            self.cache[key] = result, size
            self.cache_bytes += size
        return result

    def _read(self, handle, name):
        value = handle.get_tensor(name)
        field = name.rsplit('.', 1)[-1]
        category = 'refinement_tensor_bytes' if field in ('p4', 'word', 'suh4', 'svh4', 'lrU4') else 'base_tensor_bytes'
        self.metrics[category] += value.nbytes
        return value

    def _load(self, layer, expert, level):
        manifest = self._manifest(layer)
        result = {}

        for projection in ('gate', 'up', 'down'):
            meta = manifest['proj_meta'][projection]
            if meta['base_K'] != 1.5 or meta['base_var'] != 'sign' or meta['res_rule']['kind'] != 'uniform':
                raise ValueError('Unsupported NestQuant Flash projection format')
            parts = []
            for shard in range(8):
                handle, keys = self._handle(layer, shard)
                prefix = f'E{expert}.{projection}.'
                fields = ('base', 'var', 'suh2', 'svh2')
                if level == 4:
                    fields += ('p4', 'word', 'suh4', 'svh4')
                part = {name: self._read(handle, prefix + name) for name in fields}
                if prefix + 'lrU2' in keys:
                    vkey = prefix + 'lrV'
                    if vkey not in keys:
                        if projection != 'up' or not meta['lr']['shared_V']:
                            raise ValueError('Missing low-rank input directions')
                        vkey = f'E{expert}.gate.lrV'
                    part.update(lrV=self._read(handle, vkey), lrU2=self._read(handle, prefix + 'lrU2'))
                    if level == 4:
                        part['lrU4'] = self._read(handle, prefix + 'lrU4')
                parts.append(part)
            merged = {name: np.concatenate([p[name] for p in parts]) for name in (('base', 'var') if level == 2 else ('base', 'p4', 'word', 'var'))}
            for scale_level in ((2,) if level == 2 else (2, 4)):
                for axis in ('su', 'sv'):
                    name = f'{axis}h{scale_level}'
                    split = (projection == 'down') == (axis == 'su')
                    merged[name] = np.concatenate([p[name] for p in parts]) if split else parts[0][name]
            if 'lrV' in parts[0]:
                merged['lrV'] = np.concatenate([p['lrV'] for p in parts], axis=1) if projection == 'down' else parts[0]['lrV']
                for name in (('lrU2',) if level == 2 else ('lrU2', 'lrU4')):
                    merged[name] = parts[0][name] if projection == 'down' else np.concatenate([p[name] for p in parts], axis=1)
            units = meta['tk'] * meta['tn']
            expected_shapes = dict(base=(units * 384,), var=(units * 8,), suh2=(meta['k'],), svh2=(meta['n'],))
            if level == 4:
                expected_shapes.update(p4=(units * 8 * int(meta['res_rule']['K'] * 32),),
                                       word=(units,), suh4=(meta['k'],), svh4=(meta['n'],))
            for field, shape in expected_shapes.items():
                if merged[field].shape != shape:
                    raise ValueError(f'Invalid {projection}.{field} shape: {merged[field].shape}, expected {shape}')
            if merged['base'].dtype != np.uint8 or merged['var'].dtype != np.uint8:
                raise ValueError('Packed base and sign variants must be uint8')
            if np.any(merged['var'] > 1):
                raise ValueError('Invalid sign variant')
            if level == 4:
                if merged['p4'].dtype != np.uint8 or merged['word'].dtype != np.int32:
                    raise ValueError('Invalid refinement plane dtypes')
                if np.any((merged['word'] & 255) == 0):
                    raise ValueError('Invalid zero denominator in refinement block')
            if level == 2:
                merged.update(p4=np.zeros(1, dtype=np.uint8), word=np.zeros(1, dtype=np.int32))
            result[projection] = ({name: mx.array(value) for name, value in merged.items()}, meta)
        arrays = [value for planes, _ in result.values() for value in planes.values()]
        mx.eval(arrays)
        return result

    def _upgrade(self, layer, expert, base):
        reader = RefinementReader(self.root)
        reader.handles, reader.manifests = self.handles, self.manifests
        payload, count, _ = reader.read(layer, expert)
        self.metrics['refinement_tensor_bytes'] += count
        return self._materialize(layer, expert, payload, base)

    def _materialize(self, layer, expert, payload, base=None):
        base = self.base_cache[layer, expert] if base is None else base
        result = {}
        for projection, (base_planes, meta) in base.items():
            planes = dict(base_planes)
            planes.update({name: mx.array(value) for name, value in payload[projection].items()})
            result[projection] = planes, meta
        mx.eval([value for planes, _ in result.values() for value in planes.values()])
        return result

    def begin_decode(self, wanted, priorities, budgets):
        if self.prefetch is None:
            return
        if wanted.shape != (42, 288) or priorities.shape != wanted.shape or not np.isfinite(priorities).all():
            raise ValueError('Invalid asynchronous predictor output')
        self.async_targets = {(int(layer) + 3, int(expert)) for layer, expert in np.argwhere(wanted)}
        self.async_priorities = {(layer + 3, expert): float(priorities[layer, expert])
                                 for layer in range(42) for expert in range(288)}
        self.async_order = sorted(self.async_targets, key=lambda key: (-self.async_priorities[key], key))
        self.async_budgets = budgets
        for layer, count in budgets.items():
            keys = [key for key in self.cache if key[0] == layer]
            keys.sort(key=lambda key: (key[:2] in self.async_targets, self.async_priorities[key[:2]]))
            for key in keys[:max(0, len(keys) - count)]:
                _, removed = self.cache.pop(key)
                self.cache_bytes -= removed
        self.poll_refinements()

    def poll_refinements(self):
        if self.prefetch is None:
            return

        def accept(key, completed):
            payload, count, elapsed = completed
            self.metrics['refinement_tensor_bytes'] += count
            self.metrics['async_load_seconds'] += elapsed
            if key not in self.async_targets or key not in self.base_cache:
                self.metrics['async_discarded'] += 1
                return
            layer, expert = key
            size = self.refinement_bytes(layer, expert)
            actual = sum(value.nbytes for planes in payload.values() for value in planes.values())
            if actual != size:
                raise ValueError('Refinement payload differs from its reserved size')
            candidates = [item for item in self.cache if item[0] == layer]
            if len(candidates) >= self.async_budgets[layer]:
                victim = min(candidates, key=lambda item: (item[:2] in self.async_targets,
                                                          self.async_priorities[item[:2]]))
                if victim[:2] in self.async_targets:
                    raise RuntimeError('Asynchronous admission would evict a desired expert')
                _, removed = self.cache.pop(victim)
                self.cache_bytes -= removed
            if self.cache_bytes + size > self.cache_limit:
                raise RuntimeError('Asynchronous admission exceeds the refinement budget')
            started = time.perf_counter()
            result = self._materialize(layer, expert, payload)
            self.cache[layer, expert, 4] = result, size
            self.cache_bytes += size
            self.metrics['async_install_seconds'] += time.perf_counter() - started
            self.metrics['async_loaded'] += 1

        self.prefetch.poll(accept)
        for key in self.async_order:
            if key in self.base_cache and (*key, 4) not in self.cache and key not in self.prefetch.pending:
                if not self.prefetch.submit(key, self.refinement_bytes(*key)):
                    break

    def finish_decode(self):
        if self.prefetch is not None:
            def discarded(completed):
                _, count, elapsed = completed
                self.metrics['refinement_tensor_bytes'] += count
                self.metrics['async_load_seconds'] += elapsed
                self.metrics['async_discarded'] += 1
            self.prefetch.discard(discarded)
            self.async_targets.clear()
            self.async_order.clear()

    def preload_base(self):
        if not self.resident_base:
            raise ValueError('Base preloading requires resident-base mode')
        if self.base_cache_limit is not None:
            raise ValueError('A bounded base cache loads experts on demand')
        for layer in range(3, 45):
            for expert in range(288):
                self.get(layer, expert, 2)
            print(json.dumps(dict(event='base_loaded', layer=layer, resident_bytes=self.base_bytes)), flush=True)

    def apply(self, layer: int, expert: int, x: mx.array, level: int):
        project = projection_matvec if x.ndim == 1 else projection_matmul
        projections = self.get(layer, expert, level)
        gate_planes, gate_meta = projections['gate']
        up_planes, up_meta = projections['up']
        down_planes, down_meta = projections['down']
        gate = project(gate_planes, gate_meta, x, level)
        up = project(up_planes, up_meta, x, level)
        gate = mx.minimum(gate, 10.0)
        up = mx.clip(up, -10.0, 10.0)
        intermediate = gate * mx.sigmoid(gate) * up
        return project(down_planes, down_meta, intermediate, level)

    def close(self):
        try:
            if self.prefetch is not None:
                self.prefetch.close()
        finally:
            if self.reader is not None:
                self.reader.close()
            self._clear()

    def _clear(self):
        self.cache.clear()
        self.cache_bytes = 0
        self.base_cache.clear()
        self.base_sizes.clear()
        self.base_bytes = 0
        for handle, _ in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


class RefinementReader:
    def __init__(self, root, use_preadv=False):
        self.root = root
        self.use_preadv = use_preadv
        self.handles = {}
        self.manifests = {}

    _manifest = ExpertStore._manifest

    def _handle(self, layer, shard):
        if not self.use_preadv:
            return ExpertStore._handle(self, layer, shard)[0]
        key = layer, shard
        if key not in self.handles:
            self.handles[key] = PositionalTensorReader(self.root / 'layers' / f'L{layer}' / f'tp{shard}.safetensors')
        return self.handles[key]

    def read(self, layer, expert):
        started = time.perf_counter()
        bytes_read = 0
        manifest = self._manifest(layer)
        result = {}
        for projection in ('gate', 'up', 'down'):
            meta = manifest['proj_meta'][projection]
            rank = manifest['per_expert'][str(expert)]['lr_rank'][projection]
            fields = ('p4', 'word', 'suh4', 'svh4')
            if rank:
                fields += ('lrU4',)
            parts = []
            for shard in range(8):
                handle = self._handle(layer, shard)
                part = {name: handle.get_tensor(f'E{expert}.{projection}.{name}') for name in fields}
                bytes_read += sum(value.nbytes for value in part.values())
                parts.append(part)
            merged = {name: np.concatenate([p[name] for p in parts]) for name in ('p4', 'word')}
            for axis in ('su', 'sv'):
                name = f'{axis}h4'
                split = (projection == 'down') == (axis == 'su')
                merged[name] = np.concatenate([p[name] for p in parts]) if split else parts[0][name]
            if rank:
                merged['lrU4'] = parts[0]['lrU4'] if projection == 'down' else np.concatenate([p['lrU4'] for p in parts], axis=1)
            units = meta['tk'] * meta['tn']
            shapes = dict(p4=(units * 8 * int(meta['res_rule']['K'] * 32),), word=(units,),
                          suh4=(meta['k'],), svh4=(meta['n'],))
            if 'lrU4' in merged:
                shapes['lrU4'] = (rank, meta['n'])
            for field, shape in shapes.items():
                if merged[field].shape != shape:
                    raise ValueError(f'Invalid {projection}.{field} shape: {merged[field].shape}, expected {shape}')
            if merged['p4'].dtype != np.uint8 or merged['word'].dtype != np.int32:
                raise ValueError('Invalid refinement plane dtypes')
            if np.any((merged['word'] & 255) == 0):
                raise ValueError('Invalid zero denominator in refinement block')
            result[projection] = {name: np.array(value, copy=True) for name, value in merged.items()}
        return result, bytes_read, time.perf_counter() - started

    def close(self):
        for handle in self.handles.values():
            if self.use_preadv:
                handle.close()
            else:
                handle[0].__exit__(None, None, None)
        self.handles.clear()


class NestQuantExperts(torch.nn.Module):
    def __init__(self, layer: int, store: ExpertStore, runtime):
        super().__init__()
        self.layer = layer
        self.store = store
        self.runtime = runtime

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != 4096:
            raise ValueError('Expected Flash token representations with 4096 channels')
        x_cpu = hidden_states.detach().float().cpu().numpy()
        ids = top_k_index.detach().cpu().numpy()
        weights = top_k_weights.detach().float().cpu().numpy()
        norms = np.square(x_cpu).sum(axis=-1)
        self.runtime.routing[self.layer] = (ids.copy(), weights.copy(), norms)
        x = mx.array(x_cpu).astype(mx.float16).astype(mx.float32)
        started = time.perf_counter()
        if self.runtime.phase == 'prefill':
            result = mx.zeros(hidden_states.shape, dtype=mx.float32)
            for expert in np.unique(ids):
                rows, columns = np.where(ids == expert)
                row_ids = mx.array(rows.astype(np.int32))
                output = self.store.apply(self.layer, int(expert), x[row_ids], 4)
                routing_weights = mx.array(weights[rows, columns].astype(np.float16)).astype(mx.float32)
                result = result.at[row_ids].add(output * routing_weights[:, None])
                mx.eval(result)
        else:
            if hidden_states.shape[0] != 1:
                raise ValueError('Decode currently supports one sequence')
            wanted = self.runtime.wanted[self.layer - 3]
            self.store.poll_refinements()
            outputs = []
            for expert, weight in zip(ids[0], weights[0]):
                desired = bool(wanted[int(expert)])
                ready = (self.layer, int(expert), 4) in self.store.cache
                level = 4 if (ready if self.store.prefetch is not None else desired) else 2
                self.runtime.decode_ready_experts += int(level == 4)
                if hasattr(self.runtime, 'ready_at_use'):
                    self.runtime.ready_at_use[self.layer - 3, int(expert)] = level == 4
                if self.store.prefetch is not None and desired and not ready:
                    self.store.metrics['async_fallbacks'] += 1
                outputs.append(self.store.apply(self.layer, int(expert), x[0], level) * float(np.float16(weight)))
            result = mx.stack(outputs).sum(axis=0)[None]
            mx.eval(result)
        self.store.metrics['expert_forward_seconds'] += time.perf_counter() - started
        return torch.from_numpy(np.array(result)).to(device=hidden_states.device, dtype=hidden_states.dtype)
