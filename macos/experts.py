import json
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from decoder import projection_matvec, projection_matmul


class ExpertStore:
    def __init__(self, root: Path, cache_gib: float = 4.0, resident_base: bool = False):
        if cache_gib < 0:
            raise ValueError('Expert cache size must be nonnegative')
        self.root = root
        self.cache_limit = int(cache_gib * 1024**3)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.resident_base = resident_base
        self.base_cache = {}
        self.base_bytes = 0
        self.handles = {}
        self.manifests = {}
        self.metrics = dict(hits=0, misses=0, load_seconds=0.0, expert_forward_seconds=0.0,
                            base_tensor_bytes=0, refinement_tensor_bytes=0)

    def _handle(self, layer: int, shard: int):
        key = layer, shard
        if key not in self.handles:
            path = self.root / 'layers' / f'L{layer}' / f'tp{shard}.safetensors'
            handle = safe_open(path, framework='numpy')
            if handle.metadata().get('format') != 'nestquant-v1-tp-safetensors':
                raise ValueError(f'Unsupported checkpoint container: {path}')
            self.handles[key] = handle, set(handle.keys())
        return self.handles[key]

    def get(self, layer: int, expert: int, level: int):
        if not 3 <= layer <= 44 or not 0 <= expert < 288 or level not in (2, 4):
            raise ValueError('Invalid Flash expert coordinates')
        key = layer, expert, level
        if self.resident_base and level == 2 and (layer, expert) in self.base_cache:
            self.metrics['hits'] += 1
            return self.base_cache[layer, expert]
        if key in self.cache:
            self.metrics['hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.metrics['misses'] += 1
        if self.resident_base and level == 4:
            base = self.get(layer, expert, 2)
            started = time.perf_counter()
            result = self._upgrade(layer, expert, base)
            size = sum(value.nbytes for projection, (planes, _) in result.items()
                       for name, value in planes.items() if value is not base[projection][0].get(name))
        else:
            started = time.perf_counter()
            result = self._load(layer, expert, level)
            size = sum(value.nbytes for planes, _ in result.values() for value in planes.values())
        self.metrics['load_seconds'] += time.perf_counter() - started
        if self.resident_base and level == 2:
            self.base_cache[layer, expert] = result
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
        if layer not in self.manifests:
            path = self.root / 'layers' / f'L{layer}' / 'manifest.json'
            self.manifests[layer] = json.loads(path.read_text())
        manifest = self.manifests[layer]
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
        result = {}
        for projection, (base_planes, meta) in base.items():
            fields = ('p4', 'word', 'suh4', 'svh4')
            if 'lrV' in base_planes:
                fields += ('lrU4',)
            parts = []
            for shard in range(8):
                handle, _ = self._handle(layer, shard)
                parts.append({name: self._read(handle, f'E{expert}.{projection}.{name}') for name in fields})
            merged = {name: np.concatenate([p[name] for p in parts]) for name in ('p4', 'word')}
            for axis in ('su', 'sv'):
                name = f'{axis}h4'
                split = (projection == 'down') == (axis == 'su')
                merged[name] = np.concatenate([p[name] for p in parts]) if split else parts[0][name]
            if 'lrV' in base_planes:
                merged['lrU4'] = parts[0]['lrU4'] if projection == 'down' else np.concatenate([p['lrU4'] for p in parts], axis=1)
            units = meta['tk'] * meta['tn']
            shapes = dict(p4=(units * 8 * int(meta['res_rule']['K'] * 32),), word=(units,),
                          suh4=(meta['k'],), svh4=(meta['n'],))
            if 'lrU4' in merged:
                shapes['lrU4'] = base_planes['lrU2'].shape
            for field, shape in shapes.items():
                if merged[field].shape != shape:
                    raise ValueError(f'Invalid {projection}.{field} shape: {merged[field].shape}, expected {shape}')
            if merged['p4'].dtype != np.uint8 or merged['word'].dtype != np.int32:
                raise ValueError('Invalid refinement plane dtypes')
            if np.any((merged['word'] & 255) == 0):
                raise ValueError('Invalid zero denominator in refinement block')
            planes = dict(base_planes)
            planes.update({name: mx.array(value) for name, value in merged.items()})
            result[projection] = planes, meta
        mx.eval([value for planes, _ in result.values() for value in planes.values()])
        return result

    def preload_base(self):
        if not self.resident_base:
            raise ValueError('Base preloading requires resident-base mode')
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
        self.cache.clear()
        self.cache_bytes = 0
        self.base_cache.clear()
        self.base_bytes = 0
        for handle, _ in self.handles.values():
            handle.__exit__(None, None, None)
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
            outputs = []
            for expert, weight in zip(ids[0], weights[0]):
                level = 4 if wanted[int(expert)] else 2
                outputs.append(self.store.apply(self.layer, int(expert), x[0], level) * float(np.float16(weight)))
            result = mx.stack(outputs).sum(axis=0)[None]
            mx.eval(result)
        self.store.metrics['expert_forward_seconds'] += time.perf_counter() - started
        return torch.from_numpy(np.array(result)).to(device=hidden_states.device, dtype=hidden_states.dtype)
