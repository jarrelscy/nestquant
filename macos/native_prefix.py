"""Bounded in-process checkpoints for the native single-sequence GLM runtime."""

from collections import OrderedDict
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, PoolingCache


def leaves(cache):
    for entry in cache:
        if type(entry) is CacheList:
            yield from leaves(entry.caches)
        elif type(entry) in (ArraysCache, KVCache, PoolingCache):
            yield entry
        else:
            raise RuntimeError(f'Unsupported native prefix cache component: {type(entry).__name__}')


def clone(value):
    if isinstance(value, mx.array):
        return mx.contiguous(mx.array(value))
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    return value


def arrays(value):
    return [item for _, item in tree_flatten(value) if isinstance(item, mx.array)]


def predictor_snapshot(hot):
    p = hot.predictor
    if hot.traces or hot.phase != 'prefill' or hot.position != p.t or p.ahead is not None or p.ahead_started:
        raise RuntimeError('Prefix checkpoints require a committed prefill boundary')
    policies = [(policy.inner.t, policy.state.copy(), policy.cur.copy()) for policy in p.policies]
    history = [tuple(tensor.clone() for tensor in entry) if entry else None for entry in p.net.cache]
    state = (p.t, p.prepared_t, p.net.position, p.previous_ids.copy(), p.previous_q.copy(),
             p.block.copy(), policies, history)
    size = sum(tensor.numel() * tensor.element_size() for entry in history if entry for tensor in entry)
    size += sum(item.nbytes for item in state[3:6])
    size += sum(ema.nbytes + chosen.nbytes for _, ema, chosen in policies)
    return state, size


def restore_predictor(hot, state):
    p = hot.predictor
    clock, p.prepared_t, p.net.position, ids, q, block, policies, history = state
    p.t = hot.position = clock
    p.previous_ids, p.previous_q, p.block = ids.copy(), q.copy(), block.copy()
    p.net.cache = [tuple(tensor.clone() for tensor in entry) if entry else None for entry in history]
    for policy, (pos, ema, chosen) in zip(p.policies, policies, strict=True):
        policy.inner.t = pos
        policy.inner.state = ema.copy()
        policy.inner.cur = chosen.copy()


@dataclass
class Checkpoint:
    parts: tuple
    logits: mx.array
    predictor: tuple | None
    bytes: int


class PrefixCache:
    def __init__(self, model, hot, chunk, boundary=32, budget=512 * 1024**2):
        if not 0 < chunk <= boundary or boundary % chunk or budget <= 0:
            raise ValueError('Prefix boundaries must be aligned with prefill chunks')
        self.chunk = chunk
        self.model = model
        self.hot = hot
        self.boundary = boundary
        self.budget = budget
        self.entries = OrderedDict()
        self.bytes = 0
        self.evictions = 0

    def put(self, key, model_cache, logits, hot, guard):
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        entries = tuple(leaves(model_cache))
        if any(getattr(entry, 'is_speculating', False) for entry in entries):
            raise RuntimeError('Cannot snapshot provisional native cache state')
        raw = tuple(entry.prefix_cache_snapshot() for entry in entries)
        sidecar, cpu_bytes = predictor_snapshot(hot) if hot is not None else (None, 0)
        size = sum(item.nbytes for item in arrays(raw)) + logits.nbytes + cpu_bytes
        if size > self.budget:
            return
        while self.bytes + size > self.budget:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.bytes
            self.evictions += 1
        states, result = clone(raw), clone(logits)
        mx.eval(result, arrays(states))
        guard.check(force=True)
        self.entries[key] = Checkpoint(tuple(zip(map(type, entries), states, strict=True)), result, sidecar, size)
        self.bytes += size

    def restore(self, ids, model_cache, hot, guard):
        key = max((key for key in self.entries if ids[:len(key)] == key
                   and (len(key) % self.boundary == 0 or len(key) == len(ids))), key=len, default=None)
        if key is None:
            return 0, None
        state = self.entries[key]
        if (state.predictor is None) != (hot is None):
            raise RuntimeError('Native prefix checkpoint precision differs')
        for entry, (kind, part) in zip(leaves(model_cache), state.parts, strict=True):
            if type(entry) is not kind:
                raise RuntimeError('Native prefix checkpoint component differs')
            entry.prefix_cache_restore(clone(part))
        if hot is not None:
            restore_predictor(hot, state.predictor)
        mx.eval([entry.state for entry in model_cache])
        guard.check(force=True)
        self.entries.move_to_end(key)
        return len(key), state.logits

    def prefill(self, run, model, ids, model_cache, hot, guard, reuse=True):
        if model is not self.model or hot is not self.hot:
            raise RuntimeError('Prefix checkpoints belong to one loaded runtime')
        ids = tuple(ids)
        hits, logits = self.restore(ids, model_cache, hot, guard) if reuse else (0, None)
        for start in range(hits, len(ids), self.boundary):
            end = min(len(ids), start + self.boundary)
            chunks = [list(ids[i:min(i+self.chunk, end)]) for i in range(start, end, self.chunk)]
            logits = run(model, chunks, model_cache, guard, hot)
            if reuse:
                self.put(ids[:end], model_cache, logits, hot, guard)
        return logits, hits
