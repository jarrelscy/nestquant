"""Per-launch GPU time of the routed-expert kernels for decode and speculative verify (random weights).

Records every Metal kernel launch made by one MoE layer pass (gate, up, SwiGLU, down), then replays
each kernel name on its own with the recorded, already evaluated inputs. Times are ms per MoE layer.
Run inside the GPU lock: /tmp/nq_model_lock.sh <owner> python profile_verify_launches.py
"""
import os
import random
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx

_RECORD = []
_original = mx.fast.metal_kernel


def _recording_kernel(*args, **kwargs):
    kernel = _original(*args, **kwargs)
    name = kwargs.get('name', args[0] if args else '?')

    def call(*a, **kw):
        out = kernel(*a, **kw)
        if _RECORD is not None and _RECORD_ON[0]:
            _RECORD.append((name, kernel, a, kw))
        return out
    return call


_RECORD_ON = [False]
mx.fast.metal_kernel = _recording_kernel

import decoder  # noqa: E402
from microbench_window import make_layer  # noqa: E402

COLD = 0xffffffff
L = 4


def timed(fn, n=30):
    for _ in range(5):
        mx.eval(fn())
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) / L * 1e3


def main():
    mx.random.seed(0)
    random.seed(3)
    layers = [make_layer(288, 64, 1) for _ in range(L)]
    for label, tokens, uniq in (('decode-1', 1, 8), ('verify-4', 4, 23), ('verify-8', 8, 37)):
        pool = random.sample(range(288), uniq)
        ids_list = []
        for _ in range(tokens):
            ids_list += random.sample(pool, 8)
        ids = mx.array(ids_list, dtype=mx.uint32)
        slot = {e: i % 64 for i, e in enumerate(dict.fromkeys(ids_list))}
        half_hot = mx.array([slot[e] if e % 2 == 0 else COLD for e in ids_list], dtype=mx.uint32)
        x = mx.random.normal((ids.size, 4096))
        mx.eval(x)
        for precision, refined in (('1.5-bit', None), ('half-4-bit', half_hot)):
            def run():
                out = []
                for layer in layers:
                    out.append(_mlp(layer, x, ids, refined))
                return out
            total = timed(run)
            _RECORD.clear()
            _RECORD_ON[0] = True
            mx.eval(run())
            _RECORD_ON[0] = False
            by_name = defaultdict(list)
            for name, kernel, a, kw in _RECORD:
                by_name[name].append((kernel, a, kw))
            parts = []
            for name, calls in by_name.items():
                ms = timed(lambda: [k(*a, **kw) for k, a, kw in calls])
                parts.append(f'{name} x{len(calls) // L} {ms:.3f}')
            print(f'{label:9s} rows {ids.size:2d} uniq {len(set(ids_list)):2d} {precision:10s} '
                  f'total {total:.3f} ms/layer | ' + ' | '.join(parts), flush=True)


def _mlp(layer, x, ids, refined):
    def project(name, value):
        base, refinements, meta = layer[name]
        options = dict(refinements=refinements, refined_route=refined) if refined is not None else {}
        return decoder.projection_fused(base, meta, value, ids, **options)
    gate = mx.minimum(project('gate', x), 10.0)
    up = mx.clip(project('up', x), -10.0, 10.0)
    return project('down', gate * mx.sigmoid(gate) * up)


if __name__ == '__main__':
    main()
