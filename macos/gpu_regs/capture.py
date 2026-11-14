"""Dump the Metal source of every nq_matvec window kernel launched by expert_mlp_fused (decode-1/verify-2/verify-8 x
base/mixed/refined) into OUT/<name>.metal, then run ./stats on them.  Usage: capture.py OUT [JSON settings]"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import mlx.core as mx

import decoder
from microbench_window import make_layer

out_dir = sys.argv[1]
settings = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
for key, value in settings.items():
    setattr(decoder, key, tuple(value) if isinstance(value, list) else value)
os.makedirs(out_dir, exist_ok=True)

raw = mx.fast.metal_kernel
seen = set()


EDITS = settings.pop('SRC_RE', [])


def wrapped(**kw):
    if kw['name'].startswith('nq_matvec'):
        import re
        for pattern, replacement in EDITS:
            kw['source'], hits = re.subn(pattern, replacement, kw['source'])
            assert hits, pattern
    kernel = raw(**kw)
    if not kw['name'].startswith('nq_matvec'):
        return kernel

    def call(*a, **k):
        return kernel(*a, **k, verbose=True)
    return call


mx.fast.metal_kernel = wrapped
mx.random.seed(1)
random.seed(1)
depth, hot = 24, 8
layer = make_layer(depth, hot, 1)
planes = {n: (layer[n][0], layer[n][2]) for n in ('gate', 'up', 'down')}
refs = {n: layer[n][1] for n in ('gate', 'up', 'down')}
for tokens in (1, 2, 8):
    ids = [e for _ in range(tokens) for e in random.sample(range(16), 8)]
    route = mx.array(ids, dtype=mx.uint32)
    mixed = mx.array([e % hot if e % 3 else 0xffffffff for e in ids], dtype=mx.uint32)
    x = mx.random.normal((tokens, 4096)).astype(mx.float16).astype(mx.float32)
    for refined in (None, mixed):
        y = decoder.expert_mlp_fused(planes, x, route, 8, refinements=refs if refined is not None else None,
                                     refined_route=refined)
        mx.eval(y)
