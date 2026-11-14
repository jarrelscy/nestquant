"""Check mlx-vlm's weight-once verify qmv (optimized_affine_linear) against the per-token path the backbone uses for
2-8 token blocks (singleton gather_qmm), on every quantized projection of the prepared backbone, and time both.
Loads only the backbone (~5.5 GiB). Run inside the GPU lock."""
import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.models.linear import native_batch_linear
from mlx_vlm.models.quantized_verifier import optimized_affine_linear
from native_backbone import load_prepared

HOME = Path.home()


def timed(fn, n=20):
    for _ in range(3):
        mx.eval(fn())
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - t)
    return statistics.median(samples) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    parser.add_argument('--backbone', type=Path, default=HOME / 'Models/GLM/backbone-affine4-e8-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mx.set_memory_limit(24 * 1024**3)
    model = load_prepared(args.model, args.backbone, lambda *a: None)
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)]
    mx.random.seed(0)
    summary = dict(modules=len(modules), checked=0, unsupported=[], mismatches=[], shapes={})
    timing = {}
    for name, module in modules:
        k = module.weight.shape[1] * 32 // module.bits
        for t in (2, 3, 4, 5, 6, 7, 8):
            x = (mx.random.normal((1, t, k)) * 0.5).astype(module.scales.dtype)
            fast = optimized_affine_linear(module, x)
            if fast is None:
                summary['unsupported'].append(name)
                break
            ref = native_batch_linear(module, x)
            single = mx.concatenate([module(x[:, i:i + 1]) for i in range(t)], axis=1)
            mx.eval(fast, ref, single)
            summary['checked'] += 1
            if not (mx.array_equal(fast, ref).item() and mx.array_equal(fast, single).item()):
                summary['mismatches'].append(dict(name=name, t=t, vs_block=float(mx.max(mx.abs(fast - ref)).item()),
                                                  vs_decode=float(mx.max(mx.abs(fast - single)).item())))
        shape = (module.weight.shape[0], k, module.bits)
        if shape not in summary['shapes'] and name not in summary['unsupported']:
            summary['shapes'][shape] = name
            for t in (1, 4, 8):
                x = (mx.random.normal((1, t, k)) * 0.5).astype(module.scales.dtype)
                timing[f'{shape}/T{t}'] = dict(block_ms=round(timed(lambda: native_batch_linear(module, x)), 4),
                                               weight_once_ms=round(timed(lambda: optimized_affine_linear(module, x)), 4))
    summary['unsupported'] = sorted(set(summary['unsupported']))
    summary['shapes'] = {str(k): v for k, v in summary['shapes'].items()}
    summary['timing'] = timing
    summary['mismatch_count'] = len(summary['mismatches'])
    summary['mismatches'] = summary['mismatches'][:40]
    print(json.dumps(summary, indent=1))
    args.output.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
