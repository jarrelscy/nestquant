import argparse
import json
from pathlib import Path

import mlx.core as mx

from experts import ExpertStore


def verify(model):
    reference = ExpertStore(model, cache_gib=0, resident_base=True, base_cache_gib=0.03)
    staged = ExpertStore(model, cache_gib=0, resident_base=True, base_cache_gib=0.03, async_base=True)
    try:
        for layer in (3, 44):
            experts = (0, 7, 98, 177)
            staged.queue_base(layer, experts)
            for future, _ in staged.base_prefetch.pending.values():
                future.result(timeout=60)
            for expert in experts:
                expected = reference.get(layer, expert, 2)
                actual = staged.get(layer, expert, 2)
                for name, (planes, meta) in expected.items():
                    observed, observed_meta = actual[name]
                    if meta != observed_meta or set(planes) != set(observed):
                        raise RuntimeError('Async base changed the packed layout')
                    if not all(mx.array_equal(value, observed[key]).item() for key, value in planes.items()):
                        raise RuntimeError('Async base changed packed weights')
            if staged.base_prefetch.pending:
                raise RuntimeError('A requested base expert did not release its staging reservation')
        fallback = staged.get(44, 287, 2)
        expected = reference.get(44, 287, 2)
        if not mx.array_equal(fallback['down'][0]['base'], expected['down'][0]['base']).item():
            raise RuntimeError('The synchronous demand fallback did not return the required base')
        if staged.metrics['base_async_reads'] < 1 or staged.base_bytes > staged.base_cache_limit:
            raise RuntimeError('Async coverage or the cache bound was not verified')
        return dict(status='passed', metrics=staged.metrics,
                    peak_staging_bytes=staged.base_prefetch.peak_reserved_bytes,
                    staging_limit_bytes=staged.base_prefetch.budget_bytes)
    finally:
        try:
            staged.close()
        finally:
            reference.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model)))
