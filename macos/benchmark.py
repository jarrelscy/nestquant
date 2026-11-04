import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import torch

from generate import Runtime


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompt', default='Write a Python function that merges overlapping intervals. Explain its algorithm, edge cases, and time complexity. Include examples.')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cache-gib', type=float, default=4)
    parser.add_argument('--resident-base', action='store_true')
    parser.add_argument('--resident-hot', action='store_true')
    parser.add_argument('--hot-fraction', type=float, default=2630 / 12096)
    parser.add_argument('--prefill-chunk-size', type=int, default=256)
    args = parser.parse_args()
    if args.runs < 3 or args.tokens < 16 or args.prefill_chunk_size < 1:
        parser.error('Use at least three runs, sixteen output tokens, and a positive chunk size')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve earlier measurements')
    torch.set_num_threads(8)
    mx.set_cache_limit(1024**3)
    started = time.perf_counter()
    runtime = Runtime(args.model, args.cache_gib, hot_fraction=args.hot_fraction,
                      resident_base=args.resident_base, resident_hot=args.resident_hot)
    load_seconds = time.perf_counter() - started
    metadata = dict(
        device=mx.device_info(), model=str(args.model.resolve()), load_seconds=load_seconds,
        packages={name: importlib.metadata.version(name) for name in ('mlx', 'torch', 'transformers', 'safetensors')},
        hot_fraction=args.hot_fraction, hot_slots=runtime.hot_slots, cache_gib=args.cache_gib,
        resident_base=args.resident_base, resident_hot=args.resident_hot, resident_base_bytes=runtime.store.base_bytes,
        prefill_chunk_size=args.prefill_chunk_size, requested_output_tokens=args.tokens, prompt=args.prompt,
        sampling='greedy', method='One discarded warm-up, then at least three measured requests in one process',
        limitations='Process lifetime peak includes model loading; measurements do not grade answer quality',
    )
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    reports = []
    try:
        warm = runtime.generate(args.prompt, min(16, args.tokens), prefill_chunk_size=args.prefill_chunk_size)
        (args.output_dir / 'warmup.json').write_text(json.dumps(warm, indent=2) + '\n')
        for run in range(args.runs):
            report = runtime.generate(args.prompt, args.tokens, prefill_chunk_size=args.prefill_chunk_size)
            (args.output_dir / f'run-{run + 1}.json').write_text(json.dumps(report, indent=2) + '\n')
            if report['decode_tokens_per_second'] is None or report['generated_tokens'] < args.tokens:
                raise RuntimeError(f'Run produced {report["generated_tokens"]} tokens; {args.tokens} required. '
                                   'Use a prompt that produces a longer answer for every compared setting.')
            reports.append(report)
    finally:
        runtime.close()
    result = dict(
        status='complete', runs=len(reports),
        decode_tokens_per_second=summary([r['decode_tokens_per_second'] for r in reports]),
        time_to_first_token_seconds=summary([r['ttft_seconds'] for r in reports]),
        prefill_tokens_per_second=summary([r['prefill_tokens_per_second'] for r in reports]),
        prefill_disk_read_bytes=summary([r['disk_reads']['prefill'] for r in reports]),
        decode_disk_read_bytes=summary([r['disk_reads']['decode'] for r in reports]),
        process_peak_physical_footprint_bytes=max(r['resources']['after_decode']['lifetime_peak_physical_footprint_bytes'] for r in reports),
        generated_tokens=[r['generated_tokens'] for r in reports],
        identical_generated_tokens=all(r['generated_ids'] == reports[0]['generated_ids'] for r in reports),
        metadata=metadata,
    )
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
