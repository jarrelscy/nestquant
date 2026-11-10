import argparse
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import torch

from generate import Runtime


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompt', default='Write a Python function that merges overlapping intervals. Explain its algorithm, edge cases, and time complexity. Include examples.')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cache-gib', type=float, default=4)
    parser.add_argument('--resident-base', action='store_true')
    parser.add_argument('--base-cache-gib', type=float)
    parser.add_argument('--resident-hot', action='store_true')
    parser.add_argument('--async-refinements', action='store_true')
    parser.add_argument('--async-base', action='store_true')
    parser.add_argument('--memory-limit-gib', type=float, default=84)
    parser.add_argument('--wired-memory', action='store_true')
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
                      resident_base=args.resident_base, resident_hot=args.resident_hot, wired_memory=args.wired_memory,
                      async_refinements=args.async_refinements, memory_limit_gib=args.memory_limit_gib,
                      base_cache_gib=args.base_cache_gib, async_base=args.async_base)
    load_seconds = time.perf_counter() - started
    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(['git', '-C', str(repository), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(repository), 'status', '--porcelain'], capture_output=True, text=True)
    metadata = dict(
        source_commit=source.stdout.strip() if source.returncode == 0 else None,
        source_dirty=bool(dirty.stdout) if dirty.returncode == 0 else None,
        expected_model_revision='4023870a6f2c6097e37fa5d2e54040db64b011e9',
        device=mx.device_info(), model=str(args.model.resolve()), load_seconds=load_seconds,
        packages={name: importlib.metadata.version(name) for name in ('mlx', 'torch', 'transformers', 'safetensors')},
        async_refinements=args.async_refinements, async_base=args.async_base,
        memory_limit_gib=args.memory_limit_gib, base_cache_gib=args.base_cache_gib,
        hot_fraction=args.hot_fraction, hot_slots=runtime.hot_slots, cache_gib=args.cache_gib,
        wired_memory=args.wired_memory, resident_base=args.resident_base, resident_hot=args.resident_hot, resident_base_bytes=runtime.store.base_bytes,
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
        first_decode_step_seconds=summary([r['decode_step_seconds'][0] for r in reports]),
        subsequent_decode_tokens_per_second=summary([(len(r['decode_step_seconds']) - 1) / sum(r['decode_step_seconds'][1:]) for r in reports]),
        decode_tokens_per_second=summary([r['decode_tokens_per_second'] for r in reports]),
        decode_hot_expert_fraction=summary([r['decode_hot_expert_fraction'] for r in reports]),
        decode_desired_hot_expert_fraction=summary([r['decode_desired_hot_expert_fraction'] for r in reports]),
        async_fallbacks=summary([r['expert_metrics_by_phase']['decode']['async_fallbacks'] for r in reports]),
        decode_base_misses=summary([r['expert_metrics_by_phase']['decode']['base_misses'] for r in reports]),
        decode_base_evictions=summary([r['expert_metrics_by_phase']['decode']['base_evictions'] for r in reports]),
        decode_base_load_seconds=summary([r['expert_metrics_by_phase']['decode']['base_load_seconds'] for r in reports]),
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
