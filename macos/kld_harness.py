"""Score teacher-forced KL divergence for the reference runtime.

Prefill rows are captured from layer-major prefill, which reconstructs every routed expert at 4 bits.
Decode rows come from single-token steps under the configured hot allocation and readiness policy.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open
import torch

import generate as reference
from kld_alignment import decode_reference_rows, load_tokens, teacher_rows
from kld_math import summarize, tokenwise_kld
from metrics import snapshot

ROWS = 128


class PrefillTap:
    def __enter__(self):
        self.original = reference.layer_major_prefill
        self.hidden = None

        def tapped(*args, **kwargs):
            output = self.original(*args, **kwargs)
            self.hidden = output.last_hidden_state
            return output

        reference.layer_major_prefill = tapped
        return self

    def __exit__(self, *error):
        reference.layer_major_prefill = self.original


def start_request(runtime):
    runtime.store.finish_decode()
    runtime.reset()


@torch.inference_mode()
def prefill_logits(runtime, tokens, chunk):
    start_request(runtime)
    with PrefillTap() as tap:
        runtime.prefill(tokens, chunk)
    hidden = tap.hidden
    if hidden is None or hidden.shape[1] != len(tokens):
        raise RuntimeError('Prefill did not return one hidden state per token')
    logits = np.empty((len(tokens), runtime.head.shape[0]), dtype=np.float32)
    for start in range(0, len(tokens), ROWS):
        logits[start:start + ROWS] = torch.nn.functional.linear(
            hidden[:, start:start + ROWS], runtime.head).float().cpu().numpy()[0]
    if not np.isfinite(logits).all():
        raise FloatingPointError('Nonfinite prefill logits')
    return logits


def decode_logits(runtime, tokens, prompt_length, chunk, limit):
    start_request(runtime)
    runtime.prefill(tokens[:prompt_length], chunk)
    before = runtime.store.metrics.copy()
    forced = tokens[prompt_length:prompt_length + limit] if limit else tokens[prompt_length:]
    rows, seconds = [], []
    for token in forced:
        started = time.perf_counter()
        rows.append(runtime.step(token).numpy())
        seconds.append(time.perf_counter() - started)
    runtime.store.finish_decode()
    routed = runtime.decode_routed_experts
    stats = dict(
        steps=len(seconds), decode_tokens_per_second=len(seconds) / sum(seconds),
        desired_hot_share=runtime.decode_hot_experts / routed, executed_hot_share=runtime.decode_ready_experts / routed,
        expert_metrics={key: value - before[key] for key, value in runtime.store.metrics.items()},
        predictor=runtime.predictor_metrics.report())
    return np.stack(rows), stats


def score(teacher, student):
    kld = np.concatenate([tokenwise_kld(teacher[start:start + ROWS], student[start:start + ROWS])
                          for start in range(0, len(teacher), ROWS)])
    return kld, summarize(kld, teacher, student)


def pooled(items):
    kld = np.concatenate([item['kld'] for item in items])
    agree = sum(item['summary']['top1_agreement'] * item['summary']['count'] for item in items)
    return dict(count=int(kld.size), mean=float(kld.mean()), p50=float(np.quantile(kld, 0.5)),
                p95=float(np.quantile(kld, 0.95)), p99=float(np.quantile(kld, 0.99)), max=float(kld.max()),
                top1_agreement=agree / kld.size)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--mode', choices=('teacher-prefill', 'reference-prefill', 'decode'), required=True)
    parser.add_argument('--tokens', type=Path, nargs='+', required=True)
    parser.add_argument('--teacher-logits', type=Path, nargs='*', default=[])
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--max-decode-tokens', type=int, default=0)
    parser.add_argument('--cache-gib', type=float, default=4)
    parser.add_argument('--resident-base', action='store_true')
    parser.add_argument('--base-cache-gib', type=float)
    parser.add_argument('--async-refinements', action='store_true')
    parser.add_argument('--memory-limit-gib', type=float, default=84)
    parser.add_argument('--guard-above-default-cap', action='store_true',
                        help='After loading, raise the process guard to --memory-limit-gib under an external host watchdog')
    parser.add_argument('--hot-fraction', type=float, default=0.02)
    parser.add_argument('--prefill-chunk-size', type=int, default=64)
    args = parser.parse_args()
    report_path = args.output_dir / f'{args.arm}.{args.mode}.json'
    if report_path.exists():
        parser.error('Use a fresh arm name for this output directory')
    if args.mode == 'teacher-prefill' and len(args.teacher_logits) != len(args.tokens):
        parser.error('Give one teacher logit file per token window')
    if args.mode != 'teacher-prefill' and args.reference_dir is None:
        parser.error('Trace scoring requires a reference directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = reference.Runtime(
        args.model, args.cache_gib, 'mps', args.hot_fraction, args.resident_base, False, False,
        args.async_refinements, args.memory_limit_gib, args.base_cache_gib)
    if args.guard_above_default_cap:
        runtime.memory_guard.limit_bytes = max(runtime.memory_guard.limit_bytes, int(args.memory_limit_gib * 1024**3))
    items = []
    try:
        for index, path in enumerate(args.tokens):
            tokens, prompt_length = load_tokens(path)
            if len(tokens) > 4096:
                raise ValueError('The reference path is limited to 4096 tokens')
            started = time.perf_counter()
            if args.mode == 'teacher-prefill':
                with safe_open(args.teacher_logits[index], 'np') as file:
                    teacher = file.get_tensor('logits')
                student = teacher_rows(prefill_logits(runtime, tokens, args.prefill_chunk_size), len(teacher))
                kld, summary = score(teacher, student)
                item = dict(name=path.stem, tokens=len(tokens), reference='BF16 teacher logits')
            elif args.mode == 'reference-prefill':
                if prompt_length is None:
                    raise ValueError('Reference capture requires a generated trace')
                args.reference_dir.mkdir(parents=True, exist_ok=True)
                np.save(args.reference_dir / f'{path.stem}.npy', prefill_logits(runtime, tokens, args.prefill_chunk_size))
                kld, summary = None, None
                item = dict(name=path.stem, tokens=len(tokens), prompt_tokens=prompt_length)
            else:
                if prompt_length is None:
                    raise ValueError('Decode scoring requires a generated trace')
                student, stats = decode_logits(runtime, tokens, prompt_length, args.prefill_chunk_size,
                                               args.max_decode_tokens)
                teacher = decode_reference_rows(np.load(args.reference_dir / f'{path.stem}.npy', mmap_mode='r'),
                                                prompt_length, len(student))
                kld, summary = score(np.asarray(teacher), student)
                item = dict(name=path.stem, tokens=len(tokens), prompt_tokens=prompt_length,
                            reference='all-4-bit layer-major prefill of the same trace', decode=stats)
            item.update(seconds=time.perf_counter() - started, summary=summary, resources=snapshot())
            if kld is not None:
                np.save(args.output_dir / f'{args.arm}.{path.stem}.kld.npy', kld)
                item['kld'] = kld
            items.append(item)
            print(json.dumps(dict(event='scored', name=item['name'], seconds=round(item['seconds'], 1),
                                  mean_kld=summary['mean'] if summary else None,
                                  top1=summary['top1_agreement'] if summary else None)), flush=True)
        scored = [item for item in items if 'kld' in item]
        report = dict(
            arm=args.arm, mode=args.mode, pooled=pooled(scored) if scored else None,
            configuration=dict(hot_fraction=args.hot_fraction, hot_slots=runtime.hot_slots,
                               hot_experts_per_layer=runtime.hot_slots / 42, resident_base=args.resident_base,
                               base_cache_gib=args.base_cache_gib, cache_gib=args.cache_gib,
                               async_refinements=args.async_refinements, prefill_chunk_size=args.prefill_chunk_size,
                               max_decode_tokens=args.max_decode_tokens, fixed_fraction=0.2, jt_mix=0.75,
                               predictor_preset='spark_256K_U_2630_flat50',
                               memory_guard_bytes=runtime.memory_guard.limit_bytes),
            items=[{key: value for key, value in item.items() if key != 'kld'} for item in items])
        report_path.write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps(dict(event='complete', arm=args.arm, mode=args.mode, pooled=report['pooled'])), flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
