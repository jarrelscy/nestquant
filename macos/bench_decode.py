"""Load the full-base Spark runtime once and compare decode variants in one process."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import torch

import native_generate as ng
from metrics import MemoryGuard
from native_refinements import NativeHotRefinements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--mlx-limit-gib', type=int, default=92)
    parser.add_argument('--variants', default='serial:0,pipelined:0,pipelined:4,serial:4')
    parser.add_argument('--prompt', default='Explain how a hash map works, in detail.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(args.mlx_limit_gib * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, 0.2174, 128, 22, 4, 32, 512)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    results = []
    try:
        warm = hot.warm(4, lambda: guard.check(force=True))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=True,
                                           return_dict=False, add_generation_prompt=True, chat_template=template)
        default_threads = torch.get_num_threads()
        import decoder
        original_kernel = decoder.mx.fast.metal_kernel

        def skip_matvec(**kw):
            # Timing probe: the window matvec writes zeros and returns, so GPU time drops while the
            # number of encoded operations stays the same.
            if kw['name'].startswith('nq_matvec_window'):
                kw = dict(kw, name=kw['name'] + '_probe',
                          source=kw['source'].replace('uint group = threadgroup_position_in_grid.x;',
                                                      'partial[0] = 0.0f; return;\n    uint group = threadgroup_position_in_grid.x;', 1))
            return original_kernel(**kw)

        for variant in args.variants.split(','):
            mode, threads = variant.split(':')
            decoder._window_kernel.cache_clear()
            decoder.mx.fast.metal_kernel = skip_matvec if mode == 'probe' else original_kernel
            ng.PIPELINE = mode in ('pipelined', 'probe')
            torch.set_num_threads(int(threads) or default_threads)
            before = ng.vm_snapshot()['counters']
            out = ng.generate(model, tokenizer, ids, args.tokens, set(), ng.layer_major_prefill, 8, hot, guard)
            after = ng.vm_snapshot()['counters']
            r = out['refinement']
            steps = sorted(out['decode_step_seconds'][4:])
            results.append(dict(
                variant=variant, torch_threads=torch.get_num_threads(), decode_tps=out['decode_tps'],
                median_step_ms=1000 * steps[len(steps) // 2], ttft_seconds=out['ttft_seconds'],
                ready_call_fraction=r['coverage']['call_fractions']['ready'],
                installed=r['installed'], publication_s=r['decode_publication_seconds'],
                predictor_s=r['predictor_seconds'], prefill_bytes=r['prefill_bytes'],
                compressions=after['Compressions'] - before['Compressions'],
                decompressions=after['Decompressions'] - before['Decompressions'],
                pageins=after['Pageins'] - before['Pageins'], text=out['text'][:60],
                phases_ms={k: round(1000 * v, 2) for k, v in (out['pipeline_phase_seconds'] or {}).items()},
                step_p90_ms=1000 * steps[int(len(steps) * 0.9)], step_max_ms=1000 * steps[-1],
                slow_steps=sum(step > 0.2 for step in steps),
                peak_gib=ng.process_usage().lifetime_max_phys_footprint / 2**30,
                phys_gib=ng.process_usage().phys_footprint / 2**30,
                compressor_gib=after['Pages occupied by compressor'] * 16384 / 2**30,
                pressure=ng.system_pressure()))
            print(json.dumps(results[-1]), flush=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()
    args.output.write_text(json.dumps(dict(warm=warm, results=results), indent=1))


if __name__ == '__main__':
    main()
