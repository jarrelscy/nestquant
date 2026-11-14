"""Forced-token replay comparison of the short-block kernels (backbone_fast flags) against the previous code, on
the full runtime with a frozen hot set, at several context lengths.

For each context: a batched prefill of repository source, 24 plain decode tokens (whose tokens become the forced
block), then the hot set is frozen and every configuration verifies the same 1/4/8-token blocks from the same
cache state (rolled back after each call). Reports max/mean absolute logit difference and top-1 agreement against
the reference configuration, plus interleaved verify timings (median ms per call).
Run inside the GPU lock: /tmp/nq_model_lock.sh kernels python check_kernels_exact.py --output out.json
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import mlx.core as mx
import torch

HOME = Path.home()
FLAGS = ('FAST_INDEXER', 'TRIM_TOPK')


def configured(backbone_fast, enabled):
    stack = ExitStack()
    for flag in FLAGS:
        if hasattr(backbone_fast, flag):
            stack.enter_context(patch.object(backbone_fast, flag, flag in enabled))
    return stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    parser.add_argument('--archive', type=Path, default=HOME / 'Models/GLM/resident-banks-v1')
    parser.add_argument('--backbone', type=Path, default=HOME / 'Models/GLM/backbone-affine4-e8-v1')
    parser.add_argument('--drafter', type=Path, default=HOME / 'Models/GLM/drafters/dflash2')
    parser.add_argument('--contexts', default='1000,8000,16000,30000')
    parser.add_argument('--blocks', default='1,4,8')
    parser.add_argument('--configs', default='ref:,new:FAST_INDEXER+TRIM_TOPK,notrim:FAST_INDEXER',
                        help='name:flag+flag,... ; the first is the reference')
    parser.add_argument('--repeat', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import backbone_fast
    import native_dflash as nd
    import native_generate as ng
    import profile_dflash as pdf
    from metrics import MemoryGuard
    from native_prefill import BatchedPrefill
    from native_refinements import NativeHotRefinements
    from mlx_vlm.speculative.common import verify_forward
    from mlx_vlm.speculative.cache_state import abort_speculative_round
    from mlx_vlm.speculative.drafters import load_drafter
    from transformers import AutoTokenizer

    configs = []
    for item in args.configs.split(','):
        name, _, flags = item.partition(':')
        configs.append((name, set(filter(None, flags.split('+')))))
    blocks = [int(b) for b in args.blocks.split(',')]

    nd._install_capture()
    nd._relax_compatibility()
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(96 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, 0.15, 128, 16, nd.READERS, 48, 512)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    results = []
    try:
        hot.warm(4, lambda: guard.check(force=True))
        nd.BATCHED = BatchedPrefill(args.model)
        drafter, _ = load_drafter(str(args.drafter))
        wanted = list(drafter.config.target_layer_ids)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        files = sorted(Path(pdf.__file__).resolve().parent.glob('*.py'))
        publish, flip = hot.publish, hot.flip
        for context in map(int, args.contexts.split(',')):
            hot.publish, hot.flip = publish, flip
            ids = pdf.context_ids(tokenizer, template, files, context)
            cache = model.make_cache()
            hot.reset()
            with configured(backbone_fast, configs[0][1]):
                logits, _ = nd.capture_prefill(model, drafter, ids, cache, hot, guard)
                token = ng.greedy(logits)
                _, tokens, _, _ = pdf.plain_decode(ng, model, cache, hot, guard, token, 24, None)
            hot.drain()
            hot.publish = lambda *a, **k: None
            hot.flip = lambda *a, **k: None
            hot.start_pipelined_decode()

            def verify(bs):
                out, state = verify_forward(model, mx.array([tokens[:bs]], dtype=mx.int32), cache,
                                            capture_layer_ids=wanted)
                hidden = mx.concatenate(out.hidden_states, axis=-1)
                mx.eval(out.logits, hidden)
                abort_speculative_round(state)
                hot.traces.clear()
                return out.logits.astype(mx.float32), hidden.astype(mx.float32)

            row = dict(context=len(ids))
            outputs = {}
            for name, flags in configs:
                with configured(backbone_fast, flags):
                    outputs[name] = {bs: verify(bs) for bs in blocks}
            ref_name = configs[0][0]
            for name, _ in configs[1:]:
                for bs in blocks:
                    ref_logits, ref_hidden = outputs[ref_name][bs]
                    got_logits, got_hidden = outputs[name][bs]
                    diff = mx.abs(ref_logits - got_logits)
                    row[f'{name}_b{bs}'] = dict(
                        logits_equal=bool(mx.array_equal(ref_logits, got_logits).item()),
                        max_abs=float(diff.max().item()), mean_abs=float(diff.mean().item()),
                        top1=float((ref_logits.argmax(-1) == got_logits.argmax(-1)).astype(mx.float32).mean().item()),
                        hidden_max_abs=float(mx.abs(ref_hidden - got_hidden).max().item()))
            timing = {name: {bs: [] for bs in blocks} for name, _ in configs}
            for r in range(args.repeat):
                for name, flags in (configs if r % 2 == 0 else configs[::-1]):
                    with configured(backbone_fast, flags):
                        for bs in blocks:
                            mx.synchronize()
                            start = time.perf_counter()
                            verify(bs)
                            timing[name][bs].append(time.perf_counter() - start)
            row['verify_ms'] = {name: {bs: round(1000 * statistics.median(v), 2) for bs, v in per.items()}
                                for name, per in timing.items()}
            print(json.dumps(row), flush=True)
            results.append(row)
            del cache
            mx.clear_cache()
    finally:
        hot.close()
        for bank in banks:
            bank.close()
        if nd.BATCHED is not None:
            nd.BATCHED.close()
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
