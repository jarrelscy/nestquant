"""Bitwise comparison of target verify outputs (logits and captured drafter hidden states) under the decode-path
options, on the full runtime with a frozen 4-bit hot set. Each configuration verifies the same blocks from the
same cache state (rolled back after every call). Also compares single-token decode logits.
Run inside the GPU lock."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

import mlx.core as mx
import torch

HOME = Path.home()

CONFIGS = {
    'base': dict(FUSED_MLP=False, WEIGHT_ONCE=False, SPLIT_PRECISION=False),
    'fused': dict(FUSED_MLP=True, WEIGHT_ONCE=False, SPLIT_PRECISION=False),
    'weight_once': dict(FUSED_MLP=False, WEIGHT_ONCE=True, SPLIT_PRECISION=False),
    'split': dict(FUSED_MLP=False, WEIGHT_ONCE=False, SPLIT_PRECISION=True),
    'all': dict(FUSED_MLP=True, WEIGHT_ONCE=True, SPLIT_PRECISION=True),
}


def configured(ng, backbone_fast, decoder, settings):
    stack = ExitStack()
    stack.enter_context(patch.object(ng, 'FUSED_MLP', settings['FUSED_MLP']))
    stack.enter_context(patch.object(backbone_fast, 'WEIGHT_ONCE', settings['WEIGHT_ONCE']))
    stack.enter_context(patch.object(decoder, 'SPLIT_PRECISION', settings['SPLIT_PRECISION']))
    return stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    parser.add_argument('--archive', type=Path, default=HOME / 'Models/GLM/resident-banks-v1')
    parser.add_argument('--backbone', type=Path, default=HOME / 'Models/GLM/backbone-affine4-e8-v1')
    parser.add_argument('--drafter', type=Path, default=HOME / 'Models/GLM/drafters/dflash2')
    parser.add_argument('--context', type=int, default=3000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import backbone_fast
    import decoder
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
    result = {}
    try:
        hot.warm(4, lambda: guard.check(force=True))
        nd.BATCHED = BatchedPrefill(args.model)
        drafter, _ = load_drafter(str(args.drafter))
        wanted = list(drafter.config.target_layer_ids)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        files = sorted(Path(pdf.__file__).resolve().parent.glob('*.py'))
        ids = pdf.context_ids(tokenizer, template, files, args.context)
        cache = model.make_cache()
        hot.reset()
        logits, _ = nd.capture_prefill(model, drafter, ids, cache, hot, guard)
        token = ng.greedy(logits)
        # Some generated tokens to verify (plain decode), then freeze the hot set.
        _, tokens, _, _ = pdf.plain_decode(ng, model, cache, hot, guard, token, 24, None)
        hot.drain()
        hot.publish = lambda *a, **k: None
        hot.flip = lambda *a, **k: None
        hot.start_pipelined_decode()
        lm = model
        outputs = {}
        for name, settings in CONFIGS.items():
            with configured(ng, backbone_fast, decoder, settings):
                per = {}
                for bs in (1, 2, 4, 8):
                    inputs = mx.array([tokens[:bs]], dtype=mx.int32)
                    out, state = verify_forward(lm, inputs, cache, capture_layer_ids=wanted)
                    hidden = mx.concatenate(out.hidden_states, axis=-1)
                    mx.eval(out.logits, hidden)
                    per[bs] = (out.logits.astype(mx.float32), hidden.astype(mx.float32))
                    abort_speculative_round(state)
                    hot.traces.clear()
                outputs[name] = per
        for name in CONFIGS:
            result[name] = {}
            for bs in (1, 2, 4, 8):
                ref_logits, ref_hidden = outputs['base'][bs]
                got_logits, got_hidden = outputs[name][bs]
                result[name][bs] = dict(
                    logits_equal=bool(mx.array_equal(ref_logits, got_logits).item()),
                    hidden_equal=bool(mx.array_equal(ref_hidden, got_hidden).item()),
                    max_logit_diff=float(mx.max(mx.abs(ref_logits - got_logits)).item()),
                    argmax_equal=bool(mx.array_equal(ref_logits.argmax(-1), got_logits.argmax(-1)).item()))
            print(json.dumps({name: result[name]}), flush=True)
        # Verify rows against single-token decode of the same tokens (row t of a block vs t-th decode step).
        with configured(ng, backbone_fast, decoder, CONFIGS['all']):
            singles = []
            states = []
            for t in range(8):
                out, state = verify_forward(lm, mx.array([[tokens[t]]], dtype=mx.int32), cache, capture_layer_ids=wanted)
                mx.eval(out.logits)
                singles.append(out.logits.astype(mx.float32))
                state.commit([1])
                hot.traces.clear()
        block = outputs['all'][8][0]
        stacked = mx.concatenate(singles, axis=1)
        result['block8_vs_decode'] = dict(max_logit_diff=float(mx.max(mx.abs(block - stacked)).item()),
                                          argmax_equal=bool(mx.array_equal(block.argmax(-1), stacked.argmax(-1)).item()))
        print(json.dumps(result['block8_vs_decode']), flush=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()
        if nd.BATCHED is not None:
            nd.BATCHED.close()
    args.output.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
