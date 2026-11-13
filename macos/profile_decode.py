"""Profile native decode with the full resident base and Spark refinements.

Loads the runtime once, prefills a prompt, then times decode steps in parts: predictor before the
forward, forward plus GPU evaluation, and the commit. It then repeats the forward with routed
experts at 1.5 bits only and with routed experts removed, which isolates the backbone.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx

import native_generate as ng
from metrics import MemoryGuard
from native_refinements import NativeHotRefinements


def median_ms(values):
    return 1000 * statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--hot-fraction', type=float, default=0.2174)
    parser.add_argument('--budget-gib', type=int, default=22)
    parser.add_argument('--decoder-loop', default='window')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(92 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, 4, 32, 512)
    result = {}
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56,
                                   decoder_loop=args.decoder_loop)
    try:
        result['warm'] = hot.warm(4, lambda: guard.check(force=True))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explain how a hash map works, in detail.'}],
                                            tokenize=True, return_dict=False, add_generation_prompt=True,
                                            chat_template=template)
        cache = model.make_cache()
        hot.reset()
        chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
        started = time.perf_counter()
        logits = ng.layer_major_prefill(model, chunks, cache, guard, hot)
        result['prefill_seconds'] = time.perf_counter() - started
        result['prompt_tokens'] = len(ids)
        token = ng.greedy(logits)
        parts = dict(begin=[], forward=[], commit=[], sample=[], total=[])
        for _ in range(args.steps):
            t0 = time.perf_counter()
            hot.begin_decode(token)
            t1 = time.perf_counter()
            logits = model(mx.array([[token]]), cache=cache).logits
            mx.eval(logits, [entry.state for entry in cache])
            t2 = time.perf_counter()
            hot.commit([token])
            t3 = time.perf_counter()
            token = ng.greedy(logits)
            t4 = time.perf_counter()
            for name, value in zip(parts, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                parts[name].append(value)
            guard.check()
        result['mixed_step_ms'] = {name: median_ms(values[8:]) for name, values in parts.items()}
        result['refinement_publication_s'] = hot.reads['decode_publication_seconds']

        # Same cache position each time; only timing matters for these variants.
        def forward_only(steps=24):
            times = []
            for _ in range(steps):
                t0 = time.perf_counter()
                out = model(mx.array([[token]]), cache=cache).logits
                mx.eval(out, [entry.state for entry in cache])
                times.append(time.perf_counter() - t0)
            return median_ms(times[4:])

        hot.phase = 'prefill_disabled'
        layers = [layer.mlp for layer in model.layers if isinstance(layer.mlp, ng.PagedExperts)]
        saved = [mlp._hot for mlp in layers]
        for mlp in layers:
            mlp._hot = None
        result['base_only_forward_ms'] = forward_only()
        original = ng.PagedExperts.__call__

        def shared_only(self, x):
            self.gate(x)
            return self.shared_experts(x)

        ng.PagedExperts.__call__ = shared_only
        result['no_routed_experts_forward_ms'] = forward_only()
        ng.PagedExperts.__call__ = original
        for mlp, value in zip(layers, saved):
            mlp._hot = value
        result['memory_gib'] = ng.process_usage().lifetime_max_phys_footprint / 2**30
    finally:
        hot.close()
        for bank in banks:
            bank.close()
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
