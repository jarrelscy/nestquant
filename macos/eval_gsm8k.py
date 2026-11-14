"""GSM8K accuracy and decode speed with DFlash2 speculative decoding (full 4-bit verify by default)."""

import argparse
import json
from pathlib import Path
import re
import time

import mlx.core as mx
import torch

import native_dflash as nd
import native_generate as ng
from metrics import MemoryGuard
from native_refinements import NativeHotRefinements

INSTRUCTION = ('Solve the following math problem step by step. Keep the reasoning short. '
               'End your reply with a final line of the form "The answer is N", where N is a number.\n\n')


def reference_answer(answer):
    return _number(answer.split('####')[-1])


def _number(text):
    text = text.replace(',', '').replace('$', '').strip()
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(match.group()) if match else None


def predicted_answer(text):
    found = re.findall(r'answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)', text, re.I)
    if found:
        return _number(found[-1])
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return _number(numbers[-1]) if numbers else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--drafter', type=Path, required=True)
    parser.add_argument('--problems', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--hot-fraction', type=float, default=0.2)
    parser.add_argument('--budget-gib', type=int, default=22)
    parser.add_argument('--mode', choices=('dflash', 'dflash_base', 'plain'), default='dflash')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    nd._install_capture()
    nd._relax_compatibility()
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(96 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, 4, 32, 512)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    lines = args.problems.read_text().splitlines()[args.start:args.start + args.limit]
    problems = [json.loads(line) for line in lines]
    rows = []
    try:
        hot.warm(4, lambda: guard.check(force=True))
        hot.draft_rows_base = args.mode == 'dflash_base'
        from native_prefill import BatchedPrefill
        nd.BATCHED = BatchedPrefill(args.model)
        drafter = None
        if args.mode != 'plain':
            from mlx_vlm.speculative.drafters import load_drafter
            drafter, _ = load_drafter(str(args.drafter))
            mx.eval(drafter.parameters())
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        end = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
        eos = {end} if isinstance(end, int) else set(end)
        for index, problem in enumerate(problems):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': INSTRUCTION + problem['question']}],
                                                tokenize=True, return_dict=False, add_generation_prompt=True,
                                                chat_template=template)
            started = time.perf_counter()
            if args.mode == 'plain':
                out = ng.generate(model, tokenizer, ids, args.max_tokens, eos, nd.BATCHED, 8, hot, guard)
                stats = dict(decode_tps=out['decode_tps'], ttft=out['ttft_seconds'])
            else:
                out = nd.block_generate(model, drafter, ids, args.max_tokens, eos, hot, guard, -8)
                stats = dict(decode_tps=out['decode_tps'], ttft=out['ttft_seconds'],
                             accepted=out['mean_accepted'], drafted=out['mean_drafted'])
            generated = out['generated_ids']
            text = tokenizer.decode([t for t in generated if t not in eos], skip_special_tokens=True)
            expected, predicted = reference_answer(problem['answer']), predicted_answer(text)
            correct = predicted is not None and expected is not None and abs(predicted - expected) < 1e-6
            rows.append(dict(index=args.start + index, correct=correct, expected=expected, predicted=predicted,
                             tokens=len(generated), seconds=time.perf_counter() - started, completion=text, **stats))
            guard.check(force=True)
            tps = [r['decode_tps'] for r in rows if r['decode_tps']]
            print(json.dumps(dict(event='problem', index=args.start + index, correct=correct,
                                  accuracy=f"{sum(r['correct'] for r in rows)}/{len(rows)}",
                                  decode_tps=round(stats['decode_tps'] or 0, 1),
                                  mean_tps=round(sum(tps) / max(len(tps), 1), 1))), flush=True)
            args.output.write_text(json.dumps(rows, indent=1))
    finally:
        hot.close()
        for bank in banks:
            bank.close()
    tps = [r['decode_tps'] for r in rows if r['decode_tps']]
    summary = dict(mode=args.mode, problems=len(rows), accuracy=sum(r['correct'] for r in rows) / max(len(rows), 1),
                   mean_decode_tps=sum(tps) / max(len(tps), 1),
                   mean_accepted=sum(r.get('accepted', 0) for r in rows) / max(len(rows), 1),
                   mean_tokens=sum(r['tokens'] for r in rows) / max(len(rows), 1),
                   mean_ttft=sum(r['ttft'] for r in rows) / max(len(rows), 1))
    print(json.dumps(dict(event='summary', **summary)), flush=True)
    args.output.write_text(json.dumps(dict(summary=summary, results=rows), indent=1))


if __name__ == '__main__':
    main()
