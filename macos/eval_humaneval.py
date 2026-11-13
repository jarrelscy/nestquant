"""HumanEval pass@1 for plain decode and DFlash2 speculative decoding, graded in a local sandbox.

Each completion is executed with the problem's tests under sandbox-exec (no network, writes only in a
temporary directory) with a timeout, using the system Python in isolated mode.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import mlx.core as mx
import torch

import native_dflash as nd
import native_generate as ng
from metrics import MemoryGuard
from native_refinements import NativeHotRefinements

SANDBOX = '''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null") (literal "/dev/tty"))
'''


def extract_code(text, entry_point):
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    candidates = [block for block in blocks if f'def {entry_point}' in block] or blocks
    return candidates[0] if candidates else text


def grade(problem, completion, timeout=15):
    code = extract_code(completion, problem['entry_point'])
    header = '\n'.join(line for line in problem['prompt'].splitlines() if line.startswith(('import ', 'from ')))
    program = f"{header}\n{code}\n\n{problem['test']}\ncheck({problem['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix='humaneval-') as directory:
        path = Path(directory) / 'solution.py'
        path.write_text(program)
        profile = SANDBOX.format(tmp=str(Path(directory).resolve()))
        try:
            result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, '/usr/bin/python3', '-I', str(path)],
                                    cwd=directory, capture_output=True, timeout=timeout, text=True)
            return result.returncode == 0, result.stderr[-300:]
        except subprocess.TimeoutExpired:
            return False, 'timeout'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--drafter', type=Path, required=True)
    parser.add_argument('--problems', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--hot-fraction', type=float, default=0.15)
    parser.add_argument('--budget-gib', type=int, default=16)
    parser.add_argument('--modes', default='plain,dflash,dflash_base')
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
    problems = [json.loads(line) for line in args.problems.read_text().splitlines()][:args.limit]
    modes = args.modes.split(',')
    results = {mode: [] for mode in modes}
    try:
        hot.warm(4, lambda: guard.check(force=True))
        from native_prefill import BatchedPrefill
        nd.BATCHED = BatchedPrefill(args.model)
        from mlx_vlm.speculative.drafters import load_drafter
        drafter, _ = load_drafter(str(args.drafter))
        mx.eval(drafter.parameters())
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        end = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
        eos = {end} if isinstance(end, int) else set(end)
        for index, problem in enumerate(problems):
            request = ('Complete the following Python function. Reply with the complete function, including its '
                       'imports, in a single Python code block and no explanation.\n\n```python\n'
                       + problem['prompt'] + '```')
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': request}], tokenize=True,
                                                return_dict=False, add_generation_prompt=True, chat_template=template)
            for mode in modes:
                hot.draft_rows_base = mode == 'dflash_base'
                started = time.perf_counter()
                if mode == 'plain':
                    out = ng.generate(model, tokenizer, ids, args.max_tokens, eos, nd.BATCHED, 8, hot, guard)
                    generated = out['generated_ids']
                    stats = dict(decode_tps=out['decode_tps'], ttft=out['ttft_seconds'])
                else:
                    out = nd.block_generate(model, drafter, ids, args.max_tokens, eos, hot, guard, -8)
                    generated = out['generated_ids']
                    stats = dict(decode_tps=out['decode_tps'], ttft=out['ttft_seconds'],
                                 accepted=out['mean_accepted'], drafted=out['mean_drafted'])
                text = tokenizer.decode([t for t in generated if t not in eos], skip_special_tokens=True)
                passed, error = grade(problem, text)
                results[mode].append(dict(task_id=problem['task_id'], passed=passed, tokens=len(generated),
                                          seconds=time.perf_counter() - started, error=None if passed else error,
                                          completion=text, **stats))
                guard.check(force=True)
            summary = {mode: f"{sum(r['passed'] for r in rows)}/{len(rows)}" for mode, rows in results.items()}
            print(json.dumps(dict(event='problem', index=index, task_id=problem['task_id'], **summary,
                                  last={m: round(results[m][-1]['decode_tps'] or 0, 1) for m in modes})), flush=True)
            args.output.write_text(json.dumps(results, indent=1))
    finally:
        hot.close()
        for bank in banks:
            bank.close()
    report = {}
    for mode, rows in results.items():
        tps = [r['decode_tps'] for r in rows if r['decode_tps']]
        report[mode] = dict(pass_at_1=sum(r['passed'] for r in rows) / max(len(rows), 1), problems=len(rows),
                            mean_decode_tps=sum(tps) / max(len(tps), 1),
                            mean_tokens=sum(r['tokens'] for r in rows) / max(len(rows), 1))
    print(json.dumps(dict(event='summary', **report)), flush=True)
    args.output.write_text(json.dumps(dict(summary=report, results=results), indent=1))


if __name__ == '__main__':
    main()
