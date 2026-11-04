import argparse
import json
from pathlib import Path

import mlx.core as mx
import torch

from generate import Runtime


CASES = (
    ('exact_reply', 'Reply with exactly GLM_MAC_OK and nothing else.', 'GLM_MAC_OK'),
    ('arithmetic', 'What is 19 + 23? Reply with only the integer answer.', '42'),
    ('ordering', 'Sort these integers in ascending order: 3, 1, 2. Reply with only a JSON array.', [1, 2, 3]),
)


def answer_matches(text, expected):
    answer = text.strip()
    if not isinstance(expected, list):
        return answer == expected
    try:
        answer = json.loads(answer)
    except json.JSONDecodeError:
        return False
    return isinstance(answer, list) and all(type(value) is int for value in answer) and answer == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hot-fraction', type=float, default=1.0)
    parser.add_argument('--cache-gib', type=float, default=4)
    parser.add_argument('--resident-base', action='store_true')
    parser.add_argument('--resident-hot', action='store_true')
    parser.add_argument('--wired-memory', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file to preserve earlier results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    mx.set_cache_limit(1024**3)
    report = dict(status='running', wired_memory=args.wired_memory, hot_fraction=args.hot_fraction, resident_base=args.resident_base, resident_hot=args.resident_hot,
                  scope='Three short text smoke checks, not a general quality evaluation', checks=[])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    runtime = None
    try:
        runtime = Runtime(args.model, args.cache_gib, hot_fraction=args.hot_fraction,
                          resident_base=args.resident_base, resident_hot=args.resident_hot, wired_memory=args.wired_memory)
        for name, prompt, expected in CASES:
            result = runtime.generate(prompt, max_tokens=32)
            report['checks'].append(dict(name=name, passed=answer_matches(result['text'], expected),
                                        expected=expected, generation=result))
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        report['status'] = 'passed' if all(check['passed'] for check in report['checks']) else 'failed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if runtime is not None:
            runtime.close()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], checks=[dict(name=c['name'], passed=c['passed']) for c in report['checks']])), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
