"""Multi-turn prefix-reuse check against a running dflash_server.py (terminus-2-like conversation).

Turns 1-3 append the assistant reply and a new user message with ~2K tokens of fake terminal output, so turns 2-3
should extend the cached prefix. The final prompt is then sent again unchanged, which forces a fresh prefill (the
cache can only be extended), and the replies are compared. A last request truncates the previous reply to exercise
the checkpoint rollback path.
"""

import argparse
import json
import random
import time
import urllib.request

SYSTEM = ("You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a "
          "task description and the output from previously executed commands. Your goal is to solve the task by "
          "providing batches of shell commands.\n\nFormat your response as JSON with the following structure:\n"
          '{\n  "analysis": "Analyze the current state based on the terminal output provided.",\n'
          '  "plan": "Describe your plan for the next steps.",\n'
          '  "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}],\n  "task_complete": false\n}\n'
          "Respond with JSON only.")

TASK = ("Task: The Python project in /app fails its test suite after a dependency upgrade. Find the failing test, "
        "fix the code (not the tests) and make `pytest -q` pass.\n\nCurrent terminal state:\n")


def terminal_output(seed, lines=80):
    rng = random.Random(seed)
    words = ['config', 'utils', 'parser', 'models', 'views', 'tests', 'cache', 'client', 'server', 'schema']
    out = [f'root@a1b2c3:/app# {rng.choice(["ls -la src", "pip install -e .", "pytest -q", "find . -name *.py"])}']
    for i in range(lines):
        kind = rng.random()
        if kind < 0.35:
            name = f'{rng.choice(words)}_{rng.choice(words)}.py'
            out.append(f'-rw-r--r-- 1 root root {rng.randint(200, 40000):6d} Nov {rng.randint(1, 14):2d} '
                       f'{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} src/{name}')
        elif kind < 0.65:
            pkg = rng.choice(['numpy', 'pydantic', 'requests', 'urllib3', 'attrs', 'pluggy', 'idna', 'certifi'])
            out.append(f'Requirement already satisfied: {pkg}>={rng.randint(1, 3)}.{rng.randint(0, 20)} in '
                       f'/usr/local/lib/python3.12/site-packages (from app==0.{seed}.0) ({rng.randint(1, 3)}.'
                       f'{rng.randint(0, 30)}.{rng.randint(0, 9)})')
        elif kind < 0.9:
            out.append(f'tests/test_{rng.choice(words)}.py::test_{rng.choice(words)}_{rng.choice(words)} '
                       f'{"PASSED" if rng.random() < 0.9 else "FAILED"} [{min(100, i // 2):3d}%]')
        else:
            out.append(f'E   AssertionError: expected {rng.randint(0, 99)} == {rng.randint(0, 99)} in '
                       f'src/{rng.choice(words)}.py:{rng.randint(10, 400)}')
    out.append('root@a1b2c3:/app# ')
    return '\n'.join(out)


def post(url, body, timeout=3600):
    request = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        out = json.loads(response.read())
    out['client_seconds'] = round(time.perf_counter() - started, 2)
    return out


def summary(label, out):
    t = out['timings']
    row = dict(turn=label, mode=t.get('mode'), prompt_tokens=out['usage']['prompt_tokens'],
               reused=t.get('reused_tokens'), prefilled=t.get('prefilled_tokens'), prefill_s=t.get('prefill_seconds'),
               prefill_tok_s=t.get('prefill_tok_s'), completion_tokens=out['usage']['completion_tokens'],
               decode_tok_s=t.get('decode_tok_s'), accepted_per_round=t.get('mean_accepted'),
               finish=out['choices'][0]['finish_reason'], first_top5=t.get('first_top5'), peak_gib=t.get('peak_gib'), client_s=out['client_seconds'])
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18090/v1/chat/completions')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--turns', type=int, default=3)
    parser.add_argument('--output')
    args = parser.parse_args()
    messages = [{'role': 'system', 'content': SYSTEM}]
    rows, replies = [], []
    for turn in range(1, args.turns + 1):
        content = (TASK if turn == 1 else 'New terminal output:\n') + terminal_output(turn)
        messages.append({'role': 'user', 'content': content})
        out = post(args.url, dict(model='glm-nestquant', messages=messages, max_tokens=args.max_tokens,
                                  temperature=0.7))
        rows.append(summary(f'turn{turn}', out))
        reply = out['choices'][0]['message']['content']
        replies.append(reply)
        print(json.dumps(dict(turn=turn, reply_head=reply[:400])), flush=True)
        if turn < args.turns:
            messages.append({'role': 'assistant', 'content': reply})
    # Same final prompt again: cannot extend (needs one new token) and the checkpoint sits at its end -> fresh.
    fresh = post(args.url, dict(model='glm-nestquant', messages=messages, max_tokens=args.max_tokens))
    rows.append(summary('final_prompt_fresh', fresh))
    fresh_reply = fresh['choices'][0]['message']['content']
    same = next((i for i, (a, b) in enumerate(zip(replies[-1], fresh_reply)) if a != b),
                min(len(replies[-1]), len(fresh_reply)))
    print(json.dumps(dict(compare='reused_vs_fresh', matching_prefix_chars=same, reused_chars=len(replies[-1]),
                          fresh_chars=len(fresh_reply), identical=replies[-1] == fresh_reply,
                          fresh_head=fresh_reply[:200])), flush=True)
    # Truncated previous reply -> the prompt diverges inside the cached reply -> rollback to the checkpoint.
    messages_rb = messages + [{'role': 'assistant', 'content': fresh_reply[:len(fresh_reply) // 2]},
                              {'role': 'user', 'content': 'Your previous reply was cut off. Reply again, briefly.'}]
    rollback = post(args.url, dict(model='glm-nestquant', messages=messages_rb, max_tokens=64))
    rows.append(summary('truncated_reply_rollback', rollback))
    print(json.dumps(dict(rollback_reply_head=rollback['choices'][0]['message']['content'][:200])), flush=True)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(dict(rows=rows, replies=replies, fresh_reply=fresh_reply), f, indent=1)


if __name__ == '__main__':
    main()
