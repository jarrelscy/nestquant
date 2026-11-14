"""MMLU-Pro accuracy through the OpenAI-compatible server (zero-shot chain of thought, greedy)."""

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

LETTERS = 'ABCDEFGHIJ'


def prompt(row):
    options = '\n'.join(f'({LETTERS[i]}) {option}' for i, option in enumerate(row['options']))
    return (f"The following is a multiple choice question about {row['category']}. Think step by step and then "
            f'finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
            f"Question: {row['question']}\nOptions:\n{options}\nAnswer: Let's think step by step.")


def extract(text):
    found = re.findall(r'answer is \(?([A-J])\)?', text)
    if found:
        return found[-1]
    found = re.findall(r'\b([A-J])\)', text)
    return found[-1] if found else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    p.add_argument('--api-key', default='local')
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='keep results already in --output and skip those questions')
    a = p.parse_args()
    rows = json.loads(a.questions.read_text())
    results = []
    if a.resume and a.output.exists():
        saved = json.loads(a.output.read_text())
        results = saved['results'] if isinstance(saved, dict) else saved
    done_ids = {r['question_id'] for r in results}
    for index, row in enumerate(rows):
        if row['question_id'] in done_ids:
            continue
        body = json.dumps(dict(model='glm-nestquant', max_tokens=a.max_tokens, temperature=0,
                               messages=[{'role': 'user', 'content': prompt(row)}])).encode()
        request = urllib.request.Request(a.url, body, {'Content-Type': 'application/json',
                                                       'Authorization': f'Bearer {a.api_key}'})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=1800) as response:
            reply = json.loads(response.read())
        seconds = time.perf_counter() - started
        text = reply['choices'][0]['message']['content'] or ''
        predicted = extract(text)
        correct = predicted == row['answer']
        usage = reply.get('usage', {})
        timings = reply.get('timings', {})
        results.append(dict(question_id=row['question_id'], category=row['category'], answer=row['answer'],
                            predicted=predicted, correct=correct, seconds=seconds,
                            completion_tokens=usage.get('completion_tokens'), prompt_tokens=usage.get('prompt_tokens'),
                            decode_tok_s=timings.get('decode_tok_s') or timings.get('predicted_per_second'),
                            finish=reply['choices'][0].get('finish_reason'), completion=text))
        done = sum(r['correct'] for r in results)
        print(json.dumps(dict(index=index, correct=correct, predicted=predicted, answer=row['answer'],
                              accuracy=f'{done}/{len(results)}', tokens=usage.get('completion_tokens'),
                              seconds=round(seconds, 1))), flush=True)
        a.output.write_text(json.dumps(results, indent=1))
    tok = [r['completion_tokens'] / r['seconds'] for r in results if r['completion_tokens']]
    summary = dict(questions=len(results), accuracy=sum(r['correct'] for r in results) / len(results),
                   no_answer=sum(r['predicted'] is None for r in results),
                   truncated=sum(r['finish'] == 'length' for r in results),
                   mean_completion_tokens=sum(r['completion_tokens'] or 0 for r in results) / len(results),
                   mean_end_to_end_tok_s=sum(tok) / max(len(tok), 1))
    print(json.dumps(dict(event='summary', **summary)), flush=True)
    a.output.write_text(json.dumps(dict(summary=summary, results=results), indent=1))


if __name__ == '__main__':
    main()
