"""Humanity's Last Exam (text-only sample) through an OpenAI-compatible server.

Uses HLE's official answer-format system prompts and judge prompt (github.com/centerforaisafety/hle). Phase 1
generates answers, phase 2 judges them with --judge-url (the same server by default). Multiple-choice answers are
also graded by exact letter match, and exact-match answers by a strict normalised string match, as checks on the
judge. Results are written after every question; rerunning with the same --output resumes.
"""

import argparse
import json
import math
from pathlib import Path
import re
import time
import urllib.request

SYSTEM_EXACT = ('Your response should be in the following format:\nExplanation: {your explanation for your final answer}\n'
                'Exact Answer: {your succinct, final answer}\n'
                'Confidence: {your confidence score between 0% and 100% for your answer}')
SYSTEM_MC = ('Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\n'
             'Answer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}')
JUDGE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there are any inconsistencies, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


def chat(url, key, messages, max_tokens, model, effort=None):
    payload = dict(model=model, max_tokens=max_tokens, temperature=0, messages=messages)
    if effort:
        payload['reasoning_effort'] = effort
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, body, {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.loads(response.read())
    choice = reply['choices'][0]
    message = choice['message']
    text = message['content'] or ''
    if message.get('reasoning_content') and not text:
        text = message['reasoning_content']        # cut off while thinking: keep the reasoning for inspection
    return text, choice.get('finish_reason'), reply.get('usage', {}), reply.get('timings', {})


def final_answer(text):
    found = re.findall(r'(?:Exact Answer|Answer)\s*:\s*(.+)', text)
    return found[-1].strip() if found else None


def normalise(value):
    value = (value or '').strip().lower()
    value = re.sub(r'^\$+|\$+$', '', value)
    value = re.sub(r'\\(text|mathrm|textbf)\{([^}]*)\}', r'\2', value)
    value = re.sub(r'[\s\\$]', '', value).rstrip('.')
    return value


def mc_letter(value):
    match = re.match(r'\(?([A-Za-z])\)?(?:[.:)\s]|$)', (value or '').strip())
    return match.group(1).upper() if match else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    p.add_argument('--judge-url', default=None)
    p.add_argument('--api-key', default='local')
    p.add_argument('--model', default='glm5.3-flash')
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--reasoning-effort', default=None, help='low, high or max: answer with thinking on')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.questions.read_text())
    results = json.loads(a.output.read_text())['results'] if a.output.exists() else {}
    save = lambda summary=None: a.output.write_text(json.dumps(dict(summary=summary, results=results), indent=1))
    for index, row in enumerate(rows):
        entry = results.get(row['id'])
        if entry and 'response' in entry:
            continue
        system = SYSTEM_MC if row['answer_type'] == 'multipleChoice' else SYSTEM_EXACT
        started = time.perf_counter()
        text, finish, usage, timings = chat(a.url, a.api_key, [{'role': 'system', 'content': system},
                                                               {'role': 'user', 'content': row['question']}],
                                            a.max_tokens, a.model, a.reasoning_effort)
        results[row['id']] = dict(category=row['category'], answer_type=row['answer_type'], answer=row['answer'],
                                  response=text, finish=finish, seconds=round(time.perf_counter() - started, 1),
                                  completion_tokens=usage.get('completion_tokens'),
                                  decode_tok_s=timings.get('decode_tok_s'))
        save()
        print(json.dumps(dict(phase='answer', index=index, tokens=usage.get('completion_tokens'), finish=finish,
                              seconds=results[row['id']]['seconds'])), flush=True)
    judge_url = a.judge_url or a.url
    for index, row in enumerate(rows):
        entry = results[row['id']]
        if 'judge_correct' in entry:
            continue
        text, _, _, _ = chat(judge_url, a.api_key, [{'role': 'user', 'content': JUDGE.format(
            question=row['question'], response=entry['response'], correct_answer=row['answer'])}], 1024, a.model)
        verdict = re.findall(r'correct\s*:\s*\**\s*(yes|no)', text, flags=re.I)
        entry['judge'] = text
        entry['judge_correct'] = bool(verdict) and verdict[-1].lower() == 'yes'
        extracted = final_answer(entry['response'])
        if row['answer_type'] == 'multipleChoice':
            entry['strict_correct'] = mc_letter(extracted) is not None and mc_letter(extracted) == mc_letter(row['answer'])
        else:
            entry['strict_correct'] = extracted is not None and normalise(extracted) == normalise(row['answer'])
        save()
        print(json.dumps(dict(phase='judge', index=index, judge=entry['judge_correct'], strict=entry['strict_correct'])),
              flush=True)
    done = [results[row['id']] for row in rows]
    n, judged, strict = len(done), sum(e['judge_correct'] for e in done), sum(e['strict_correct'] for e in done)
    by_type = {}
    for e in done:
        t = by_type.setdefault(e['answer_type'], [0, 0])
        t[0] += e['judge_correct']
        t[1] += 1
    summary = dict(questions=n, judge_accuracy=judged / n, judge_ci95=wilson(judged, n), strict_accuracy=strict / n,
                   by_answer_type={k: f'{v[0]}/{v[1]}' for k, v in by_type.items()},
                   truncated=sum(e['finish'] == 'length' for e in done),
                   no_final_answer=sum(final_answer(e['response']) is None for e in done),
                   mean_completion_tokens=sum(e['completion_tokens'] or 0 for e in done) / n)
    save(summary)
    print(json.dumps(dict(event='summary', **summary)), flush=True)


if __name__ == '__main__':
    main()
