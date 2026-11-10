"""Token loading and row alignment for teacher-forced KL divergence scoring."""

import json

import numpy as np


def load_tokens(path):
    if path.suffix == '.npy':
        return [int(token) for token in np.load(path).reshape(-1)], None
    record = json.loads(path.read_text())
    tokens, prompt = record['token_ids'], record['prompt_ids']
    if not prompt or tokens[:len(prompt)] != prompt or len(tokens) <= len(prompt):
        raise ValueError('A trace must extend its nonempty prompt tokens')
    return [int(token) for token in tokens], len(prompt)


def decode_reference_rows(reference, prompt_length, steps):
    if prompt_length < 1 or steps < 1 or prompt_length + steps > len(reference):
        raise ValueError('Decode steps must follow a nonempty prompt and stay inside the reference')
    return reference[prompt_length:prompt_length + steps]


def teacher_rows(prefill, teacher_positions):
    # A teacher captured on an n-token window scores n - 1 next-token predictions.
    if teacher_positions != len(prefill) - 1:
        raise ValueError('Teacher positions must equal the window length minus one')
    return prefill[:teacher_positions]
