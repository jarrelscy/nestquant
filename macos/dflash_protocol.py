"""Model-free pieces of the DFlash2 OpenAI-compatible server: request parsing, prefix-reuse planning, stop strings
and response/SSE formatting. Kept free of MLX so it can be unit tested on the CPU."""

import json
import time
from uuid import uuid4

MODEL = 'glm-nestquant'
DEFAULT_MAX_TOKENS = 4096


class RequestError(ValueError):
    def __init__(self, message, status=400, code=None, kind='invalid_request_error'):
        super().__init__(message)
        self.status, self.code, self.kind = status, code, kind

    def payload(self):
        return {'error': {'message': str(self), 'type': self.kind, 'param': None, 'code': self.code}}


def context_error(prompt_tokens, limit):
    # Phrased like OpenAI's error so LiteLLM maps it to ContextWindowExceededError (terminus-2 then summarizes).
    return RequestError(f"This model's maximum context length is {limit} tokens. However, your messages resulted "
                        f"in {prompt_tokens} tokens. Please reduce the length of the messages.",
                        code='context_length_exceeded')


def _text_content(content, role):
    if content is None:
        if role in ('assistant', 'tool'):
            return ''
        raise RequestError(f'Message content is required for role {role}')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get('type') in ('text', 'input_text') and isinstance(part.get('text'), str):
                parts.append(part['text'])
            elif isinstance(part, dict) and part.get('type') in ('image_url', 'image', 'input_audio', 'video_url'):
                raise RequestError('Only text content is supported')
            # Anything else (e.g. cache_control-only parts) is ignored.
        return ''.join(parts)
    raise RequestError('Unsupported message content')


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise RequestError('Expected a non-empty messages list')
    out = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError('Each message must be an object')
        role = message.get('role')
        if role == 'developer':
            role = 'system'
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise RequestError(f'Unsupported chat message role: {role!r}')
        entry = {'role': role, 'content': _text_content(message.get('content'), role)}
        if role == 'assistant':
            if isinstance(message.get('reasoning_content'), str):
                entry['reasoning_content'] = message['reasoning_content']
            calls = message.get('tool_calls') or []
            if not isinstance(calls, list):
                raise RequestError('Invalid assistant tool calls')
            rendered = []
            for call in calls:
                function = call.get('function') if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get('name'), str):
                    raise RequestError('Invalid tool function')
                arguments = function.get('arguments') or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError as error:
                        raise RequestError('Tool call arguments must be JSON') from error
                if not isinstance(arguments, dict):
                    raise RequestError('Expected JSON object tool arguments')
                rendered.append({'id': call.get('id'), 'type': 'function',
                                 'function': {'name': function['name'], 'arguments': arguments}})
            if rendered:
                entry['tool_calls'] = rendered
        if role == 'tool' and message.get('tool_call_id') is not None:
            entry['tool_call_id'] = message['tool_call_id']
        out.append(entry)
    return out


def parse_request(body):
    """Validate an OpenAI chat-completions body. Unknown fields are ignored (LiteLLM adds its own)."""
    if not isinstance(body, dict):
        raise RequestError('Expected a JSON object')
    messages = normalize_messages(body.get('messages'))
    n = body.get('n', 1)
    if n not in (None, 1):
        raise RequestError('Only n=1 is supported')
    limit = body.get('max_completion_tokens')
    if limit is None:
        limit = body.get('max_tokens')
    if limit is None:
        limit = DEFAULT_MAX_TOKENS
    if type(limit) is not int or limit < 1:
        raise RequestError('max_tokens must be a positive integer')
    stop = body.get('stop')
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop) or len(stop) > 16:
        raise RequestError('stop must be a string or a list of up to 16 strings')
    stop = [s for s in stop if s]
    tools = body.get('tools') or []
    if not isinstance(tools, list) or any(
            not isinstance(tool, dict) or tool.get('type', 'function') != 'function'
            or not isinstance(tool.get('function'), dict) or not isinstance(tool['function'].get('name'), str)
            for tool in tools):
        raise RequestError('Expected function tool schemas')
    if body.get('tool_choice') == 'none':
        tools = []
    temperature = body.get('temperature')
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise RequestError('temperature must be a number')
    stream = bool(body.get('stream', False))
    options = body.get('stream_options') or {}
    return dict(messages=messages, max_tokens=limit, stop=stop, tools=tools, stream=stream,
                include_usage=bool(isinstance(options, dict) and options.get('include_usage')),
                model=body.get('model') or MODEL, temperature=temperature)


# ---------------------------------------------------------------------------------------------- prefix reuse

def common_prefix_length(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def plan_reuse(cached_ids, checkpoint_length, new_ids):
    """Decide how to prefill `new_ids` given the tokens the live cache holds.

    The cache can only be extended (append-only recurrent state) or restored to the checkpoint taken at the end of
    the previous prompt. At least one new token must be prefilled to produce logits.

    Returns (mode, reused_tokens) with mode 'extend', 'rollback' or 'fresh'.
    """
    if not cached_ids or not new_ids:
        return 'fresh', 0
    usable = min(common_prefix_length(cached_ids, new_ids), len(new_ids) - 1)
    if usable == len(cached_ids):
        return 'extend', usable
    if checkpoint_length and checkpoint_length <= usable and checkpoint_length <= len(cached_ids):
        return 'rollback', checkpoint_length
    return 'fresh', 0


# ---------------------------------------------------------------------------------------------- stop strings

def find_stop(text, stops):
    """Index of the earliest stop-string occurrence in text, or None."""
    found = [i for i in (text.find(s) for s in stops) if i >= 0]
    return min(found) if found else None


def safe_prefix_length(text, stops):
    """Length of the prefix of `text` that can be emitted while streaming: it ends before any partial stop string
    at the end of `text` and before an incomplete UTF-8 replacement character."""
    end = len(text)
    while end > 0 and text[end - 1] == '�':
        end -= 1
    hold = 0
    for stop in stops:
        for k in range(min(len(stop) - 1, end), 0, -1):
            if text.endswith(stop[:k], 0, end):
                hold = max(hold, k)
                break
    return end - hold


# ---------------------------------------------------------------------------------------------- responses

def completion_id():
    return 'chatcmpl-' + uuid4().hex


def completion_response(model, message, finish_reason, prompt_tokens, completion_tokens, cached_tokens, timings,
                        cid=None):
    return dict(id=cid or completion_id(), object='chat.completion', created=int(time.time()), model=model,
                choices=[dict(index=0, message=message, finish_reason=finish_reason, logprobs=None)],
                usage=dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                           total_tokens=prompt_tokens + completion_tokens,
                           prompt_tokens_details=dict(cached_tokens=cached_tokens)),
                timings=timings)


def chunk(cid, model, delta, finish_reason=None, usage=None):
    payload = dict(id=cid, object='chat.completion.chunk', created=int(time.time()), model=model,
                   choices=[dict(index=0, delta=delta, finish_reason=finish_reason, logprobs=None)])
    if usage is not None:
        payload['usage'] = usage
    return payload


def sse(payload):
    return b'data: ' + json.dumps(payload, allow_nan=False).encode() + b'\n\n'


SSE_DONE = b'data: [DONE]\n\n'
