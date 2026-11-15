"""Anthropic Messages API on top of the OpenAI-style request handling (for Claude Code and other Anthropic clients).

Requests are converted to the OpenAI chat format that dflash_protocol.parse_request validates; responses are converted
back to Anthropic messages. Tool use maps to the model's native tool calls (the server's GLM tool parser).
"""

import json
from uuid import uuid4

from dflash_protocol import RequestError, parse_request


def _text_of(content):
    """Text of a tool_result or system content (string or list of blocks); images become a marker."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                parts.append(block.get('text') or '')
            elif isinstance(block, dict) and block.get('type') == 'image':
                parts.append('[image]')
        return '\n'.join(parts)
    raise RequestError('Unsupported content')


# Claude Code sends thinking={"type": "adaptive"} on every request. Honouring it means every agent turn reasons at
# full effort (often 30k+ tokens, ~20 minutes here), so Anthropic-style thinking is ignored unless the server enables
# it (--anthropic-thinking). OpenAI-style reasoning_effort is always honoured.
ANTHROPIC_THINKING = False


def strip_volatile(text):
    """Drop lines that change on every request at the start of a prompt. Claude Code begins its system prompt with
    `x-anthropic-billing-header: ...; cch=<per-request value>;`, which would defeat prefix reuse at token 0."""
    if 'x-anthropic-billing-header' not in text:
        return text
    return '\n'.join(line for line in text.split('\n')
                     if not line.lstrip().startswith('x-anthropic-billing-header')).lstrip('\n')


def _image_url(block):
    source = block.get('source') or {}
    if source.get('type') == 'base64':
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    if source.get('type') == 'url':
        return source.get('url')
    raise RequestError('Unsupported image source')


def to_openai(body):
    """Anthropic Messages request -> OpenAI chat-completions request body."""
    if not isinstance(body, dict):
        raise RequestError('Expected a JSON object')
    messages = []
    system = strip_volatile(_text_of(body.get('system')))
    if system:
        messages.append({'role': 'system', 'content': system})
    for message in body.get('messages') or []:
        if not isinstance(message, dict):
            raise RequestError('Each message must be an object')
        role, content = message.get('role'), message.get('content')
        if role == 'system':      # Claude Code also sends system messages inside the conversation
            text = strip_volatile(_text_of(content))
            if text:
                messages.append({'role': 'system', 'content': text})
            continue
        if role not in ('user', 'assistant'):
            raise RequestError(f'Unsupported message role: {role!r}')
        if isinstance(content, str) or content is None:
            messages.append({'role': role, 'content': content or ''})
            continue
        if not isinstance(content, list):
            raise RequestError('Message content must be a string or a list of blocks')
        if role == 'assistant':
            text, calls = [], []
            for block in content:
                kind = block.get('type') if isinstance(block, dict) else None
                if kind == 'text':
                    text.append(block.get('text') or '')
                elif kind == 'tool_use':
                    calls.append({'id': block.get('id'), 'type': 'function',
                                  'function': {'name': block.get('name'), 'arguments': block.get('input') or {}}})
                # thinking and redacted_thinking blocks are not replayed (thinking is off).
            entry = {'role': 'assistant', 'content': ''.join(text)}
            if calls:
                entry['tool_calls'] = calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            kind = block.get('type') if isinstance(block, dict) else None
            if kind == 'tool_result':
                result = _text_of(block.get('content'))
                if block.get('is_error'):
                    result = 'Error: ' + result
                messages.append({'role': 'tool', 'tool_call_id': block.get('tool_use_id'), 'content': result})
            elif kind == 'text':
                parts.append({'type': 'text', 'text': block.get('text') or ''})
            elif kind == 'image':
                parts.append({'type': 'image_url', 'image_url': {'url': _image_url(block)}})
            elif kind == 'document':
                raise RequestError('Document blocks are not supported')
        if parts:
            messages.append({'role': 'user', 'content': parts})
    tools = []
    for tool in body.get('tools') or []:
        if not isinstance(tool, dict) or 'input_schema' not in tool:
            continue     # server-side tools (web search etc.) have no local implementation
        tools.append({'type': 'function', 'function': {'name': tool.get('name'),
                                                       'description': tool.get('description') or '',
                                                       'parameters': tool.get('input_schema') or {}}})
    choice = body.get('tool_choice') or {}
    out = {'messages': messages, 'max_tokens': body.get('max_tokens'), 'stream': bool(body.get('stream')),
           'stop': body.get('stop_sequences') or None, 'temperature': body.get('temperature'),
           'top_p': body.get('top_p'), 'tools': tools,
           'model': body.get('model')}
    if isinstance(choice, dict) and choice.get('type') == 'none':
        out['tool_choice'] = 'none'
    if ANTHROPIC_THINKING and isinstance(body.get('thinking'), dict):
        out['thinking'] = body['thinking']
    return out


def parse_messages_request(body):
    return parse_request(to_openai(body))


def message_id():
    return 'msg_' + uuid4().hex[:24]


def to_anthropic(response, model):
    """OpenAI chat completion (Service.run) -> Anthropic message."""
    choice = response['choices'][0]
    message = choice['message']
    content = []
    if message.get('reasoning_content'):
        content.append({'type': 'thinking', 'thinking': message['reasoning_content'], 'signature': ''})
    if message.get('content'):
        content.append({'type': 'text', 'text': message['content']})
    for call in message.get('tool_calls') or []:
        arguments = call['function'].get('arguments')
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {'value': arguments}
        call_id = call.get('id') or ''
        content.append({'type': 'tool_use', 'id': call_id if call_id.startswith('toolu_') else 'toolu_' + call_id,
                        'name': call['function']['name'], 'input': arguments})
    stop = {'tool_calls': 'tool_use', 'length': 'max_tokens'}.get(choice['finish_reason'], 'end_turn')
    usage = response.get('usage', {})
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
    return {'id': message_id(), 'type': 'message', 'role': 'assistant', 'model': model, 'content': content,
            'stop_reason': stop, 'stop_sequence': None,
            'usage': {'input_tokens': max(0, usage.get('prompt_tokens', 0) - cached),
                      'cache_read_input_tokens': cached, 'cache_creation_input_tokens': 0,
                      'output_tokens': usage.get('completion_tokens', 0)}}


def sse_event(kind, payload):
    return f'event: {kind}\ndata: {json.dumps(payload)}\n\n'.encode()


PING = sse_event('ping', {'type': 'ping'})


def stream_events(message):
    """The event sequence of a streamed Anthropic message whose content is already complete."""
    start = dict(message, content=[], stop_reason=None, stop_sequence=None,
                 usage=dict(message['usage'], output_tokens=0))
    events = [sse_event('message_start', {'type': 'message_start', 'message': start})]
    for index, block in enumerate(message['content']):
        if block['type'] == 'thinking':
            events.append(sse_event('content_block_start', {'type': 'content_block_start', 'index': index,
                                                            'content_block': {'type': 'thinking', 'thinking': ''}}))
            events.append(sse_event('content_block_delta', {'type': 'content_block_delta', 'index': index,
                                                            'delta': {'type': 'thinking_delta', 'thinking': block['thinking']}}))
        elif block['type'] == 'text':
            events.append(sse_event('content_block_start', {'type': 'content_block_start', 'index': index,
                                                            'content_block': {'type': 'text', 'text': ''}}))
            events.append(sse_event('content_block_delta', {'type': 'content_block_delta', 'index': index,
                                                            'delta': {'type': 'text_delta', 'text': block['text']}}))
        else:
            events.append(sse_event('content_block_start', {
                'type': 'content_block_start', 'index': index,
                'content_block': {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}}))
            events.append(sse_event('content_block_delta', {
                'type': 'content_block_delta', 'index': index,
                'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}}))
        events.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
    events.append(sse_event('message_delta', {'type': 'message_delta',
                                              'delta': {'stop_reason': message['stop_reason'], 'stop_sequence': None},
                                              'usage': {'output_tokens': message['usage']['output_tokens']}}))
    events.append(sse_event('message_stop', {'type': 'message_stop'}))
    return events


def error_payload(error):
    kind = {401: 'authentication_error', 403: 'permission_error', 404: 'not_found_error',
            429: 'rate_limit_error', 503: 'overloaded_error'}.get(error.status, 'invalid_request_error')
    if error.status >= 500 and error.status != 503:
        kind = 'api_error'
    return {'type': 'error', 'error': {'type': kind, 'message': error.args[0] if error.args else str(error)}}

