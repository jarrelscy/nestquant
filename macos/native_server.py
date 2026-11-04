"""Experimental loopback-only OpenAI text endpoint for native NestQuant."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
import signal
import time
from uuid import uuid4

import mlx.core as mx
from transformers import AutoTokenizer
from mlx_vlm.tools import load_tool_module, process_tool_calls

from checkpoint import validate
from metrics import MemoryGuard
from native_generate import greedy, layer_major_prefill, load_runtime
from native_refinements import NativeHotRefinements
from native_prefix import PrefixCache

MODEL = 'glm-nestquant-local'
MAX_BODY = 128 * 1024


def parse_request(body):
    if not isinstance(body, dict) or body.get('model', MODEL) != MODEL:
        raise ValueError('Select the local GLM model')
    if (body.get('stream', False) or body.get('n', 1) != 1 or body.get('temperature', 0) != 0
            or body.get('tool_choice', 'auto') not in ('auto', 'none')):
        raise ValueError('Use one greedy non-streaming completion with auto or no tool selection')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError('Expected between 1 and 128 messages')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported chat message role')
        content = message.get('content')
        if not isinstance(content, str) and not (message['role'] == 'assistant' and content is None):
            raise ValueError('Use text-only messages')
        calls = message.get('tool_calls', [])
        if not isinstance(calls, list) or (calls and message['role'] != 'assistant'):
            raise ValueError('Invalid assistant tool calls')
        for call in calls:
            function = call.get('function') if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get('name'), str):
                raise ValueError('Invalid tool function')
            arguments = function.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Expected JSON object tool arguments')
            function['arguments'] = arguments
    tools = body.get('tools', [])
    if not isinstance(tools, list) or len(tools) > 32 or any(
        not isinstance(tool, dict) or tool.get('type') != 'function'
        or not isinstance(tool.get('function'), dict) or not isinstance(tool['function'].get('name'), str)
        for tool in tools
    ):
        raise ValueError('Expected up to 32 function schemas')
    limit = body.get('max_completion_tokens', body.get('max_tokens', 128))
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError('Output must be between 1 and 512 tokens')
    return messages, [] if body.get('tool_choice') == 'none' else tools, limit


class Session:
    def __init__(self, root, archive, backbone, mixed):
        self.guard = MemoryGuard(82)
        validate(root)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.template = (Path(__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        eos = json.loads((root / 'generation_config.json').read_text())['eos_token_id']
        self.eos = {eos} if isinstance(eos, int) else set(eos)
        self.hot = NativeHotRefinements(root, .02, 160) if mixed else None
        try:
            self.model, self.banks = load_runtime(root, backbone, archive, [160] * 42, self.guard, self.hot)
        except BaseException:
            if self.hot is not None:
                self.hot.close()
            raise
        self.prefix = PrefixCache(self.model, self.hot, 8, boundary=32)
        self.parser = load_tool_module('glm47')

    def completion(self, body):
        messages, tools, limit = parse_request(body)
        ids = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False,
                                                 add_generation_prompt=True, chat_template=self.template)
        if not ids or len(ids) + limit > 4096:
            raise ValueError('Rendered prompt and completion exceed 4096 tokens')
        cache = self.model.make_cache()
        if self.hot is not None:
            self.hot.reset()
        tick = time.monotonic()
        logits, hits = self.prefix.prefill(layer_major_prefill, self.model, ids, cache, self.hot, self.guard)
        token = greedy(logits)
        ttft = time.monotonic() - tick
        generated = []
        stop = False
        for index in range(limit):
            if token in self.eos:
                stop = True
                break
            generated.append(token)
            if index + 1 == limit:
                break
            if self.hot is not None:
                self.hot.begin_decode(token)
            logits = self.model(mx.array([[token]]), cache=cache).logits
            mx.eval(logits, [entry.state for entry in cache])
            if self.hot is not None:
                self.hot.commit([token])
            token = greedy(logits)
            self.guard.check()
        elapsed = time.monotonic() - tick
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        message = dict(role='assistant', content=self.tokenizer.decode(generated, skip_special_tokens=True))
        reason = 'stop' if stop else 'length'
        if tools:
            result = process_tool_calls(text, self.parser, tools)
            allowed = {tool['function']['name'] for tool in tools}
            calls = [call for call in result.calls if call['function']['name'] in allowed]
            if calls:
                message['tool_calls'] = [{key: call[key] for key in ('id', 'type', 'function')} for call in calls]
                message['content'] = result.remaining_text
                reason = 'tool_calls'
        return dict(id='chatcmpl-'+uuid4().hex, object='chat.completion', created=int(time.time()), model=MODEL,
                    choices=[dict(index=0, message=message, finish_reason=reason)],
                    usage=dict(prompt_tokens=len(ids), completion_tokens=len(generated),
                               total_tokens=len(ids)+len(generated), prompt_tokens_details=dict(cached_tokens=hits)),
                    timings=dict(ttft_seconds=ttft, elapsed_seconds=elapsed, computed_prompt_tokens=len(ids)-hits,
                                 prefix_cache_bytes=self.prefix.bytes, prefix_evictions=self.prefix.evictions))

    def close(self):
        try:
            if self.hot is not None:
                self.hot.close()
        finally:
            for bank in self.banks:
                bank.close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def reply(self, status, payload):
        response = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        if self.path == '/health':
            self.reply(200, {'status': 'ok'})
        elif self.path == '/v1/models':
            self.reply(200, dict(object='list', data=[dict(id=MODEL, object='model', owned_by='local')]))
        else:
            self.reply(404, {'error': {'message': 'Unknown endpoint'}})

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.reply(404, {'error': {'message': 'Unknown endpoint'}})
            return
        try:
            if (self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',
                                                  f'localhost:{self.server.server_port}')
                    or self.headers.get_content_type() != 'application/json'
                    or self.headers.get('Origin') is not None):
                raise ValueError('Expected a local JSON client')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('Request body exceeds the accepted range')
            body = json.loads(self.rfile.read(length))
            response = self.server.session.completion(body)
        except (ValueError, UnicodeError, TimeoutError):
            self.reply(400, {'error': {'message': 'Invalid or unsupported completion request'}})
            return
        except MemoryError as exc:
            print(f'Runtime memory guard stopped service: {exc}', flush=True)
            raise SystemExit(75)
        self.reply(200, response)

    def log_message(self, format, *args):
        pass


def stop(_signal, _frame):
    raise SystemExit(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--port', type=int, default=18085)
    p.add_argument('--refinements', action='store_true')
    a = p.parse_args()
    if not 1024 <= a.port <= 65535:
        p.error('Use a port between 1024 and 65535')
    logging.getLogger('mlx_vlm.server').disabled = True
    signal.signal(signal.SIGTERM, stop)
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(76 * 1024**3)
    with HTTPServer(('127.0.0.1', a.port), Handler) as server:
        server.session = Session(a.model, a.archive, a.backbone, a.refinements)
        try:
            server.serve_forever()
        finally:
            server.session.close()
