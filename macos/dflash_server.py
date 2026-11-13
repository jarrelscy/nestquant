"""Loopback OpenAI-compatible chat-completions server for GLM-5.3-Flash NestQuant with DFlash2 speculative decoding.

One request runs at a time (others queue, then get 503). Decoding is greedy: DFlash2 with the adaptive verify-block
policy (block_size -8) and full 4-bit verification. The model cache, the refinement predictor state and the
drafter's target hidden states are kept between requests. When a prompt starts with the tokens the cache holds
(previous prompt + generated reply), only the new suffix is prefilled. When it diverges inside the previous reply
(e.g. the re-rendered assistant turn tokenizes differently, or a stop string truncated it), the state is restored to
a checkpoint taken at the end of the previous prompt and prefilling resumes from there. Otherwise a fresh prefill
runs. Thinking is off (spark/flash/thinking_off.jinja); temperature and other sampling fields are accepted and
ignored.

Run under the model lock, for example:
  /tmp/nq_model_lock.sh server ~/src/nestquant/.venv-macos/bin/python macos/dflash_server.py --port 18090
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import sys
import queue
import threading
import time
from concurrent.futures import Future

from dflash_protocol import (MODEL, RequestError, SSE_DONE, chunk, completion_id, completion_response,
                             common_prefix_length, context_error, find_stop, parse_request, plan_reuse, safe_prefix_length, sse)

MAX_BODY = 16 * 1024**2
HOME = Path.home()


class GpuThread:
    """Runs submitted calls on the thread that calls run_forever(). MLX streams belong to the thread that created
    them, so every model call must happen on the thread that loaded the model."""

    def __init__(self):
        self.jobs = queue.Queue()

    def call(self, fn, *args, **kwargs):
        if threading.current_thread() is getattr(self, 'owner', None):
            return fn(*args, **kwargs)
        future = Future()
        self.jobs.put((future, fn, args, kwargs))
        return future.result()

    def run_forever(self, stop):
        self.owner = threading.current_thread()
        while not stop.is_set():
            try:
                future, fn, args, kwargs = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)


class Service:
    """Model-agnostic request handling around a backend with encode/decode/generate (see DFlashSession)."""

    def __init__(self, backend, model_name=MODEL, queue_timeout=900.0, log=print, gpu=None):
        self.gpu = gpu
        self.backend = backend
        self.model_name = model_name
        self.queue_timeout = queue_timeout
        self.lock = threading.Lock()
        self.log = log
        self.requests = 0

    def prepare(self, parsed):
        try:
            ids = self.backend.encode(parsed['messages'], parsed['tools'])
        except RequestError:
            raise
        except Exception as error:   # template errors and the like
            raise RequestError(f'Could not render the chat template: {type(error).__name__}: {error}') from error
        limit = self.backend.max_context
        if len(ids) >= limit:
            raise context_error(len(ids), limit)
        return ids, min(parsed['max_tokens'], limit - len(ids))

    def run(self, parsed, ids, max_tokens, emit=None, cid=None):
        """Generate one completion. emit(text_delta) is called while streaming. Returns the response dict."""
        if not self.lock.acquire(timeout=self.queue_timeout):
            raise RequestError('The server is busy with another request', status=503, kind='server_error')
        try:
            return self._run(parsed, ids, max_tokens, emit, cid)
        finally:
            self.lock.release()

    def _run(self, parsed, ids, max_tokens, emit, cid):
        backend = self.backend
        eos = backend.eos
        stops = parsed['stop']
        window = max((len(s) for s in stops), default=0) + 16

        def visible(tokens):
            return backend.decode([t for t in tokens if t not in eos])

        def should_stop(generated):
            return find_stop(visible(generated[-window:]), stops) is not None

        sent = [0]

        def on_token(generated):
            text = visible(generated)
            cut = find_stop(text, stops) if stops else None
            end = cut if cut is not None else safe_prefix_length(text, stops)
            if end > sent[0]:
                emit(text[sent[0]:end])
                sent[0] = end

        started = time.perf_counter()
        args = (ids, max_tokens, should_stop if stops else None, on_token if emit else None)
        out = self.gpu.call(backend.generate, *args) if self.gpu is not None else backend.generate(*args)
        generated = list(out['generated_ids'])
        if out['stopped'] == 'eos':
            generated = generated[:-1]
        text = visible(generated)
        finish = 'stop' if out['stopped'] else 'length'
        if stops:
            cut = find_stop(text, stops)
            if cut is not None:
                text, finish = text[:cut], 'stop'
        message = dict(role='assistant', content=text)
        if parsed['tools'] and hasattr(backend, 'tool_calls'):
            parsed_calls = backend.tool_calls(text, parsed['tools'])
            if parsed_calls:
                calls, remaining = parsed_calls
                message['tool_calls'] = calls
                message['content'] = remaining or None
                finish = 'tool_calls'
        if emit is not None and len(text) > sent[0] and 'tool_calls' not in message:
            emit(text[sent[0]:])
        stats = out.get('stats', {})
        timings = dict(stats, elapsed_seconds=round(time.perf_counter() - started, 3))
        self.requests += 1
        self.log(json.dumps(dict(event='request', n=self.requests, prompt_tokens=len(ids),
                                 completion_tokens=len(generated), finish_reason=finish, **timings)), flush=True)
        response = completion_response(self.model_name, message, finish, len(ids), len(generated),
                                       stats.get('reused_tokens', 0), timings, cid)
        return response


class DFlashSession:
    """GLM-5.3-Flash NestQuant runtime + DFlash2 drafter with cross-request prefix reuse."""

    def __init__(self, args):
        import mlx.core as mx
        import torch
        from transformers import AutoTokenizer
        from mlx_vlm.speculative.drafters import load_drafter
        from mlx_vlm.tools import load_tool_module

        import native_dflash as nd
        import native_generate as ng
        from metrics import MemoryGuard
        from native_prefill import BatchedPrefill
        from native_refinements import NativeHotRefinements

        self.mx, self.nd, self.ng = mx, nd, ng
        nd._install_capture()
        nd._relax_compatibility()
        mx.set_cache_limit(256 * 1024**2)
        mx.set_memory_limit(96 * 1024**3)
        mx.set_wired_limit(92 * 1024**3)
        torch.set_num_threads(4)
        self.guard = MemoryGuard(args.guard_gib, 0.8)
        self.guard.allowed_pressure = (1, 2)
        self.hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, 4, 32, 512)
        self.model, self.banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, self.guard,
                                                 self.hot, True, 56)
        try:
            self.hot.warm(4, lambda: self.guard.check(force=True))
            self.hot.draft_rows_base = False        # full 4-bit verification
            nd.BATCHED = BatchedPrefill(args.model)
            self.drafter, kind = load_drafter(str(args.drafter))
            mx.eval(self.drafter.parameters())
            self.guard.check(force=True)
            config = self.drafter.config
            types = getattr(config, 'layer_types', None) or []
            window = getattr(config, 'sliding_window', None)
            # The drafter attends to at most window - 1 context rows; keep that many target hidden rows.
            self.window = window - 1 if window and types and all(t == 'sliding_attention' for t in types) else None
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            self.template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
            end = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
            self.eos = {end} if isinstance(end, int) else set(end)
            self.tool_parser = load_tool_module('glm47')
        except BaseException:
            self.close()
            raise
        self.max_context = args.max_context
        self.block_size = args.block_size
        self.reuse = not args.no_reuse
        self.cache = None
        self.cache_ids = []
        self.ctx_hidden = None
        self.checkpoint = None
        print(json.dumps(dict(event='ready', drafter=kind, target_layers=nd.target_layers(self.drafter),
                              drafter_window=self.window, max_context=self.max_context,
                              memory_gib=round(self._footprint(), 2))), flush=True)

    # ------------------------------------------------------------------ text
    def encode(self, messages, tools):
        return list(self.tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=True,
                                                       return_dict=False, add_generation_prompt=True,
                                                       chat_template=self.template))

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def tool_calls(self, text, tools):
        from mlx_vlm.tools import process_tool_calls
        result = process_tool_calls(text, self.tool_parser, tools)
        allowed = {tool['function']['name'] for tool in tools}
        calls = [call for call in result.calls if call['function']['name'] in allowed]
        if not calls:
            return None
        out = []
        for call in calls:
            arguments = call['function'].get('arguments')
            out.append(dict(id=call.get('id') or 'call_' + completion_id()[9:33], type='function',
                            function=dict(name=call['function']['name'],
                                          arguments=arguments if isinstance(arguments, str) else json.dumps(arguments))))
        return out, result.remaining_text

    # ------------------------------------------------------------------ state
    def _footprint(self, peak=False):
        usage = self.ng.process_usage()
        return (usage.lifetime_max_phys_footprint if peak else usage.phys_footprint) / 2**30

    def _leaves(self):
        from native_prefix import leaves
        return list(leaves(self.cache))

    def _kv_offset(self):
        from mlx_vlm.models.cache import KVCache
        return next(leaf.offset for leaf in self._leaves() if type(leaf) is KVCache)

    def _trim_ctx(self, ctx):
        if self.window is not None and ctx.shape[1] > self.window:
            ctx = ctx[:, -self.window:]
        return self.mx.contiguous(ctx)

    def _take_checkpoint(self, length):
        """Restorable state at a committed prefill boundary. Attention caches are append-only, so they keep only
        their offset; recurrent and pooled-indexer state and the predictor are copied (small)."""
        from mlx_vlm.models.cache import KVCache
        from native_prefix import arrays, clone, predictor_snapshot
        parts = []
        for leaf in self._leaves():
            if getattr(leaf, 'is_speculating', False):
                raise RuntimeError('Cannot checkpoint provisional cache state')
            if type(leaf) is KVCache:
                parts.append(('kv', leaf.offset))
            else:
                parts.append(('snapshot', clone(leaf.prefix_cache_snapshot())))
        self.mx.eval(arrays([part for kind, part in parts if kind == 'snapshot']))
        predictor, _ = predictor_snapshot(self.hot)
        return dict(length=length, parts=parts, predictor=predictor, position=self.hot.position,
                    ctx=self.ctx_hidden)

    def _restore_checkpoint(self):
        from native_prefix import arrays, clone, restore_predictor
        checkpoint = self.checkpoint
        leaves = self._leaves()
        if len(leaves) != len(checkpoint['parts']):
            raise RuntimeError('Checkpoint does not match the cache layout')
        for leaf, (kind, part) in zip(leaves, checkpoint['parts']):
            if kind == 'kv':
                if leaf.offset < part:
                    raise RuntimeError('Attention cache is shorter than its checkpoint')
                leaf.offset = part
            else:
                leaf.prefix_cache_restore(clone(part))
        self.mx.eval(arrays([leaf.state for leaf in leaves]))
        restore_predictor(self.hot, checkpoint['predictor'])
        predictor = self.hot.predictor
        predictor.ahead, predictor.ahead_started = None, False
        self.hot.position = checkpoint['position']
        self.ctx_hidden = checkpoint['ctx']
        self.cache_ids = self.cache_ids[:checkpoint['length']]

    def _resume_prefill(self):
        """Switch the refinement state from decode back to prefill without discarding committed history."""
        hot = self.hot
        hot.phase = 'prefill'
        hot.prefill_bank = None
        hot.prefill_layer = None
        hot.chunk_index = 0
        hot.traces.clear()
        predictor = hot.predictor
        if predictor.ahead is not None:
            # Decode planned the next (never committed) token; drop it and its row in the jT history.
            predictor.ahead, predictor.ahead_started = None, False
            extra = predictor.net.position - predictor.t
            if extra > 0:
                predictor.net.cache = [None if entry is None else tuple(t[..., :-extra, :] for t in entry)
                                       for entry in predictor.net.cache]
                predictor.net.position = predictor.t
        if hot.position != predictor.t:
            raise RuntimeError('Predictor clock differs from the committed token count')

    def drop_state(self):
        self.cache, self.cache_ids, self.ctx_hidden, self.checkpoint = None, [], None, None
        self.mx.clear_cache()

    # ------------------------------------------------------------------ generation
    def generate(self, ids, max_tokens, should_stop=None, on_token=None):
        mx, nd, ng = self.mx, self.nd, self.ng
        mode, reused = ('fresh', 0)
        if self.reuse and self.cache is not None:
            mode, reused = plan_reuse(self.cache_ids, self.checkpoint['length'] if self.checkpoint else 0, ids)
        lcp = common_prefix_length(self.cache_ids, ids)
        cache, cache_ids, ctx, checkpoint = self.cache, self.cache_ids, self.ctx_hidden, self.checkpoint
        # Until this request finishes cleanly the live state is unknown; a failure leaves nothing to reuse.
        self.cache, self.cache_ids, self.ctx_hidden, self.checkpoint = None, [], None, None
        started = time.perf_counter()
        if mode == 'fresh':
            if cache is not None:
                del cache
                cache_ids, ctx, checkpoint = [], None, None
                mx.clear_cache()
            self.cache = self.model.make_cache()
            self.hot.reset()
            self.ctx_hidden = None
        else:
            self.cache, self.cache_ids, self.ctx_hidden, self.checkpoint = cache, cache_ids, ctx, checkpoint
            if mode == 'rollback':
                self._restore_checkpoint()
            self._resume_prefill()
            if self.cache_ids != ids[:reused] or self._kv_offset() != reused:
                raise RuntimeError('Reused cache does not hold the expected prefix')
            self.checkpoint, cache = None, None
        suffix = ids[reused:]
        logits, hidden = nd.capture_prefill(self.model, self.drafter, suffix, self.cache, self.hot, self.guard,
                                            suffix=reused > 0)
        ctx = hidden if self.ctx_hidden is None else mx.concatenate([self.ctx_hidden, hidden], axis=1)
        self.ctx_hidden = self._trim_ctx(ctx)
        mx.eval(self.ctx_hidden)
        del ctx, hidden
        first = ng.greedy(logits)
        prefill_seconds = time.perf_counter() - started
        flat = logits.reshape(-1).astype(mx.float32)
        top = mx.argsort(-flat)[:5]
        top5 = [[int(i), round(float(v), 3)] for i, v in zip(top.tolist(), flat[top].tolist())]
        self.cache_ids = list(ids)
        checkpoint = self._take_checkpoint(len(ids)) if self.reuse else None
        out = nd.dflash_decode(self.model, self.drafter, self.cache, self.ctx_hidden, first, max_tokens, self.eos,
                               self.hot, self.guard, self.block_size, should_stop=should_stop, on_token=on_token,
                               record_hidden=self.reuse)
        stats = dict(mode=mode, reused_tokens=reused, prefilled_tokens=len(suffix), first_top5=top5,
                     prefill_path=nd.LAST_PREFILL_PATH,
                     common_prefix=lcp,
                     prefill_seconds=round(prefill_seconds, 3),
                     prefill_tok_s=round(len(suffix) / prefill_seconds, 1) if prefill_seconds > 0 else None,
                     decode_seconds=round(out['decode_seconds'], 3),
                     decode_tok_s=round(out['decode_tps'], 2) if out['decode_tps'] else None,
                     rounds=out['rounds'], mean_accepted=round(out['mean_accepted'], 3),
                     mean_drafted=round(out['mean_drafted'], 3),
                     memory_gib=round(self._footprint(), 2), peak_gib=round(self._footprint(peak=True), 2))
        if self.reuse:
            committed = out['committed_ids']
            self.cache_ids = list(ids) + committed
            offset = self._kv_offset()
            if offset != len(self.cache_ids) or any(getattr(leaf, 'is_speculating', False) for leaf in self._leaves()):
                print(json.dumps(dict(event='cache_mismatch', kv_offset=offset, expected=len(self.cache_ids))),
                      flush=True)
                self.drop_state()
            else:
                rows = out['committed_hidden']
                if rows:
                    self.ctx_hidden = self._trim_ctx(mx.concatenate([self.ctx_hidden] + rows, axis=1))
                    mx.eval(self.ctx_hidden)
                self.checkpoint = checkpoint
                stats['cached_tokens_after'] = len(self.cache_ids)
        else:
            self.drop_state()
        return dict(generated_ids=out['generated_ids'], stopped=out['stopped'], stats=stats)

    def close(self):
        try:
            if self.nd.BATCHED is not None:
                self.nd.BATCHED.close()
            self.hot.close()
        finally:
            for bank in self.banks:
                bank.close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'nestquant-dflash/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0].rstrip('/')
        service = self.server.service
        if path in ('/health', '/v1/health'):
            self.reply(200, {'status': 'ok', 'busy': service.lock.locked()})
        elif path == '/v1/models':
            self.reply(200, dict(object='list', data=[dict(id=service.model_name, object='model', created=0,
                                                         owned_by='local')]))
        elif path.startswith('/v1/models/'):
            self.reply(200, dict(id=service.model_name, object='model', created=0, owned_by='local'))
        else:
            self.reply(404, {'error': {'message': 'Unknown endpoint', 'type': 'invalid_request_error'}})

    def do_POST(self):
        path = self.path.split('?')[0].rstrip('/')
        if path not in ('/v1/chat/completions', '/chat/completions'):
            self.reply(404, {'error': {'message': 'Unknown endpoint', 'type': 'invalid_request_error'}})
            return
        service = self.server.service
        try:
            if self.headers.get('Origin') is not None:
                raise RequestError('Browser origins are not accepted', status=403)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise RequestError('Request body is empty or too large')
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RequestError('Request body is not valid JSON') from error
            parsed = parse_request(body)
            ids, max_tokens = service.prepare(parsed)
        except RequestError as error:
            self.reply(error.status, error.payload())
            return
        except (ValueError, TimeoutError) as error:
            self.reply(400, RequestError(str(error)).payload())
            return
        try:
            if parsed['stream']:
                self.stream(service, parsed, ids, max_tokens)
            else:
                self.reply(200, service.run(parsed, ids, max_tokens))
        except RequestError as error:
            self.safe_reply(error.status, error.payload())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.server.on_failure(error)
            self.safe_reply(500, {'error': {'message': f'Runtime error: {type(error).__name__}: {error}',
                                            'type': 'server_error'}})

    def safe_reply(self, status, payload):
        if getattr(self, '_streaming', False):
            return
        try:
            self.reply(status, payload)
        except OSError:
            pass

    def stream(self, service, parsed, ids, max_tokens):
        cid = completion_id()
        model = service.model_name
        alive = [True]
        started = [False]

        def write(data):
            if not alive[0]:
                return
            try:
                if not started[0]:
                    self._streaming = started[0] = True
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.write(sse(chunk(cid, model, {'role': 'assistant', 'content': ''})))
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                alive[0] = False

        def emit(text):
            write(sse(chunk(cid, model, {'content': text})))

        response = service.run(parsed, ids, max_tokens, emit=emit, cid=cid)
        choice = response['choices'][0]
        if choice['message'].get('tool_calls'):
            calls = [dict(index=i, **call) for i, call in enumerate(choice['message']['tool_calls'])]
            write(sse(chunk(cid, model, {'tool_calls': calls})))
        write(sse(chunk(cid, model, {}, choice['finish_reason'])))
        if parsed['include_usage']:
            payload = chunk(cid, model, {}, None, response['usage'])
            payload['choices'] = []
            write(sse(payload))
        write(SSE_DONE)

    def log_message(self, format, *args):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service):
        super().__init__(address, Handler)
        self.service = service

    def on_failure(self, error):
        print(json.dumps(dict(event='request_failed', error=f'{type(error).__name__}: {error}')), flush=True)
        backend = self.service.backend
        if hasattr(backend, 'drop_state'):
            gpu = self.service.gpu
            gpu.call(backend.drop_state) if gpu is not None else backend.drop_state()
        if isinstance(error, MemoryError) and hasattr(backend, 'guard'):
            try:
                backend.guard.check(force=True)
            except MemoryError:
                print(json.dumps(dict(event='exit', reason='memory guard still failing')), flush=True)
                threading.Thread(target=self.shutdown, daemon=True).start()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    p.add_argument('--archive', type=Path, default=HOME / 'Models/GLM/resident-banks-v1')
    p.add_argument('--backbone', type=Path, default=HOME / 'Models/GLM/backbone-affine4-e8-v1')
    p.add_argument('--drafter', type=Path, default=HOME / 'Models/GLM/drafters/dflash2')
    p.add_argument('--port', type=int, default=18090)
    p.add_argument('--model-name', default=MODEL, help='Name reported by /v1/models and in responses')
    p.add_argument('--max-context', type=int, default=32768, help='Prompt plus completion tokens (up to 65536)')
    p.add_argument('--block-size', type=int, default=-8, help='DFlash verify block; negative = adaptive up to N')
    p.add_argument('--hot-fraction', type=float, default=0.15)
    p.add_argument('--budget-gib', type=int, default=16)
    p.add_argument('--guard-gib', type=float, default=98)
    p.add_argument('--queue-timeout', type=float, default=900)
    p.add_argument('--no-reuse', action='store_true', help='Fresh prefill for every request')
    a = p.parse_args()
    if not 1024 <= a.port <= 65535 or not 1024 <= a.max_context <= 65536 or a.block_size == 0:
        p.error('Use a port in 1024-65535, --max-context in 1024-65536 and a non-zero block size')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    session = DFlashSession(a)
    gpu, stop = GpuThread(), threading.Event()
    try:
        with Server(('127.0.0.1', a.port), Service(session, a.model_name, a.queue_timeout, gpu=gpu)) as server:
            print(json.dumps(dict(event='listening', url=f'http://127.0.0.1:{a.port}/v1')), flush=True)
            serving = threading.Thread(target=lambda: (server.serve_forever(), stop.set()), daemon=True)
            serving.start()
            try:
                gpu.run_forever(stop)       # model work stays on the main thread
            finally:
                server.shutdown()
    finally:
        print(json.dumps(dict(event='shutdown', peak_gib=round(session._footprint(peak=True), 2))), flush=True)
        session.close()


if __name__ == '__main__':
    main()
