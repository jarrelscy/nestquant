"""CPU-only tests for the DFlash2 chat server: HTTP layer with a fake backend, prefix planning and stop strings."""

import http.client
import json
import threading
import time
import unittest

from dflash_protocol import (RequestError, common_prefix_length, find_stop, parse_request, plan_reuse,
                             safe_prefix_length)
from dflash_server import GpuThread, Server, Service

EOS = 0


class FakeBackend:
    """Characters are tokens (ord); generate replays a scripted reply then EOS, like dflash_decode."""

    def __init__(self, reply='Hello world', max_context=4096, delay=0.0):
        self.reply, self.max_context, self.delay = reply, max_context, delay
        self.eos = {EOS}
        self.calls = []

    def encode(self, messages, tools):
        text = ''.join(f"<{m['role']}>{m['content']}" for m in messages) + '<assistant>'
        return [ord(c) for c in text]

    def decode(self, ids):
        return ''.join(chr(i) for i in ids)

    def generate(self, ids, max_tokens, should_stop=None, on_token=None):
        self.calls.append(dict(ids=ids, max_tokens=max_tokens))
        time.sleep(self.delay)
        script = [ord(c) for c in self.reply] + [EOS]
        generated, stopped = [], None
        for token in script:
            generated.append(token)
            if token in self.eos:
                stopped = 'eos'
            elif should_stop is not None and should_stop(generated):
                stopped = 'stop'
            if on_token is not None:
                on_token(generated)
            if stopped or len(generated) >= max_tokens:
                break
        return dict(generated_ids=generated, stopped=stopped, stats=dict(reused_tokens=3, prefill_seconds=0.1))


class ServerHarness:
    def __init__(self, backend, queue_timeout=5.0, api_key=None):
        self.server = Server(('127.0.0.1', 0), Service(backend, 'glm-nestquant', queue_timeout, log=lambda *a, **k: None),
                             api_key=api_key)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        conn.request(method, path, data, {'Content-Type': 'application/json', **(headers or {})})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, response.getheader('Content-Type'), payload

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.harness = ServerHarness(self.backend)

    def tearDown(self):
        self.harness.close()

    def chat(self, **extra):
        body = dict(model='glm-nestquant', messages=[{'role': 'user', 'content': 'hi'}], **extra)
        status, kind, payload = self.harness.request('POST', '/v1/chat/completions', body)
        return status, kind, payload

    def test_models(self):
        status, _, payload = self.harness.request('GET', '/v1/models')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)['data'][0]['id'], 'glm-nestquant')

    def test_completion_with_unknown_fields(self):
        status, _, payload = self.chat(temperature=0.7, top_p=0.9, drop_params=True, reasoning_effort=None,
                                       response_format=None, extra_unknown={'a': 1}, session_id='x', seed=3)
        self.assertEqual(status, 200, payload)
        out = json.loads(payload)
        self.assertEqual(out['choices'][0]['message'], {'role': 'assistant', 'content': 'Hello world'})
        self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
        prompt = len('<user>hi<assistant>')
        self.assertEqual(out['usage'], dict(prompt_tokens=prompt, completion_tokens=11, total_tokens=prompt + 11,
                                            prompt_tokens_details=dict(cached_tokens=3)))

    def test_length_finish(self):
        out = json.loads(self.chat(max_tokens=5)[2])
        self.assertEqual(out['choices'][0]['message']['content'], 'Hello')
        self.assertEqual(out['choices'][0]['finish_reason'], 'length')
        self.assertEqual(out['usage']['completion_tokens'], 5)

    def test_max_completion_tokens_preferred(self):
        self.chat(max_tokens=50, max_completion_tokens=7)
        self.assertEqual(self.backend.calls[-1]['max_tokens'], 7)

    def test_stop_string(self):
        out = json.loads(self.chat(stop=['o w'])[2])
        self.assertEqual(out['choices'][0]['message']['content'], 'Hell')
        self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
        out = json.loads(self.chat(stop='zzz')[2])
        self.assertEqual(out['choices'][0]['message']['content'], 'Hello world')

    def test_content_parts_and_developer_role(self):
        body = dict(messages=[{'role': 'developer', 'content': 'sys'},
                              {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b',
                                                                                         'cache_control': {}}]}])
        status, _, _ = self.harness.request('POST', '/v1/chat/completions', body)
        self.assertEqual(status, 200)
        self.assertEqual(FakeBackend.decode(None, self.backend.calls[-1]['ids']), '<system>sys<user>ab<assistant>')

    def test_context_limit(self):
        self.backend.max_context = 10
        status, _, payload = self.chat()
        self.assertEqual(status, 400)
        error = json.loads(payload)['error']
        self.assertEqual(error['code'], 'context_length_exceeded')
        self.assertIn("maximum context length is 10 tokens", error['message'])

    def test_max_tokens_clamped_to_context(self):
        self.backend.max_context = 25
        self.chat(max_tokens=1000)
        self.assertEqual(self.backend.calls[-1]['max_tokens'], 25 - len('<user>hi<assistant>'))

    def test_bad_requests(self):
        for body in (b'not json', json.dumps({'messages': []}).encode(),
                     json.dumps({'messages': [{'role': 'wizard', 'content': 'x'}]}).encode(),
                     json.dumps({'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 0}).encode()):
            status, _, payload = self.harness.request('POST', '/v1/chat/completions', body)
            self.assertEqual(status, 400, body)
            self.assertIn('error', json.loads(payload))
        self.assertEqual(self.harness.request('POST', '/v1/completions', {'prompt': 'x'})[0], 404)

    def test_streaming(self):
        status, kind, payload = self.chat(stream=True, stream_options={'include_usage': True}, stop=['rld'])
        self.assertEqual(status, 200)
        self.assertTrue(kind.startswith('text/event-stream'))
        events = [line[6:] for line in payload.decode().split('\n') if line.startswith('data: ')]
        self.assertEqual(events[-1], '[DONE]')
        chunks = [json.loads(e) for e in events[:-1]]
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices'])
        self.assertEqual(text, 'Hello wo')
        finishes = [c['choices'][0]['finish_reason'] for c in chunks if c['choices']]
        self.assertEqual(finishes[-1], 'stop')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], len('Hello world'))
        self.assertEqual(chunks[0]['choices'][0]['delta']['role'], 'assistant')


class ApiKeyTest(unittest.TestCase):
    def test_key_required_except_health(self):
        harness = ServerHarness(FakeBackend(), api_key='secret-key')
        try:
            body = dict(model='glm-nestquant', messages=[{'role': 'user', 'content': 'hi'}])
            self.assertEqual(harness.request('POST', '/v1/chat/completions', body)[0], 401)
            self.assertEqual(harness.request('GET', '/v1/models')[0], 401)
            wrong = {'Authorization': 'Bearer nope'}
            self.assertEqual(harness.request('POST', '/v1/chat/completions', body, wrong)[0], 401)
            self.assertEqual(harness.request('GET', '/health')[0], 200)
            good = {'Authorization': 'Bearer secret-key'}
            self.assertEqual(harness.request('GET', '/v1/models', headers=good)[0], 200)
            self.assertEqual(harness.request('POST', '/v1/chat/completions', body, good)[0], 200)
        finally:
            harness.close()


class ConcurrencyTest(unittest.TestCase):
    def test_second_request_gets_503_after_queue_timeout(self):
        harness = ServerHarness(FakeBackend(delay=1.0), queue_timeout=0.2)
        try:
            body = dict(messages=[{'role': 'user', 'content': 'hi'}])
            results = []
            threads = [threading.Thread(target=lambda: results.append(
                harness.request('POST', '/v1/chat/completions', body)[0])) for _ in range(2)]
            for thread in threads:
                thread.start()
                time.sleep(0.1)
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(results), [200, 503])
        finally:
            harness.close()


class GpuThreadTest(unittest.TestCase):
    def test_generate_runs_on_the_owner_thread(self):
        backend = FakeBackend()
        seen = []
        original = backend.generate
        backend.generate = lambda *a: (seen.append(threading.current_thread()), original(*a))[1]
        gpu, stop = GpuThread(), threading.Event()
        owner = threading.Thread(target=gpu.run_forever, args=(stop,), daemon=True)
        owner.start()
        harness = ServerHarness(backend)
        harness.server.service.gpu = gpu
        try:
            for _ in range(2):
                self.assertEqual(harness.request('POST', '/v1/chat/completions',
                                                 dict(messages=[{'role': 'user', 'content': 'hi'}]))[0], 200)
            self.assertEqual(seen, [owner, owner])
        finally:
            harness.close()
            stop.set()


class PrefixPlanTest(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix_length([1, 2], [1, 2, 3]), 2)
        self.assertEqual(common_prefix_length([], [1]), 0)

    def test_extend_when_cache_is_a_prefix(self):
        self.assertEqual(plan_reuse([1, 2, 3, 4], 2, [1, 2, 3, 4, 5, 6]), ('extend', 4))

    def test_rollback_to_checkpoint_when_reply_differs(self):
        # previous prompt [1,2] (checkpoint), reply [3,4] re-rendered as [3,9]
        self.assertEqual(plan_reuse([1, 2, 3, 4], 2, [1, 2, 3, 9, 5]), ('rollback', 2))

    def test_fresh_when_divergence_precedes_checkpoint(self):
        self.assertEqual(plan_reuse([1, 2, 3, 4], 3, [1, 2, 7, 4, 5]), ('fresh', 0))
        self.assertEqual(plan_reuse([1, 2, 3, 4], 0, [1, 2, 3, 9]), ('fresh', 0))
        self.assertEqual(plan_reuse([], 0, [1, 2]), ('fresh', 0))

    def test_identical_prompt_needs_one_new_token(self):
        # The cache already holds every prompt token: logits need at least one prefilled token.
        self.assertEqual(plan_reuse([1, 2, 3], 2, [1, 2, 3]), ('rollback', 2))
        self.assertEqual(plan_reuse([1, 2, 3], 0, [1, 2, 3]), ('fresh', 0))
        # A prompt that is a strict prefix of the cache.
        self.assertEqual(plan_reuse([1, 2, 3, 4, 5], 2, [1, 2, 3]), ('rollback', 2))


class StopTest(unittest.TestCase):
    def test_find_stop(self):
        self.assertEqual(find_stop('abc</cmd>def', ['</cmd>', 'zz']), 3)
        self.assertIsNone(find_stop('abc', ['d']))
        self.assertEqual(find_stop('xaybx', ['b', 'a']), 1)

    def test_safe_prefix_holds_partial_stop_and_incomplete_utf8(self):
        self.assertEqual(safe_prefix_length('hello </cm', ['</cmd>']), len('hello '))
        self.assertEqual(safe_prefix_length('hello', ['</cmd>']), 5)
        self.assertEqual(safe_prefix_length('ab�', []), 2)


class ParseTest(unittest.TestCase):
    def test_defaults_and_stop_forms(self):
        parsed = parse_request({'messages': [{'role': 'user', 'content': 'x'}], 'stop': 'END'})
        self.assertEqual(parsed['stop'], ['END'])
        self.assertEqual(parsed['max_tokens'], 4096)
        self.assertFalse(parsed['stream'])

    def test_tool_call_arguments_parsed(self):
        parsed = parse_request({'messages': [
            {'role': 'user', 'content': 'x'},
            {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"a": 1}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]})
        self.assertEqual(parsed['messages'][1]['tool_calls'][0]['function']['arguments'], {'a': 1})
        self.assertEqual(parsed['messages'][1]['content'], '')

    def test_rejects_n_and_images(self):
        with self.assertRaises(RequestError):
            parse_request({'messages': [{'role': 'user', 'content': 'x'}], 'n': 2})
        with self.assertRaises(RequestError):
            parse_request({'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {}}]}]})

    def test_image_parts_kept_in_order(self):
        from dflash_protocol import image_sources
        parsed = parse_request({'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'a'},
                                         {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
                                         {'type': 'text', 'text': 'b'}]},
            {'role': 'assistant', 'content': 'ok'},
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': 'https://example.com/x.png'}]}]})
        self.assertEqual(parsed['messages'][0]['content'][1], {'type': 'image', 'source': 'data:image/png;base64,AA=='})
        self.assertEqual(parsed['messages'][1]['content'], 'ok')
        self.assertEqual(image_sources(parsed['messages']), ['data:image/png;base64,AA==', 'https://example.com/x.png'])
        with self.assertRaises(RequestError):
            parse_request({'messages': [{'role': 'user', 'content': [{'type': 'input_audio', 'input_audio': {}}]}]})


if __name__ == '__main__':
    unittest.main()
