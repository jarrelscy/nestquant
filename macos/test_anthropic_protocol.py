"""Anthropic Messages API: request/response conversion and the /v1/messages endpoints (fake backend, CPU only)."""

import json
import unittest

import anthropic_protocol as ap
from test_dflash_server import FakeBackend, ServerHarness


class ToolBackend(FakeBackend):
    """Replies with a scripted text; a reply starting with CALL becomes one Bash tool call."""

    def tool_calls(self, text, tools):
        if not text.startswith('CALL'):
            return None
        return [dict(id='call_1', type='function',
                     function=dict(name='Bash', arguments=json.dumps({'command': 'ls -la'})))], 'Listing files.'


class ConversionTest(unittest.TestCase):
    def test_request_to_openai(self):
        body = {'model': 'claude-x', 'max_tokens': 64, 'system': [{'type': 'text', 'text': 'Be brief.'}],
                'stop_sequences': ['END'],
                'tools': [{'name': 'Bash', 'description': 'Run', 'input_schema': {'type': 'object'}},
                          {'type': 'web_search_20250305', 'name': 'web_search'}],
                'messages': [
                    {'role': 'user', 'content': 'list files'},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Sure.'},
                                                      {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash',
                                                       'input': {'command': 'ls'}}]},
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1',
                                                  'content': [{'type': 'text', 'text': 'a.txt'}]},
                                                 {'type': 'text', 'text': 'now count them'},
                                                 {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                                              'data': 'AA=='}}]}]}
        parsed = ap.parse_messages_request(body)
        roles = [m['role'] for m in parsed['messages']]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'tool', 'user'])
        self.assertEqual(parsed['messages'][0]['content'], 'Be brief.')
        self.assertEqual(parsed['messages'][2]['tool_calls'][0]['function'],
                         {'name': 'Bash', 'arguments': {'command': 'ls'}})
        self.assertEqual(parsed['messages'][3], {'role': 'tool', 'content': 'a.txt', 'tool_call_id': 'toolu_1'})
        self.assertEqual(parsed['messages'][4]['content'][1], {'type': 'image', 'source': 'data:image/png;base64,AA=='})
        self.assertEqual([t['function']['name'] for t in parsed['tools']], ['Bash'])
        self.assertEqual(parsed['stop'], ['END'])
        self.assertEqual(parsed['max_tokens'], 64)

    def test_system_messages_inside_the_conversation(self):
        parsed = ap.parse_messages_request({'max_tokens': 5, 'messages': [
            {'role': 'user', 'content': 'hi'}, {'role': 'system', 'content': [{'type': 'text', 'text': 'Reminder.'}]},
            {'role': 'user', 'content': 'go'}]})
        self.assertEqual([m['role'] for m in parsed['messages']], ['user', 'system', 'user'])

    def test_thinking_ignored_unless_enabled(self):
        body = {'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'hi'}], 'thinking': {'type': 'adaptive'}}
        self.assertIsNone(ap.parse_messages_request(body)['thinking'])
        ap.ANTHROPIC_THINKING = True
        try:
            self.assertEqual(ap.parse_messages_request(body)['thinking'], 'max')
        finally:
            ap.ANTHROPIC_THINKING = False

    def test_billing_header_is_dropped(self):
        body = lambda cch: {'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'hi'}], 'system': [
            {'type': 'text', 'text': f'x-anthropic-billing-header: cc_version=2.1.3; cch={cch};'},
            {'type': 'text', 'text': 'You are Claude Code.'}]}
        a, b = ap.parse_messages_request(body('a1')), ap.parse_messages_request(body('b2'))
        self.assertEqual(a['messages'][0]['content'], 'You are Claude Code.')
        self.assertEqual(a['messages'], b['messages'])

    def test_response_to_anthropic(self):
        response = dict(choices=[dict(message=dict(role='assistant', content='Listing.', tool_calls=[
            dict(id='call_9', type='function', function=dict(name='Bash', arguments='{"command": "ls"}'))]),
            finish_reason='tool_calls')], usage=dict(prompt_tokens=100, completion_tokens=7,
                                                      prompt_tokens_details=dict(cached_tokens=60)))
        message = ap.to_anthropic(response, 'm')
        self.assertEqual(message['stop_reason'], 'tool_use')
        self.assertEqual(message['content'][0], {'type': 'text', 'text': 'Listing.'})
        self.assertEqual(message['content'][1], {'type': 'tool_use', 'id': 'toolu_call_9', 'name': 'Bash',
                                                 'input': {'command': 'ls'}})
        self.assertEqual(message['usage'], {'input_tokens': 40, 'cache_read_input_tokens': 60,
                                            'cache_creation_input_tokens': 0, 'output_tokens': 7})


class MessagesEndpointTest(unittest.TestCase):
    def start(self, reply, api_key=None):
        self.harness = ServerHarness(ToolBackend(reply), api_key=api_key)
        self.addCleanup(self.harness.close)

    def post(self, path, body, headers=None):
        return self.harness.request('POST', path, body, headers)

    def body(self, **extra):
        return dict(model='claude-sonnet', max_tokens=50, messages=[{'role': 'user', 'content': 'hi'}],
                    tools=[{'name': 'Bash', 'input_schema': {'type': 'object'}}], **extra)

    def test_text_reply(self):
        self.start('Hello there')
        status, _, payload = self.post('/v1/messages', self.body())
        self.assertEqual(status, 200)
        message = json.loads(payload)
        self.assertEqual(message['type'], 'message')
        self.assertEqual(message['model'], 'claude-sonnet')
        self.assertEqual(message['content'], [{'type': 'text', 'text': 'Hello there'}])
        self.assertEqual(message['stop_reason'], 'end_turn')

    def test_tool_use_reply(self):
        self.start('CALL ls')
        message = json.loads(self.post('/v1/messages', self.body())[2])
        self.assertEqual(message['stop_reason'], 'tool_use')
        self.assertEqual(message['content'][1]['input'], {'command': 'ls -la'})

    def test_streaming_events(self):
        self.start('CALL ls')
        status, kind, payload = self.post('/v1/messages', self.body(stream=True))
        self.assertEqual(status, 200)
        self.assertIn('text/event-stream', kind)
        events = [line[7:] for line in payload.decode().splitlines() if line.startswith('event: ')]
        self.assertEqual(events[0], 'message_start')
        self.assertEqual(events[-2:], ['message_delta', 'message_stop'])
        self.assertIn('input_json_delta', payload.decode())
        delta = [json.loads(line[6:]) for line in payload.decode().splitlines()
                 if line.startswith('data: ') and 'message_delta' in line][0]
        self.assertEqual(delta['delta']['stop_reason'], 'tool_use')

    def test_count_tokens_and_errors(self):
        self.start('x')
        status, _, payload = self.post('/v1/messages/count_tokens', {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual(status, 200)
        self.assertGreater(json.loads(payload)['input_tokens'], 0)
        status, _, payload = self.post('/v1/messages', {'model': 'm', 'max_tokens': 5, 'messages': [{'role': 'robot', 'content': 'x'}]})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)['type'], 'error')

    def test_context_error_uses_anthropic_wording(self):
        self.harness = ServerHarness(ToolBackend('x', max_context=1024))
        self.addCleanup(self.harness.close)
        status, _, payload = self.post('/v1/messages', dict(self.body(), messages=[{'role': 'user', 'content': 'y' * 2000}]))
        self.assertEqual(status, 400)
        message = json.loads(payload)['error']['message']
        self.assertRegex(message, r'^prompt is too long: \d+ tokens > 1024 maximum$')

    def test_x_api_key(self):
        self.start('ok', api_key='secret')
        self.assertEqual(self.post('/v1/messages', self.body())[0], 401)
        self.assertEqual(self.post('/v1/messages', self.body(), {'x-api-key': 'secret'})[0], 200)


if __name__ == '__main__':
    unittest.main()
