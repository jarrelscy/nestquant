"""Verify request boundaries with the optional native serving dependencies installed."""

from email.message import Message
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from native_server import Handler, parse_request


class NativeServerTest(unittest.TestCase):
    def handler(self):
        handler = object.__new__(Handler)
        payload = json.dumps({'messages': [{'role': 'user', 'content': 'Hello'}]}).encode()
        handler.path = '/v1/chat/completions'
        handler.headers = Message()
        handler.headers['Host'] = '127.0.0.1:18085'
        handler.headers['Content-Type'] = 'application/json'
        handler.headers['Content-Length'] = str(len(payload))
        handler.rfile = io.BytesIO(payload)
        session = Mock()
        session.prepare.return_value = ([1, 2], [], 4)
        session.completion.return_value = {'choices': []}
        handler.server = SimpleNamespace(server_port=18085, session=session)
        handler.reply = Mock()
        handler.reply_bytes = Mock()
        return handler

    def test_runtime_errors_stop_instead_of_accepting_another_request(self):
        for error, code in ((RuntimeError('partial forward'), 1), (ValueError('bad logits'), 1),
                            (MemoryError('pressure'), 75)):
            with self.subTest(error=type(error).__name__):
                handler = self.handler()
                handler.server.session.completion.side_effect = error
                with self.assertRaises(SystemExit) as raised:
                    handler.do_POST()
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(handler.reply.call_args.args[0], 500)
                handler.reply_bytes.assert_not_called()

    def test_validation_error_leaves_model_untouched(self):
        handler = self.handler()
        handler.server.session.prepare.side_effect = ValueError('invalid template input')
        handler.do_POST()
        handler.server.session.completion.assert_not_called()
        self.assertEqual(handler.reply.call_args.args[0], 400)

    def test_disconnect_does_not_hide_runtime_failure(self):
        handler = self.handler()
        handler.server.session.completion.side_effect = RuntimeError('partial forward')
        handler.reply.side_effect = BrokenPipeError()
        with self.assertRaises(SystemExit):
            handler.do_POST()

    def test_disconnect_after_complete_generation_keeps_runtime_usable(self):
        handler = self.handler()
        handler.reply_bytes.side_effect = BrokenPipeError()
        handler.do_POST()
        handler.server.session.completion.assert_called_once_with(([1, 2], [], 4))
        handler.reply.assert_not_called()

    def test_invalid_result_stops_service(self):
        handler = self.handler()
        handler.server.session.completion.return_value = {'value': float('nan')}
        with self.assertRaises(SystemExit):
            handler.do_POST()

    def test_unsupported_generation_options_are_rejected(self):
        body = {'messages': [{'role': 'user', 'content': 'Hello'}]}
        for option in ({'stop': ['END']}, {'top_p': .8}, {'logprobs': True},
                       {'response_format': {'type': 'json_object'}}, {'frequency_penalty': 1},
                       {'presence_penalty': 1}, {'logit_bias': {'1': 100}}, {'seed': '42'}):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    parse_request({**body, **option})
        self.assertEqual(parse_request({**body, 'seed': 42, 'top_p': 1, 'stop': None})[2], 128)


if __name__ == '__main__':
    unittest.main()
