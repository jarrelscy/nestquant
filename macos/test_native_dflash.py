import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

import native_dflash as nd
from mlx_vlm.speculative import dflash as dflash_loop


class FakeDrafter:
    def __init__(self):
        self.config = SimpleNamespace(block_size=8)
        self.calls = []

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        self.calls.append(block_size)
        return mx.arange(block_size - 1, dtype=token_dtype)[None]


class PositionAcceptanceTest(unittest.TestCase):
    def test_counts_only_rounds_that_drafted_the_position(self):
        self.assertEqual(nd.position_acceptance([3, 1, 0, 2], [3, 3, 3, 2]), [0.75, 0.5, 0.333])


def truth(position):
    return (position * 7 + 3) % 50


class SequenceDrafter:
    """Proposes the true continuation of the position encoded in its last hidden row, wrong at every fifth
    position. Like the real drafter it sees only its own inputs, never the target cache."""

    def __init__(self):
        self.config = SimpleNamespace(block_size=8, target_layer_ids=[1])
        self.accept_lens, self.draft_lens, self.contexts = [], [], []

    def reset(self, model):
        self.accept_lens, self.draft_lens = [], []
        return []

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        bonus_position = int(hidden[0, -1, 0].item()) + 1
        self.contexts.append(hidden.shape[1])
        assert truth(bonus_position) == last_bonus
        drafts = [truth(p) if p % 5 else 49 - truth(p) for p in range(bonus_position + 1, bonus_position + block_size)]
        return mx.array([drafts], dtype=token_dtype)


class FakeHot:
    def __init__(self):
        self.traces, self.chunk_index, self.commits = {}, 0, []
        self.reads = dict(installed=0, decode_publication_seconds=0.0)

    def start_pipelined_decode(self):
        pass

    def plan(self, token):
        pass

    def publish(self):
        pass

    def flip(self, wait=False):
        pass

    def flip_for_verify(self):
        pass

    def take_trace(self, chunk_index=0):
        return self.traces.pop(chunk_index)

    def commit(self, tokens, trace=None, rows=None):
        self.commits.append(list(tokens))

    def report(self):
        return dict(coverage=dict(call_fractions=dict(ready=0.0)))


class FakeCache:
    def __init__(self, position):
        self.position = position    # position of the next token to feed (the bonus)


def run(loop, max_tokens, stop_at=None):
    prompt = 12
    cache, hot, drafter = FakeCache(prompt), FakeHot(), SequenceDrafter()
    hidden = mx.arange(prompt, dtype=mx.float32).reshape(1, prompt, 1)

    def verify(lm, inputs, prompt_cache, target_layer_ids, sampler):
        bs = inputs.shape[1]
        start = prompt_cache.position
        assert inputs[0, 0].item() == truth(start)
        hot.traces[0] = {3: (mx.arange(8 * bs, dtype=mx.uint32), mx.ones((bs, 8)), mx.ones((bs,)),
                             mx.zeros((8 * bs,), dtype=mx.uint32))}
        captured = [mx.arange(start, start + bs, dtype=mx.float32).reshape(1, bs, 1)]
        return captured, ('state', start, bs), mx.array([[truth(start + 1 + i) for i in range(bs)]])

    def commit(lm, prompt_cache, state, accepted, block_size):
        assert state[1] == prompt_cache.position and state[2] == block_size
        prompt_cache.position += accepted + 1

    should_stop = (lambda generated: stop_at in generated) if stop_at is not None else None
    with mock.patch.object(dflash_loop, '_dflash_verify_greedy', verify), \
            mock.patch.object(dflash_loop, 'commit_speculative_round', commit), \
            mock.patch.object(dflash_loop, '_reserve_dflash_target_cache', lambda *a: None), \
            mock.patch.object(dflash_loop, 'generation_stream', mx.default_stream(mx.cpu)), \
            mock.patch.object(nd, 'FAST_LOOP', loop == 'fast'):
        out = nd.dflash_decode(SimpleNamespace(), drafter, cache, hidden, truth(prompt), max_tokens, {999}, hot,
                               SimpleNamespace(check=lambda: None), -8, should_stop=should_stop, record_hidden=True)
    return out, cache, hot, drafter


class FastLoopTest(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.device)

    def test_matches_mlx_vlm_loop(self):
        for max_tokens in (2, 7, 40):
            fast, fast_cache, fast_hot, fast_drafter = run('fast', max_tokens)
            ref, ref_cache, ref_hot, ref_drafter = run('mlx', max_tokens)
            self.assertEqual(fast['generated_ids'], ref['generated_ids'])
            self.assertEqual(fast['generated_ids'], [truth(12 + i) for i in range(len(fast['generated_ids']))])
            self.assertEqual(len(fast['generated_ids']), max_tokens)
            self.assertEqual(fast_drafter.accept_lens, ref_drafter.accept_lens)
            self.assertEqual(fast_drafter.draft_lens, ref_drafter.draft_lens)
            self.assertEqual(fast_hot.commits, ref_hot.commits)
            # Every token the target cache retained is reported as committed, in order.
            self.assertEqual(fast['committed_ids'], [truth(p) for p in range(12, fast_cache.position)])
            self.assertEqual(sum(row.shape[1] for row in fast['committed_hidden']), fast_cache.position - 12)

    def test_stop_condition(self):
        target = truth(12 + 9)
        fast, *_ = run('fast', 40, stop_at=target)
        ref, *_ = run('mlx', 40, stop_at=target)
        self.assertEqual(fast['stopped'], 'stop')
        self.assertEqual(fast['generated_ids'], ref['generated_ids'])
        self.assertEqual(fast['generated_ids'][-1], target)
        self.assertEqual(fast['committed_ids'], ref['committed_ids'])


if __name__ == '__main__':
    unittest.main()
