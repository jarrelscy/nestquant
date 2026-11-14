"""Block-drafter speculative decoding (DFlash2, DSpark) with the full resident base and 4-bit refinements.

mlx-vlm 0.7.12 ships generic `dflash2` and `dspark` drafters and the `_dflash_rounds` loop, but its glm5_next
model lacks the hook those drafters need: target hidden states captured at fixed layers. This module adds it,
following oMLX's GLM adapter (the output of layer i, 0-indexed, with the hyper-connection streams averaged).
Rejected drafts are rolled back by mlx-vlm's generic cache transaction (`verify_forward`), the same path
that already serves glm5_next MTP verification.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import torch
from mlx_vlm.models.glm5_next import language as g5
from mlx_vlm.models.glm5_next.language import create_ssm_mask
from mlx_vlm.speculative import dflash as dflash_loop
from mlx_vlm.speculative.drafters import compatibility, load_drafter

import native_generate as ng
from metrics import MemoryGuard
from native_refinements import NativeHotRefinements


def _install_capture():
    text_cls = g5.Glm5NextModel if hasattr(g5, 'Glm5NextModel') else None
    if text_cls is None:
        text_cls = next(value for name, value in vars(g5).items()
                        if isinstance(value, type) and hasattr(value, '__call__') and name.endswith('TextModel'))
    if getattr(text_cls, '_nq_capture', False):
        return
    original_text = text_cls.__call__

    def text_call(self, inputs, inputs_embeds=None, cache=None, attention_mask=None, hidden_sink=None):
        request = getattr(self, '_nq_capture_request', None)
        if request is None:
            return original_text(self, inputs, inputs_embeds, cache, attention_mask, hidden_sink=hidden_sink)
        wanted, sink = request[:2]
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if attention_mask is not None:
            attention_mask = attention_mask.astype(mx.bool_)
            if attention_mask.shape[-1] != h.shape[1]:
                attention_mask = attention_mask[..., -h.shape[1]:]
        h = mx.repeat(h[:, :, None], self.config.hc_mult, axis=2)
        topk = None
        captured = {}
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            layer_mask = attention_mask
            if layer.block_type == 'linear_attention' and layer_cache is not None:
                layer_mask = create_ssm_mask(h[:, :, 0], layer_cache)
            h, topk = layer(h, layer_mask, layer_cache, topk)
            if index in wanted:
                captured[index] = _collapse(h)
        sink.extend(captured[index] for index in wanted)
        h = self.norm(h.mean(axis=2))
        if hidden_sink is not None:
            hidden_sink.append(h)
        return h

    text_cls.__call__ = text_call
    text_cls._nq_capture = True

    lm_cls = g5.LanguageModel
    original_lm = lm_cls.__call__

    def lm_call(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        ids = kwargs.pop('capture_layer_ids', None)
        if ids:
            ids = [index + CAPTURE_SHIFT for index in ids]
        if not ids:
            return original_lm(self, inputs, cache, inputs_embeds, **kwargs)
        sink = []
        self.model._nq_capture_request = (list(ids), sink, _collapse)
        try:
            out = original_lm(self, inputs, cache, inputs_embeds, **kwargs)
        finally:
            self.model._nq_capture_request = None
        out.hidden_states = sink
        return out

    lm_cls.__call__ = lm_call


def _relax_compatibility():
    # The drafters require a model-specific rollback method; the generic cache transaction covers glm5_next.
    original = compatibility.validate_dflash_target

    def relaxed(config, target_model, algorithm):
        try:
            original(config, target_model, algorithm)
        except ValueError as error:
            if 'rollback' not in str(error):
                raise
    for module_name in ('mlx_vlm.speculative.drafters.dflash2.dflash2', 'mlx_vlm.speculative.drafters.dspark.dspark'):
        module = __import__(module_name, fromlist=['validate_dflash_target'])
        module.validate_dflash_target = relaxed
    # Speculators-format DSpark checkpoints carry a copy of the target's output head. The drafter binds the
    # target's own head, so drop the copy instead of failing on an unexpected parameter.
    from mlx_vlm.speculative.drafters.dspark.dspark import DSparkDraftModel
    if not getattr(DSparkDraftModel, '_nq_drop_head', False):
        original_sanitize = DSparkDraftModel.sanitize

        def sanitize(self, weights):
            return original_sanitize(self, {key: value for key, value in weights.items()
                                            if key.removeprefix('model.') != 'lm_head.weight'})

        DSparkDraftModel.sanitize = sanitize
        DSparkDraftModel._nq_drop_head = True


def position_acceptance(accepts, drafts):
    """Fraction of rounds whose k-th draft (1-based) was accepted, among rounds that drafted at least k."""
    depth = max(drafts, default=0)
    return [round(sum(a >= k for a, d in zip(accepts, drafts) if d >= k) / max(sum(d >= k for d in drafts), 1), 3)
            for k in range(1, depth + 1)]


def layer_major_prefill_capture(model, chunks, cache, guard, hot, wanted):
    hidden = [mx.repeat(model.model.embed_tokens(mx.array([chunk]))[:, :, None],
                        model.config.hc_mult, axis=2) for chunk in chunks]
    indices = [None] * len(chunks)
    captured = {}
    for number, (layer, state) in enumerate(zip(model.layers, cache, strict=True)):
        for i, value in enumerate(hidden):
            hot.chunk_index = i
            mask = create_ssm_mask(value[:, :, 0], state) if layer.block_type == 'linear_attention' else None
            hidden[i], indices[i] = layer(value, mask, state, indices[i])
            mx.eval(hidden[i], indices[i], state.state)
            guard.check()
        if number in wanted:
            captured[number] = mx.concatenate([_collapse(value) for value in hidden], axis=1)
            mx.eval(captured[number])
    last = model.model.norm(hidden[-1][:, -1:].mean(axis=2))
    logits = model.model.embed_tokens.as_linear(last) if model.config.tie_word_embeddings else model.lm_head(last)
    mx.eval(logits)
    for i, chunk in enumerate(chunks):
        hot.commit(chunk, i)
    return logits, mx.concatenate([captured[index] for index in wanted], axis=-1)


CAPTURE_SHIFT = 0   # -1 captures the input of layer i (vLLM's EAGLE-3/Speculators convention)
CAPTURE_STREAMS = 'mean'   # or 'first': hyper-connection stream 0 only


def _collapse(h):
    return h[:, :, 0] if CAPTURE_STREAMS == 'first' else h.mean(axis=2)


def target_layers(drafter):
    config = drafter.config
    for name in ('target_layer_ids', 'aux_hidden_state_layer_ids'):
        if getattr(config, name, None):
            return [index + CAPTURE_SHIFT for index in getattr(config, name)]
    raise ValueError('Drafter does not name its target layers')


BATCHED = None   # native_prefill.BatchedPrefill when --batched-prefill is set
# Reordered round loop (fast_rounds in dflash_decode); NQ_DFLASH_LOOP=mlx selects mlx-vlm's generator loop.
# Refinement reader threads (SSD reads are latency bound; 4 threads delivered ~0.35 GiB/s each).
READERS = int(__import__('os').environ.get('NQ_REFINEMENT_READERS', '8'))
# Decode refinement reads in flight (hot_policy.DEFAULTS['pending'], env NQ_REFINEMENT_PENDING).
PENDING = __import__('hot_policy').settings()['pending']
GC_GENERATION = int(__import__('os').environ.get('NQ_GC_GENERATION', '1'))
FAST_LOOP = __import__('os').environ.get('NQ_DFLASH_LOOP', 'fast') != 'mlx'
# NQ_ROUND_TIMING=1 records host timestamps per round; =2 also waits for each draft to measure its GPU time.
ROUND_TIMING = int(__import__('os').environ.get('NQ_ROUND_TIMING', '0'))
LAST_ROUND_TIMING = []
# Draft pruning (default; NQ_DRAFT_PRUNE=0 disables): draft the full block, then verify only the prefix that maximises expected committed tokens
# per millisecond given the drafter's own per-token confidence (round cost = PRUNE_FIXED_MS + PRUNE_ROW_MS * rows).
DRAFT_PRUNE = __import__('os').environ.get('NQ_DRAFT_PRUNE', '1') != '0'
PRUNE_FIXED_MS = float(__import__('os').environ.get('NQ_PRUNE_FIXED_MS', '26'))
PRUNE_ROW_MS = float(__import__('os').environ.get('NQ_PRUNE_ROW_MS', '20'))
PRUNE_POWER = float(__import__('os').environ.get('NQ_PRUNE_POWER', '1'))
PRUNE_LOG = []      # (confidences, verified draft tokens, accepted) per round
# Measured acceptance of a reached draft token by drafter confidence (DFlash2 on GLM-5.3-Flash, 6 coding prompts,
# 493 rounds): the drafter is overconfident between 0.4 and 0.99. NQ_PRUNE_CALIBRATION=raw uses the confidence as is.
PRUNE_CALIBRATION = __import__('os').environ.get('NQ_PRUNE_CALIBRATION', 'table')
CALIBRATION_TABLE = ((0.99, 0.99), (0.95, 0.75), (0.90, 0.73), (0.80, 0.66), (0.70, 0.54), (0.60, 0.40), (0.0, 0.30))


def calibrated(p):
    if PRUNE_CALIBRATION == 'raw':
        return min(1.0, max(0.0, p))
    return next(value for floor, value in CALIBRATION_TABLE if p >= floor)


class ConfidenceSampler:
    """Greedy sampler that also keeps the drafter's probability for each proposed token."""

    def __init__(self):
        self.probs = []

    def __call__(self, value):
        return value.argmax(axis=-1)

    def sample_proposal(self, scores):
        self.probs.append(mx.softmax(scores.astype(mx.float32), axis=-1).max(axis=-1))
        return mx.argmax(scores, axis=-1)


def prune_length(confidences, fixed_ms=None, row_ms=None, power=None):
    """Number of draft tokens to verify (at least 1) that maximises expected tokens per millisecond."""
    fixed_ms = PRUNE_FIXED_MS if fixed_ms is None else fixed_ms
    row_ms = PRUNE_ROW_MS if row_ms is None else row_ms
    power = PRUNE_POWER if power is None else power
    best, best_rate, expected, survive = 1, 0.0, 1.0, 1.0
    for k, p in enumerate(confidences, 1):
        survive *= calibrated(p) ** power
        expected += survive
        rate = expected / (fixed_ms + row_ms * (k + 1))
        if rate > best_rate:
            best, best_rate = k, rate
    return best


class AdaptiveBlocks:
    """Pick each round's verify block to maximize expected committed tokens per millisecond.

    Round cost is modelled as fixed + per_row * block (fitted from measured verify rounds). Acceptance is
    tracked per draft position as the probability of accepting position j given position j-1 was accepted,
    with exponentially decaying counts so the policy follows the content (copying code vs new prose).
    """

    def __init__(self, ceiling=8, fixed_ms=18.0, per_row_ms=22.5, decay=0.7, prior=0.8, prior_weight=0.0):
        self.ceiling, self.fixed_ms, self.per_row_ms, self.decay = ceiling, fixed_ms, per_row_ms, decay
        self.trials = [prior_weight] * ceiling
        self.hits = [prior * prior_weight] * ceiling
        self.seen = 0
        self.chosen = []
        self.timings = []    # (block, round seconds) for the online cost fit

    def record_round(self, block, seconds):
        self.timings.append((block, seconds * 1000))
        recent = self.timings[-32:]
        blocks = {b for b, _ in recent}
        if len(recent) >= 6 and len(blocks) >= 2:
            n = len(recent)
            mean_b = sum(b for b, _ in recent) / n
            mean_t = sum(t for _, t in recent) / n
            var = sum((b - mean_b) ** 2 for b, _ in recent)
            slope = sum((b - mean_b) * (t - mean_t) for b, t in recent) / var
            if slope > 1:
                self.per_row_ms = slope
                self.fixed_ms = max(1.0, mean_t - slope * mean_b)

    def observe(self, accept_lens, draft_lens):
        for accepted, drafted in list(zip(accept_lens, draft_lens))[self.seen:]:
            self.trials = [t * self.decay for t in self.trials]
            self.hits = [h * self.decay for h in self.hits]
            for j in range(int(drafted)):
                if accepted >= j:
                    self.trials[j] += 1.0
                    if accepted >= j + 1:
                        self.hits[j] += 1.0
            self.seen += 1

    def choose(self, draft_model, block_total, remaining):
        accepts, drafts = draft_model.accept_lens, getattr(draft_model, 'draft_lens', [])
        self.observe(accepts, drafts)
        top = min(self.ceiling, block_total, remaining)
        if top < 2:
            return top
        # Positions not tried recently inherit the deepest observed estimate (or an optimistic prior), so
        # the policy can grow back after a run of small blocks.
        estimate, inherited = [], 0.8
        for j in range(self.ceiling):
            if self.trials[j] >= 0.5:
                inherited = self.hits[j] / self.trials[j]
            estimate.append(inherited)
        best, best_rate = 2, 0.0
        expected, survive = 1.0, 1.0
        for block in range(2, top + 1):
            survive *= estimate[block - 2]
            expected += survive
            rate = expected / (self.fixed_ms + self.per_row_ms * block)
            if rate > best_rate:
                best, best_rate = block, rate
        if accepts and drafts and accepts[-1] >= drafts[-1]:
            best = max(best, min(top, int(drafts[-1]) + 1 + 1))   # every draft accepted: grow
        if len(self.chosen) % 16 == 15:
            best = top                                            # periodic probe of the full block
        self.chosen.append(best)
        return best


ADAPTIVE = None


FAST_SUFFIX_MAX = int(__import__('os').environ.get('NQ_FAST_SUFFIX_MAX', '256'))
FAST_CHUNK = int(__import__('os').environ.get('NQ_FAST_CHUNK', '256'))
HOT_BATCHED_MAX = int(__import__('os').environ.get('NQ_HOT_BATCHED_MAX', '8192'))
LAST_PREFILL_PATH = None


def hot_set_prefill(model, drafter, ids, cache, hot, guard, chunk_size=FAST_CHUNK):
    """Prefill a short prompt or suffix through the ordinary multi-token forward with the hot set's 4-bit
    refinements already in memory (decode precision), instead of the batched path that streams every routed
    expert's refinement from SSD. Commits each chunk's routing to the predictor and captures drafter states.
    """
    wanted = target_layers(drafter)
    hot.phase = 'decode'
    hot.prefill_bank = None
    hot.chunk_index = 0
    hot.traces.clear()
    hidden_parts, logits = [], None
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        hot.plan(chunk[0])
        hot.publish()
        out = model(mx.array([chunk]), cache=cache, capture_layer_ids=wanted)
        logits = out.logits[:, -1:]
        part = mx.concatenate(out.hidden_states, axis=-1)
        mx.eval(logits, part, [entry.state for entry in cache])
        hidden_parts.append(part)
        trace = hot.take_trace()
        trace = {layer: tuple(np.asarray(value) for value in entry) for layer, entry in trace.items()}
        hot.commit(chunk, trace=trace, rows=len(chunk))
        guard.check()
    hidden = hidden_parts[0] if len(hidden_parts) == 1 else mx.concatenate(hidden_parts, axis=1)
    mx.eval(hidden)
    # Every chunk is committed, so the state is a clean prefill boundary (decode re-plans its first token).
    hot.phase = 'prefill'
    hot.traces.clear()
    if hot.predictor.ahead is not None or hot.position != hot.predictor.t:
        raise RuntimeError('Hot-set prefill left the predictor ahead of the committed tokens')
    return logits, hidden


SEED_PROMPT = __import__('os').environ.get('NQ_SEED_PROMPT', '0') == '1'
SEED_WEIGHT = float(__import__('os').environ.get('NQ_SEED_WEIGHT', '1.0'))


def seed_hot_from_prompt(hot, salience, weight=SEED_WEIGHT):
    """Seed each layer's hot pool from whole-prompt routing, as Spark's policy allows for prompt prefills
    (hot_policy.seed_from_prompt). Off by default: on recorded coding traces it lowered the 4-bit share."""
    from hot_policy import seed_from_prompt
    return seed_from_prompt(hot.predictor, salience, weight)


def refinement_supply(hot, decode_seconds):
    """Separate prediction misses from delivery misses for the hot set during decode."""
    metrics = getattr(hot, 'metrics', None)
    fractions = metrics.report()['call_fractions'] if metrics is not None else {}
    wanted = fractions.get('wanted', 0.0)
    delivered = fractions.get('wanted_ready', 0.0)
    read_bytes = hot.reads.get('decode_bytes', 0)
    return dict(wanted_call_fraction=wanted, wanted_and_ready_call_fraction=delivered,
                late_call_fraction=fractions.get('late', 0.0),
                cached_outside_wanted_fraction=fractions.get('cached_outside_wanted', 0.0),
                delivery_fraction=delivered / wanted if wanted else None,
                decode_read_gib=read_bytes / 2**30,
                decode_read_gib_per_s=read_bytes / 2**30 / decode_seconds if decode_seconds else None,
                reader_busy_seconds=hot.reads.get('decode_worker_seconds', 0.0),
                verify_wait_seconds=hot.reads.get('verify_wait_seconds', 0.0),
                delivery=getattr(hot, 'delivery', None), decode_reads=hot.reads.get('decode_reads'),
                evicted_ahead=hot.reads.get('evicted_ahead', 0),
                install_copies=hot.reads.get('install_copies', 0),
                install_copy_ms=(1000 * hot.reads['install_copy_seconds'] / hot.reads['install_copies']
                                 if hot.reads.get('install_copies') else None),
                installs=hot.reads.get('installed', 0), discarded=hot.reads.get('discarded', 0),
                reader=hot.reader_info() if hasattr(hot, 'reader_info') else None)


def capture_prefill(model, drafter, ids, cache, hot, guard, suffix=False):
    """Prefill `ids` on top of whatever `cache` already holds, capturing the drafter's target hidden states.

    Short prompts and suffixes (at most FAST_SUFFIX_MAX tokens) use the hot-set path; longer ones use the
    batched path with every routed expert at 4 bits. Returns (logits of the last token, captured hidden rows
    (1, len(ids), layers * hidden)).
    """
    global LAST_PREFILL_PATH
    wanted = target_layers(drafter)
    if 0 < len(ids) <= FAST_SUFFIX_MAX:
        LAST_PREFILL_PATH = 'hot_set'
        return hot_set_prefill(model, drafter, ids, cache, hot, guard)
    stream = not (suffix and len(ids) <= HOT_BATCHED_MAX)
    LAST_PREFILL_PATH = ('batched' if stream else 'batched_hot') if BATCHED is not None else 'layer_major'
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    if BATCHED is not None:
        BATCHED.stream_refinements = stream
        try:
            logits = BATCHED(model, chunks, cache, guard, hot, capture=(wanted, _collapse))
        finally:
            BATCHED.stream_refinements = True
        if getattr(hot, 'record', None) is not None and getattr(BATCHED, 'prompt_salience', None):
            hot._record('prompt_salience', {layer: np.asarray(value) for layer, value in BATCHED.prompt_salience.items()})
        if SEED_PROMPT and not suffix and getattr(BATCHED, 'prompt_salience', None):
            seed_hot_from_prompt(hot, BATCHED.prompt_salience)
        hidden = mx.concatenate([BATCHED.captured[index] for index in wanted], axis=-1)
        BATCHED.captured = {}
        mx.eval(hidden)
    else:
        logits, hidden = layer_major_prefill_capture(model, chunks, cache, guard, hot, wanted)
    return logits, hidden


def dflash_decode(model, drafter, cache, hidden, first, max_tokens, eos, hot, guard, block_size,
                  should_stop=None, on_token=None, record_hidden=False):
    """DFlash rounds after a prefill whose last-token logits gave `first`.

    `hidden` holds the target hidden rows of the context the drafter should see (the drafters use a 2,047-row
    sliding window, so the last rows suffice). Generation stops at an EOS token, at max_tokens, or when
    should_stop(generated_ids) returns True; tokens of the current round after the stop point are still
    consumed so that the target cache, the predictor and the returned bookkeeping end on a round boundary.

    Returns a dict with:
      generated_ids  -- output tokens, ending with the stop/EOS token if one was produced
      committed_ids  -- tokens fed through the target cache by the rounds, in order (first .. accepted drafts)
      committed_hidden -- (record_hidden) target hidden rows for committed_ids
      stopped        -- 'eos', 'stop' or None (length)
    """
    hot.start_pipelined_decode()
    if getattr(hot, 'record', None) is not None:
        hot._record('decode_start', time.perf_counter())
    hot.plan(first)
    hot.publish()
    generated, current, previous = [first], [], first
    committed_ids, committed_hidden = [], []
    output_count = None
    stopped = None
    if first in eos:
        output_count, stopped = 1, 'eos'
    elif should_stop is not None and should_stop(generated):
        output_count, stopped = 1, 'stop'
    if on_token is not None:
        on_token(generated[:output_count] if output_count else generated)
    rounds_seen, round_times, unique_rows = 0, [], []
    from concurrent.futures import ThreadPoolExecutor
    hook_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nestquant-dflash-hooks')
    pending_hook = None
    hook_wait = 0.0
    between = dict(trace=0.0, publish=0.0, guard=0.0, generator_gap=0.0)
    last_yield = None
    gc_enabled = gc.isenabled()
    gc.disable()
    start = round_start = time.perf_counter()
    process_cpu_start = time.process_time()
    predictor_start = (getattr(hot, 'seconds', 0.0), getattr(hot, 'cpu_seconds', 0.0))
    original_next = dflash_loop._dflash_next_block_size
    original_verify = dflash_loop._dflash_verify_greedy
    last_captured = []

    def recording_verify(*args, **kwargs):
        captured, states, tokens = original_verify(*args, **kwargs)
        last_captured[:] = [captured]
        return captured, states, tokens

    global ADAPTIVE
    adaptive = None
    if block_size < 0:
        adaptive = ADAPTIVE = AdaptiveBlocks(ceiling=-block_size)
        dflash_loop._dflash_next_block_size = (
            lambda draft_model, total, remaining, initial=None: adaptive.choose(draft_model, total, remaining))
        block_size = -block_size
    if record_hidden:
        dflash_loop._dflash_verify_greedy = recording_verify

    def finish_round(accepted, committed=None):
        # The round fed [previous] + its first `accepted` drafts through the target cache.
        nonlocal previous, current, pending_hook, hook_wait, round_start, rounds_seen
        rounds_seen += 1
        now = time.perf_counter()
        round_times.append(now - round_start)
        if adaptive is not None and getattr(drafter, 'draft_lens', None):
            adaptive.record_round(int(drafter.draft_lens[-1]) + 1, now - round_start)
        if committed is None:
            committed = ([previous] + current)[:accepted + 1]
        committed_ids.extend(committed)
        if record_hidden and last_captured:
            rows = mx.concatenate(last_captured[0], axis=-1)[:, :len(committed)]
            mx.async_eval(rows)
            committed_hidden.append(rows)
        t_trace = time.perf_counter()
        trace = hot.take_trace()
        mx.eval([value for entry in trace.values() for value in entry])
        trace = {layer: tuple(np.asarray(value) for value in entry) for layer, entry in trace.items()}
        ids_rows = [entry[0].reshape(-1) for entry in trace.values()]
        unique_rows.append(sum(len(set(row.tolist())) for row in ids_rows) / len(ids_rows))
        between['trace'] += time.perf_counter() - t_trace
        previous, current = (current[-1] if current else previous), []
        # Commit and plan touch only host state, so they overlap the next round's draft and verify.
        # Finished refinement reads are installed here, after the previous hook has completed.
        if pending_hook is not None:
            waited = time.perf_counter()
            pending_hook.result()
            hook_wait += time.perf_counter() - waited
        t_publish = time.perf_counter()
        hot.publish()
        between['publish'] += time.perf_counter() - t_publish

        def hook(committed=committed, trace=trace, previous=previous):
            hot.commit(committed, trace=trace, rows=len(committed))
            hot.plan(previous)

        pending_hook = hook_pool.submit(hook)
        t_guard = time.perf_counter()
        guard.check()
        between['guard'] += time.perf_counter() - t_guard
        round_start = time.perf_counter()

    def emit(token):
        # Record one output token; returns True once generation should stop.
        nonlocal output_count, stopped
        generated.append(token)
        current.append(token)
        if stopped is None:
            if token in eos:
                output_count, stopped = len(generated), 'eos'
            elif should_stop is not None and should_stop(generated):
                output_count, stopped = len(generated), 'stop'
            if on_token is not None:
                on_token(generated)
        return stopped is not None

    def fast_rounds():
        """mlx-vlm's greedy `_dflash_rounds` with the same drafts, verifies and cache transactions, reordered
        so the GPU is not idle while the host does round bookkeeping: the next draft is launched as soon as
        the walk has found the bonus token, and the cache commit, token emission (streaming, stop checks),
        trace readback, refinement publication and predictor hook run while the drafter executes. The
        verify of the next round is then encoded behind it."""
        nonlocal rounds_seen
        lm = model.language_model if hasattr(model, 'language_model') else model
        target_ids = list(drafter.config.target_layer_ids)
        block_total = dflash_loop._dflash_block_total(drafter, block_size)
        choose_ceiling = getattr(drafter, 'choose_block_ceiling', None)
        if callable(choose_ceiling):
            block_total = choose_ceiling(int(hidden.shape[1]), block_total)
        initial = getattr(drafter, 'dflash_initial_block_size', None)
        choose_initial = getattr(drafter, 'choose_initial_block_size', None)
        if callable(choose_initial):
            initial = choose_initial(int(hidden.shape[1]), block_total)
        draft_cache = drafter.reset(model)
        stream = dflash_loop.generation_stream
        with mx.stream(stream):
            dflash_loop._reserve_dflash_target_cache(cache, block_total)
        draft_fn = getattr(drafter, 'draft_block_greedy', drafter.draft_block)
        # Pruning needs the candidate selector's scores, so draft through draft_block (draft_block_greedy only
        # differs when the target provides its own argmax head).
        prune = DRAFT_PRUNE and not callable(getattr(drafter, 'argmax_from_hidden', None))
        if prune:
            draft_fn = drafter.draft_block
        sampler = ConfidenceSampler() if prune else (lambda value: value.argmax(axis=-1))
        confidence = [None]
        PRUNE_LOG.clear()

        def launch(bonus, context, emitted):
            if prune:
                bs = min(block_total, max_tokens - emitted + 1)
            else:
                bs = dflash_loop._dflash_next_block_size(drafter, block_total, max_tokens - emitted + 1, initial)
            if bs <= 1:
                return bs, None
            if prune:
                sampler.probs.clear()
            draft = draft_fn(bonus, context, draft_cache, bs, sampler, mx.int32)
            if prune and len(sampler.probs) == bs - 1:
                confidence[0] = mx.concatenate([value.reshape(-1) for value in sampler.probs])
                mx.async_eval(draft, confidence[0])
            else:
                confidence[0] = None
                mx.async_eval(draft)
            return bs, draft

        emitted, bonus, context = 1, first, hidden
        bs, draft = launch(bonus, context, emitted)
        LAST_ROUND_TIMING.clear()
        while draft is not None:
            state = None
            t_top = time.perf_counter()
            confidences = None
            if prune and confidence[0] is not None and bs > 2:
                confidences = confidence[0].tolist()
                keep = prune_length(confidences)
                if keep < bs - 1:
                    draft, bs = draft[:, :keep], keep + 1
            # Refinement copies started at the last publish that have finished by now join this verify.
            hot.flip_for_verify()
            try:
                with mx.stream(stream):
                    verify_input = mx.concatenate([mx.array([[bonus]], dtype=mx.int32), draft], axis=1)
                    captured, state, target = dflash_loop._dflash_verify_greedy(lm, verify_input, cache, target_ids,
                                                                                sampler)
                    verified_hidden = mx.concatenate(captured, axis=-1)
                # The routing trace feeds only the predictor; evaluating it with the verify keeps its readback
                # from waiting behind the next draft.
                t_built = time.perf_counter()
                trace_arrays = [value for entry in hot.traces.get(hot.chunk_index, {}).values() for value in entry]
                mx.async_eval(target, verified_hidden, trace_arrays)
                if ROUND_TIMING:
                    t_submit = time.perf_counter()
                    mx.eval(target)
                    t_ready = time.perf_counter()
                draft_row = draft.reshape(-1).tolist()
                accepted, new_tokens = dflash_loop._speculative_walk(draft, target, max_tokens - emitted)
                dflash_loop._record_speculative_round(drafter, accepted, bs - 1)
                if confidences is not None:
                    PRUNE_LOG.append((confidences, bs - 1, accepted))
                context = verified_hidden[:, :accepted + 1] if accepted < bs - 1 else verified_hidden
                committed = [bonus] + draft_row[:accepted]
                bonus = new_tokens[-1] if new_tokens else bonus
                after = emitted + len(new_tokens)
                ends = after >= max_tokens or any(token in eos for token in new_tokens)
                bs_next, draft = (None, None) if ends else launch(bonus, context, after)
                if ROUND_TIMING:
                    t_launch = time.perf_counter()
                    if ROUND_TIMING > 1 and draft is not None:
                        mx.eval(draft)
                    t_draft = time.perf_counter()
                with mx.stream(stream):
                    dflash_loop.commit_speculative_round(lm, cache, state, accepted, bs)
                state = None
            except BaseException:
                from mlx_vlm.speculative.cache_state import abort_speculative_round
                abort_speculative_round(state)
                raise
            if record_hidden:
                last_captured[:] = [[context]]
            stop = False
            for token in new_tokens:
                stop = emit(token) or stop
                if stop:
                    break
            emitted = after
            finish_round(accepted, committed)
            if ROUND_TIMING:
                LAST_ROUND_TIMING.append(dict(rows=bs, build=t_built - t_top, encode=t_submit - t_top, wait=t_ready - t_submit,
                                              walk_launch=t_launch - t_ready, draft_gpu=t_draft - t_launch,
                                              host_after=time.perf_counter() - t_draft))
            if stop:
                break
            bs = bs_next

    try:
        if stopped is None and max_tokens > 1 and FAST_LOOP and not hasattr(drafter, 'prepare_target_hidden'):
            rounds = True
            fast_rounds()
        elif stopped is None and max_tokens > 1:
            rounds = dflash_loop._dflash_rounds(model, drafter, cache, hidden, first_bonus=first,
                                                max_tokens=max_tokens, sampler=lambda value: value.argmax(axis=-1),
                                                draft_block_size=block_size, greedy_sampling=True)
            for token, _ in rounds:
                if last_yield is not None:
                    between['generator_gap'] += time.perf_counter() - last_yield
                token = int(token)
                emit(token)
                accepted = int(drafter.accept_lens[-1]) if len(drafter.accept_lens) > rounds_seen else -1
                if len(current) == accepted + 1:
                    finish_round(accepted)
                    if stopped is not None:
                        rounds.close()
                        break
                last_yield = time.perf_counter()
            if current and len(drafter.accept_lens) > rounds_seen:
                # max_tokens cut the last round's bonus token: its drafts are committed but unprocessed.
                finish_round(int(drafter.accept_lens[-1]))
        else:
            rounds = None
        mx.synchronize()
        if pending_hook is not None:
            pending_hook.result()
        decode_end = time.perf_counter()
        process_cpu = time.process_time() - process_cpu_start
    finally:
        hook_pool.shutdown(wait=True)
        dflash_loop._dflash_next_block_size = original_next
        dflash_loop._dflash_verify_greedy = original_verify
        if gc_enabled:
            gc.enable()
        # Young generations only: a full collection walks every long-lived object of the runtime (~0.8 s per
        # request). Older cycles are left to the interpreter's own full collections.
        gc.collect(GC_GENERATION)
    cleanup_seconds = time.perf_counter() - decode_end
    decode = decode_end - start
    hot.traces.clear()
    accepts = [float(a) for a in drafter.accept_lens] if rounds is not None else []
    drafts = [int(d) for d in getattr(drafter, 'draft_lens', [])] if rounds is not None else []
    output = generated[:output_count] if output_count else generated
    return dict(generated_ids=output, committed_ids=committed_ids,
                committed_hidden=committed_hidden if record_hidden else None, stopped=stopped,
                decode_seconds=decode,
                decode_tps=(len(output) - 1) / decode if len(output) > 1 and decode > 0 else None,
                rounds=len(accepts),
                mean_accepted=sum(accepts) / max(len(accepts), 1),
                mean_drafted=sum(drafts) / max(len(drafts), 1),
                position_acceptance=position_acceptance([int(a) for a in accepts], drafts),
                median_round_ms=1000 * sorted(round_times)[len(round_times) // 2] if round_times else None,
                mean_unique_experts=sum(unique_rows) / max(len(unique_rows), 1),
                hook_wait_seconds=hook_wait, round_seconds_total=sum(round_times),
                between_seconds={k: round(v, 3) for k, v in between.items()}, cleanup_seconds=cleanup_seconds,
                first_round_seconds=round_times[0] if round_times else None,
                adaptive_blocks=(adaptive.chosen if adaptive is not None else None),
                adaptive_cost_ms=((round(adaptive.fixed_ms, 1), round(adaptive.per_row_ms, 1))
                                  if adaptive is not None else None),
                refinement_installs=hot.reads['installed'], publication_seconds=hot.reads['decode_publication_seconds'],
                ready_call_fraction=hot.report()['coverage']['call_fractions']['ready'],
                refinement_supply=refinement_supply(hot, decode),
                predictor_backend=getattr(hot, 'predictor_backend', None), decode_process_cpu_seconds=process_cpu,
                decode_predictor_seconds=getattr(hot, 'seconds', 0.0) - predictor_start[0],
                decode_predictor_cpu_seconds=getattr(hot, 'cpu_seconds', 0.0) - predictor_start[1])


def block_generate(model, drafter, ids, max_tokens, eos, hot, guard, block_size):
    cache = model.make_cache()
    hot.reset()
    started = time.perf_counter()
    logits, hidden = capture_prefill(model, drafter, ids, cache, hot, guard)
    first = ng.greedy(logits)
    ttft = time.perf_counter() - started
    out = dflash_decode(model, drafter, cache, hidden, first, max_tokens, eos, hot, guard, block_size)
    out.pop('committed_hidden')
    out.pop('committed_ids')
    out.pop('stopped')
    out['ttft_seconds'] = ttft
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--drafter', type=Path, required=True)
    parser.add_argument('--hot-fraction', type=float, default=0.2)
    parser.add_argument('--budget-gib', type=int, default=22)
    parser.add_argument('--blocks', default='-8',
                        help='Comma list: 0 = plain decode, N = fixed verify block, -N = adaptive up to N (best)')
    parser.add_argument('--tokens', type=int, default=96)
    parser.add_argument('--prompt', default='Explain how a hash map works, in detail.')
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--prompt-dir', type=Path, help='Run every *.txt prompt in this directory')
    parser.add_argument('--batched-prefill', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-blocks', action='store_true', help='Verify the requested block every round')
    parser.add_argument('--capture-shift', type=int, default=0)
    parser.add_argument('--base-draft-variants', action='store_true',
                        help='Run each block twice: full verify, then draft rows at 1.5 bits')
    parser.add_argument('--draft-rows-base', action='store_true',
                        help='Verify draft rows with the 1.5-bit base only (faster; quality not yet validated)')
    parser.add_argument('--capture-streams', choices=('mean', 'first'), default='mean')
    parser.add_argument('--predictor-backend', default=None,
                        help="jT predictor: auto (default: coreml when available, else torch), torch, or coreml[:ane|:all]; env NQ_PREDICTOR_BACKEND")
    args = parser.parse_args()
    global CAPTURE_SHIFT
    CAPTURE_SHIFT = args.capture_shift
    global CAPTURE_STREAMS
    CAPTURE_STREAMS = args.capture_streams
    _install_capture()
    _relax_compatibility()
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(96 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, READERS, PENDING, 512,
                               predictor_backend=args.predictor_backend)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    global BATCHED
    if args.batched_prefill:
        from native_prefill import BatchedPrefill
        BATCHED = BatchedPrefill(args.model)
    results = []
    try:
        hot.warm(4, lambda: guard.check(force=True))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        if args.prompt_dir:
            prompts = [(path.stem, path.read_text()) for path in sorted(args.prompt_dir.glob('*.txt'))]
        else:
            prompts = [('prompt', args.prompt_file.read_text() if args.prompt_file else args.prompt)]
        encoded = [(name, tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                                        return_dict=False, add_generation_prompt=True,
                                                        chat_template=template)) for name, text in prompts]
        end = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
        eos = {end} if isinstance(end, int) else set(end)
        drafter = None
        plans = []
        for block in (int(value) for value in args.blocks.split(',')):
            plans.append((block, args.draft_rows_base and block != 0))
            if block and args.base_draft_variants:
                plans.append((block, True))
        for (prompt_name, ids), (block, base_drafts) in ((p, q) for p in encoded for q in plans):
            if (block, base_drafts) == plans[0]:
                reference = None
            hot.draft_rows_base = base_drafts
            if block == 0:
                out = ng.generate(model, tokenizer, ids, args.tokens, eos, BATCHED or ng.layer_major_prefill, 8, hot, guard)
                entry = dict(block=0, decode_tps=out['decode_tps'], ttft_seconds=out['ttft_seconds'],
                             generated_ids=out['generated_ids'])
            else:
                if drafter is None:
                    drafter, kind = load_drafter(str(args.drafter))
                    mx.eval(drafter.parameters())
                    print(json.dumps(dict(event='drafter_loaded', kind=kind, layers=target_layers(drafter),
                                          memory_gib=ng.process_usage().phys_footprint / 2**30)), flush=True)
                    guard.check(force=True)
                if args.fixed_blocks:
                    drafter.prefer_requested_block_size = True
                out = block_generate(model, drafter, ids, args.tokens, eos, hot, guard, block)
                entry = dict(block=block, **{k: v for k, v in out.items()})
            entry['draft_rows_base'] = base_drafts
            entry['prompt'] = prompt_name
            entry['prompt_tokens'] = len(ids)
            if block == 0 and reference is None:
                reference = entry['generated_ids']
            if reference is not None:
                n = min(len(reference), len(entry['generated_ids']))
                same = next((i for i in range(n) if reference[i] != entry['generated_ids'][i]), n)
                entry['tokens_matching_plain_prefix'] = same
                entry['tokens_matching_plain'] = sum(a == b for a, b in zip(reference, entry['generated_ids'])) / max(n, 1)
            entry['peak_gib'] = ng.process_usage().lifetime_max_phys_footprint / 2**30
            entry['text'] = tokenizer.decode(entry['generated_ids'][:60])
            results.append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != 'generated_ids'}), flush=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
