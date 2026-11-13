"""Greedy MTP decode with the full resident base and 4-bit refinements.

Wraps mlx_vlm's `_mtp_rounds`. Between rounds (after a round's last accepted token is yielded and
before the next draft starts) no GPU work that uses the refinement banks is queued, so this module
commits only the accepted verification rows to the jT predictor, plans the next round and publishes
finished refinement reads there.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import torch
from mlx_vlm.models.glm5_next.language import create_ssm_mask
from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
from mlx_vlm.speculative.drafters.glm5_next_mtp.split import Glm5NextMTPSplitter
from mlx_vlm.speculative.mtp import _mtp_rounds
from mlx_vlm.utils import load_model_weights

import native_generate as ng
from metrics import MemoryGuard
from native_refinements import NativeHotRefinements


def layer_major_prefill_hidden(model, chunks, cache, guard, hot):
    """layer_major_prefill that also returns the final normed hidden state at every prompt position."""
    hidden = [mx.repeat(model.model.embed_tokens(mx.array([chunk]))[:, :, None],
                        model.config.hc_mult, axis=2) for chunk in chunks]
    indices = [None] * len(chunks)
    for layer, state in zip(model.layers, cache, strict=True):
        for i, value in enumerate(hidden):
            if hot is not None:
                hot.chunk_index = i
            mask = create_ssm_mask(value[:, :, 0], state) if layer.block_type == 'linear_attention' else None
            hidden[i], indices[i] = layer(value, mask, state, indices[i])
            mx.eval(hidden[i], indices[i], state.state)
            guard.check()
    normed = mx.concatenate([model.model.norm(value.mean(axis=2)) for value in hidden], axis=1)
    last = normed[:, -1:]
    logits = model.model.embed_tokens.as_linear(last) if model.config.tie_word_embeddings else model.lm_head(last)
    mx.eval(logits, normed)
    if hot is not None:
        for i, chunk in enumerate(chunks):
            hot.commit(chunk, i)
    return logits, normed


def load_drafter(root, depth, bits):
    config, weights = Glm5NextMTPSplitter().prepare(root, block_size=depth + 1, q_bits=bits, q_group_size=64)
    draft = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict(config))
    drafter = load_model_weights(draft, config, weights, root)
    del weights
    mx.eval(drafter.parameters())
    return drafter


OVERLAP_HOOKS = True


PROFILE_ROUNDS = False


def mtp_generate(model, drafter, ids, max_tokens, eos, hot, guard, depth):
    import mlx_vlm.speculative.mtp as mtp_module
    phase_log = []
    if PROFILE_ROUNDS:
        original_verify = mtp_module._mtp_verify_target
        original_draft = drafter.draft_block

        def timed_draft(*args, **kwargs):
            started = time.perf_counter()
            out = original_draft(*args, **kwargs)
            mx.eval(out)
            phase_log.append(('draft', time.perf_counter() - started))
            return out

        def timed_verify(*args, **kwargs):
            started = time.perf_counter()
            out = original_verify(*args, **kwargs)
            mx.eval(out.hidden, out.target_tokens if out.target_tokens is not None else [])
            phase_log.append(('verify', time.perf_counter() - started))
            return out

        drafter.draft_block = timed_draft
        mtp_module._mtp_verify_target = timed_verify
    cache = model.make_cache()
    hot.reset()
    chunks = [ids[i:i + 8] for i in range(0, len(ids), 8)]
    started = time.perf_counter()
    logits, hidden = layer_major_prefill_hidden(model, chunks, cache, guard, hot)
    first = ng.greedy(logits)
    ttft = time.perf_counter() - started
    drafter.accept_lens.clear() if hasattr(drafter, 'accept_lens') else None
    drafter.draft_lens.clear() if hasattr(drafter, 'draft_lens') else None
    hot.start_pipelined_decode()
    hot.plan(first)
    hot.publish()
    generated = [first]
    previous = first
    current = []
    rounds_seen = 0
    gc_enabled = gc.isenabled()
    gc.disable()
    hook_seconds = 0.0
    round_seconds = []
    unique_rows, verify_rows = [], []
    from concurrent.futures import ThreadPoolExecutor
    hook_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nestquant-mtp-hooks') if OVERLAP_HOOKS else None
    pending_hook = None
    start = time.perf_counter()
    round_start = start
    try:
        rounds = _mtp_rounds(model, drafter, cache, hidden, {}, prompt_tokens=mx.array([ids]), first_bonus=first,
                             max_tokens=max_tokens, sampler=lambda value: value.argmax(axis=-1),
                             draft_block_size=depth + 1, greedy_sampling=True)
        for token, _ in rounds:
            token = int(token)
            generated.append(token)
            current.append(token)
            accepted = drafter.accept_lens[-1] if len(drafter.accept_lens) > rounds_seen else 0
            if len(current) == accepted + 1:
                # The round is complete: commit the verified rows for [previous bonus] + accepted drafts.
                rounds_seen += 1
                hook_start = time.perf_counter()
                round_seconds.append(hook_start - round_start)
                committed = [previous] + current[:-1]
                trace = hot.take_trace()
                # Convert on this thread: the hook thread must not create MLX operations.
                mx.eval([value for entry in trace.values() for value in entry])
                trace = {layer: tuple(np.asarray(value) for value in entry) for layer, entry in trace.items()}
                ids = [entry[0].reshape(-1) for entry in trace.values()]
                unique_rows.append(sum(len(set(row.tolist())) for row in ids) / len(ids))
                verify_rows.append(ids[0].size)
                previous = current[-1]
                current = []
                if hook_pool is None:
                    hot.commit(committed, trace=trace, rows=len(committed))
                    hot.plan(previous)
                    hot.publish()
                else:
                    # Commit and plan touch only host state, so they run while the next round drafts and
                    # verifies. Installs wait for the boundary after they finish.
                    if pending_hook is not None:
                        pending_hook.result()
                    hot.publish()

                    def hook(committed=committed, trace=trace, previous=previous):
                        hot.commit(committed, trace=trace, rows=len(committed))
                        hot.plan(previous)

                    pending_hook = hook_pool.submit(hook)
                guard.check()
                round_start = time.perf_counter()
                hook_seconds += round_start - hook_start
            if token in eos:
                rounds.close()
                break
        mx.synchronize()
        if pending_hook is not None:
            pending_hook.result()
    finally:
        if hook_pool is not None:
            hook_pool.shutdown(wait=True)
        if gc_enabled:
            gc.enable()
        gc.collect()
    decode = time.perf_counter() - start
    hot.traces.clear()
    if PROFILE_ROUNDS:
        mtp_module._mtp_verify_target = original_verify
        drafter.draft_block = original_draft
    return dict(generated_ids=generated, ttft_seconds=ttft, decode_seconds=decode,
                decode_tps=(len(generated) - 1) / decode, accept_lengths=list(drafter.accept_lens),
                mean_accepted=sum(drafter.accept_lens) / max(len(drafter.accept_lens), 1),
                hook_ms_per_round=1000 * hook_seconds / max(rounds_seen, 1),
                mean_unique_experts=sum(unique_rows) / max(len(unique_rows), 1),
                mean_verify_rows=sum(verify_rows) / max(len(verify_rows), 1),
                median_round_ms=1000 * sorted(round_seconds)[len(round_seconds) // 2] if round_seconds else None,
                round_ms=[round(1000 * value, 1) for value in round_seconds],
                draft_ms=[round(1000 * v, 1) for k, v in phase_log if k == 'draft'],
                verify_ms=[round(1000 * v, 1) for k, v in phase_log if k == 'verify'],
                refinement=hot.report())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--hot-fraction', type=float, default=0.2174)
    parser.add_argument('--budget-gib', type=int, default=22)
    parser.add_argument('--draft-bits', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--depths', default='0,1,2')
    parser.add_argument('--tokens', type=int, default=96)
    parser.add_argument('--prompt', default='Explain how a hash map works, in detail.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-rounds', action='store_true')
    args = parser.parse_args()
    global PROFILE_ROUNDS
    PROFILE_ROUNDS = args.profile_rounds
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(96 * 1024**3)
    mx.set_wired_limit(92 * 1024**3)
    torch.set_num_threads(4)
    guard = MemoryGuard(98, 0.8)
    guard.allowed_pressure = (1, 2)
    hot = NativeHotRefinements(args.model, args.hot_fraction, 128, args.budget_gib, 4, 32, 512)
    model, banks = ng.load_runtime(args.model, args.backbone, args.archive, [288] * 42, guard, hot, True, 56)
    results = []
    try:
        hot.warm(4, lambda: guard.check(force=True))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = (Path(ng.__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=True,
                                           return_dict=False, add_generation_prompt=True, chat_template=template)
        end = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
        eos = {end} if isinstance(end, int) else set(end)
        drafters = {}
        for depth in (int(value) for value in args.depths.split(',')):
            if depth == 0:
                out = ng.generate(model, tokenizer, ids, args.tokens, eos, ng.layer_major_prefill, 8, hot, guard)
                entry = dict(depth=0, decode_tps=out['decode_tps'], ttft_seconds=out['ttft_seconds'],
                             generated_ids=out['generated_ids'],
                             ready_call_fraction=out['refinement']['coverage']['call_fractions']['ready'])
            else:
                if depth not in drafters:
                    drafters.clear()
                    gc.collect()
                    drafters[depth] = load_drafter(args.model, depth, args.draft_bits)
                    guard.check(force=True)
                out = mtp_generate(model, drafters[depth], ids, args.tokens, eos, hot, guard, depth)
                entry = dict(depth=depth, decode_tps=out['decode_tps'], ttft_seconds=out['ttft_seconds'],
                             generated_ids=out['generated_ids'], mean_accepted=out['mean_accepted'],
                             hook_ms_per_round=out['hook_ms_per_round'], median_round_ms=out['median_round_ms'],
                             mean_unique_experts=out['mean_unique_experts'], mean_verify_rows=out['mean_verify_rows'],
                             round_ms=out['round_ms'], draft_ms=out['draft_ms'], verify_ms=out['verify_ms'],
                             rounds=len(out['accept_lengths']),
                             ready_call_fraction=out['refinement']['coverage']['call_fractions']['ready'])
            entry['peak_gib'] = ng.process_usage().lifetime_max_phys_footprint / 2**30
            entry['text'] = tokenizer.decode(entry['generated_ids'][:40])
            results.append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != 'generated_ids'}), flush=True)
    finally:
        hot.close()
        for bank in banks:
            bank.close()
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
