"""Run GLM Flash with bounded native MLX expert banks."""

import argparse
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.glm5_next.language import create_ssm_mask
from transformers import AutoTokenizer

from checkpoint import validate
from decoder import projection_matvec_banked
from metrics import MemoryGuard
from native_assets import REVISION, fingerprint, paths
from native_backbone import RoutedPlaceholder, load_prepared
from native_diagnostics import NativeDiagnostics
from packed_expert_bank import BaseReadStaging, PackedExpertBank
from resources import process_usage, system_pressure, vm_snapshot


class PagedExperts(nn.Module):
    def __init__(self, layer, bank, hot, cold_fastpath=False, diagnostics=None):
        super().__init__()
        self.gate = layer.gate
        self.shared_experts = layer.shared_experts
        self._bank = bank
        self._hot = hot
        self._forward_to_route_seconds = 0.0
        self._cold_fastpath = cold_fastpath
        self._cold_only_calls = 0
        self._cold_fastpath_calls = 0
        self._diagnostics = diagnostics

    def __call__(self, x):
        diagnostic = self._diagnostics
        measure = diagnostic is not None and diagnostic.synchronize_layers and diagnostic.phase == 'decode'
        if measure:
            if self._bank.fully_resident:
                raise ValueError('Synchronized diagnostics require paged expert banks')
            entered = time.perf_counter()
        ids, weights = self.gate(x)
        width = x.shape[-1]
        tokens = x.reshape(-1, width)
        started = time.perf_counter()
        if measure:
            gate_graph_seconds = started - entered
        if self._bank.fully_resident and self._hot is None:
            route = self._bank.select_device(ids.reshape(-1).astype(mx.uint32))
            self._forward_to_route_seconds += time.perf_counter() - started
        else:
            experts = tuple(ids.reshape(-1).tolist())
            self._forward_to_route_seconds += time.perf_counter() - started
            if measure:
                routed_at = time.perf_counter()
            route = mx.array(self._bank.select(experts), dtype=mx.uint32)
        if measure:
            selected_at = time.perf_counter()
        planes = self._bank.planes
        hot, hot_route = None, None
        if self._hot is not None:
            norms = mx.square(tokens.astype(mx.float32)).sum(axis=-1)
            hot, hot_route = self._hot.route(self._bank.layer, planes, experts, weights, norms)
        if measure:
            refined_at = time.perf_counter()
        use_refinements = hot is not None
        if hot is not None and self._hot.phase == 'decode' and not any(expert in hot.slots for expert in experts):
            self._cold_only_calls += 1
            if self._cold_fastpath:
                use_refinements = False
                self._cold_fastpath_calls += 1
        rounded = tokens.astype(mx.float16).astype(mx.float32)
        rows = mx.broadcast_to(rounded[:, None, :], (tokens.shape[0], ids.shape[-1], width)).reshape(route.size, width)

        def project(name, value):
            return projection_matvec_banked(*planes[name], value, route, 2,
                                            refinements=hot.planes[name] if use_refinements else None,
                                            refined_route=hot_route if use_refinements else None)

        gate = mx.minimum(project('gate', rows), 10.0)
        up = mx.clip(project('up', rows), -10.0, 10.0)
        routed = project('down', gate * mx.sigmoid(gate) * up).reshape(*ids.shape, width)
        routed = (routed * weights[..., None].astype(mx.float16).astype(mx.float32)).sum(axis=-2)
        routed = routed.astype(x.dtype)
        if measure:
            expert_graph_at = time.perf_counter()
        output = routed + self.shared_experts(x)
        if measure:
            shared_graph_at = time.perf_counter()
        self._bank.retain(output)
        if hot is not None:
            hot.retain(output)
            if self._hot.phase == 'prefill':
                mx.eval(output, norms, weights)
        if measure:
            evaluate_at = time.perf_counter()
            mx.eval(output, weights, norms if hot is not None else [])
            evaluated_at = time.perf_counter()
            diagnostic.record(
                'routed_layer', layer=self._bank.layer, expert_ids=list(experts),
                refined_slots=[hot.slots.get(expert, 0xffffffff) for expert in experts] if hot is not None else None,
                gate_graph_host_wall_seconds=gate_graph_seconds,
                route_extraction_wall_seconds=routed_at - started,
                base_selection_host_wall_seconds=selected_at - routed_at,
                refinement_route_host_wall_seconds=refined_at - selected_at,
                expert_graph_host_wall_seconds=expert_graph_at - refined_at,
                shared_graph_host_wall_seconds=shared_graph_at - expert_graph_at,
                retain_host_wall_seconds=evaluate_at - shared_graph_at,
                output_eval_wait_seconds=evaluated_at - evaluate_at,
                call_wall_seconds=evaluated_at - entered)
        return output


def load_runtime(root, backbone, archive, rows, guard, hot, full_base=False, base_budget_gib=32, preload_mib_s=0,
                 cold_fastpath=False, base_read_mode='scatter', diagnostics=None):
    if base_read_mode not in ('scatter', 'staged') or (full_base and base_read_mode != 'scatter'):
        raise ValueError('Staged required reads support paged expert-major execution only')
    if base_budget_gib not in ((56,) if full_base else (32, 40, 44)):
        raise ValueError('Use a 32, 40, or 44 GiB paged budget, or 56 GiB for the full base')
    if full_base and (len(rows) != 42 or any(row != 288 for row in rows)):
        raise ValueError('Full base residency requires all 288 experts in each routed layer')
    if full_base and hot is not None:
        raise ValueError('Full base residency currently supports base-only execution')
    if preload_mib_s not in (0, 256, 512, 1024) or (preload_mib_s and not full_base):
        raise ValueError('Preload pacing requires a supported full-base loading rate')
    def backbone_loaded(name, size):
        if diagnostics is not None:
            diagnostics.record('backbone_module_ready', module=name, bytes=size, memory=memory_sample())
        guard.check(force=True)

    model = load_prepared(root, backbone, backbone_loaded)
    banks = []
    staging = BaseReadStaging() if base_read_mode == 'staged' else None
    remaining = base_budget_gib * 1024**3
    try:
        for number, layer in enumerate(model.layers):
            if not isinstance(layer.mlp, RoutedPlaceholder):
                continue
            file, sidecar = paths(archive, number)
            evidence = json.loads(sidecar.read_text())
            if evidence['revision'] != REVISION or evidence['state'] != fingerprint(file):
                raise RuntimeError(f'Packed layer changed: {file.name}')
            bank = PackedExpertBank(file, number, rows[number - 3], REVISION, remaining, staging)
            banks.append(bank)
            remaining -= bank.bytes
            if full_base:
                try:
                    bank.preload(lambda: guard.check(force=True), preload_mib_s * 1024**2)
                except BaseException:
                    print(json.dumps(dict(event='base_layer_preload_failed', layer=number,
                                          monotonic_seconds=time.perf_counter(), stats=bank.stats,
                                          memory=memory_sample())), flush=True)
                    raise
                if evidence['state'] != fingerprint(file):
                    raise RuntimeError(f'Packed layer changed during preload: {file.name}')
                print(json.dumps(dict(event='base_layer_resident', layer=number,
                                      monotonic_seconds=time.perf_counter(), stats=bank.stats,
                                      base_bytes=sum(item.bytes for item in banks), memory=memory_sample())), flush=True)
            if hot is not None:
                hot.attach(number, bank.planes)
            layer.mlp = PagedExperts(layer.mlp, bank, hot, cold_fastpath, diagnostics).eval()
            if diagnostics is not None:
                diagnostics.record('base_bank_ready', layer=number, base_bytes=sum(item.bytes for item in banks),
                                   memory=memory_sample())
            guard.check(force=True)
        if len(banks) != 42:
            raise RuntimeError('Expected 42 packed routed layers')
        return model, banks
    except BaseException:
        for bank in banks:
            bank.close()
        raise


def layer_major_prefill(model, chunks, cache, guard, hot, diagnostics=None):
    hidden = [mx.repeat(model.model.embed_tokens(mx.array([chunk]))[:, :, None],
                        model.config.hc_mult, axis=2) for chunk in chunks]
    indices = [None] * len(chunks)
    for number, (layer, state) in enumerate(zip(model.layers, cache, strict=True)):
        if diagnostics is not None:
            diagnostics.record('prefill_layer_started', layer=number, memory=memory_sample())
        for i, value in enumerate(hidden):
            if hot is not None:
                hot.chunk_index = i
            mask = create_ssm_mask(value[:, :, 0], state) if layer.block_type == 'linear_attention' else None
            hidden[i], indices[i] = layer(value, mask, state, indices[i])
            mx.eval(hidden[i], indices[i], state.state)
            guard.check()
        if diagnostics is not None:
            diagnostics.record('prefill_layer_complete', layer=number, chunks=len(chunks), memory=memory_sample())
    value = model.model.norm(hidden[-1][:, -1:].mean(axis=2))
    logits = model.model.embed_tokens.as_linear(value) if model.config.tie_word_embeddings else model.lm_head(value)
    mx.eval(logits)
    if hot is not None:
        for i, chunk in enumerate(chunks):
            hot.commit(chunk, i)
    return logits


def token_major_prefill(model, chunks, cache, guard, hot):
    for chunk in chunks:
        logits = model(mx.array([chunk]), cache=cache, num_logits_to_keep=1).logits
        mx.eval(logits, [entry.state for entry in cache])
        if hot is not None:
            hot.commit(chunk)
        guard.check()
    return logits


def stats(model):
    result = dict(forward_to_route_seconds=0.0, cold_only_calls=0, cold_fastpath_calls=0)
    for layer in model.layers:
        if isinstance(layer.mlp, PagedExperts):
            result['forward_to_route_seconds'] += layer.mlp._forward_to_route_seconds
            result['cold_only_calls'] += layer.mlp._cold_only_calls
            result['cold_fastpath_calls'] += layer.mlp._cold_fastpath_calls
            for name, value in layer.mlp._bank.stats.items():
                result[name] = result.get(name, 0) + value
    result['process_disk_read_bytes'] = process_usage().diskio_bytesread
    result['sync_reads'] = result['misses'] - result['prefetch_hits']
    result['total_base_read_bytes'] = result['read_bytes'] + result['prefetch_read_bytes']
    return result


def memory_sample():
    usage = process_usage()
    return dict(physical_bytes=usage.phys_footprint, lifetime_physical_peak_bytes=usage.lifetime_max_phys_footprint,
                resident_bytes=usage.resident_size, wired_bytes=usage.wired_size,
                mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory(),
                mlx_cache_bytes=mx.get_cache_memory(), system_pressure=system_pressure(), system_vm=vm_snapshot())


def diff(after, before):
    return {name: value - before[name] for name, value in after.items()}


def greedy(logits):
    if not mx.isfinite(logits).all().item():
        raise RuntimeError('Nonfinite native logits')
    return int(logits.argmax(axis=-1).item())


def top_margin(logits):
    pair = mx.topk(logits.astype(mx.float32), 2, axis=-1)
    return float((pair.max() - pair.min()).item())


def generate(model, tokenizer, ids, max_tokens, eos, prefill, chunk, hot, guard, forced=None, base_prefetch=None,
             diagnostics=None):
    if base_prefetch is not None and hot is None:
        raise ValueError('Base read-ahead requires a committed predictor')
    if diagnostics is not None and (prefill is not layer_major_prefill
                                    or (diagnostics.synchronize_layers and max_tokens > 128)):
        raise ValueError('Native diagnostics require layer-major prefill and at most 128 synchronized decode steps')
    cache = model.make_cache()
    if hot is not None:
        hot.reset()
    before = stats(model)
    memory = {'loaded': memory_sample()}
    if diagnostics is not None:
        diagnostics.set_phase('prefill')
        diagnostics.record('prefill_started', prompt_tokens=len(ids), memory=memory['loaded'])
    start = time.perf_counter()
    chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    if diagnostics is not None:
        logits = prefill(model, chunks, cache, guard, hot, diagnostics)
    else:
        logits = prefill(model, chunks, cache, guard, hot)
    token = greedy(logits)
    ttft = time.perf_counter() - start
    after_prefill = stats(model)
    memory['after_prefill'] = memory_sample()
    if diagnostics is not None:
        diagnostics.record('prefill_complete', memory=memory['after_prefill'])
        diagnostics.set_phase('decode')
    output = []
    times = []
    greedy_ids = [token]
    replay_steps = []
    for index in range(max_tokens):
        token = greedy_ids[-1] if forced is None else forced[index]
        if forced is None and token in eos:
            break
        output.append(token)
        if forced is None and index + 1 == max_tokens:
            break
        if diagnostics is not None:
            diagnostics.begin_step(index)
        start = time.perf_counter()
        if hot is not None:
            hot.begin_decode(token)
        if base_prefetch is not None:
            base_prefetch.begin(hot.predictor.loading_priority())
        if diagnostics is not None:
            policy_at = time.perf_counter()
        logits = model(mx.array([[token]]), cache=cache).logits
        mx.eval(logits, [entry.state for entry in cache])
        if diagnostics is not None:
            forward_at = time.perf_counter()
        if hot is not None:
            hot.commit([token])
        if base_prefetch is not None:
            base_prefetch.drain()
        if diagnostics is not None:
            committed_at = time.perf_counter()
        greedy_ids.append(greedy(logits))
        times.append(time.perf_counter() - start)
        if forced is not None:
            replay_steps.append(dict(ready_mask=hot.readiness_digest() if hot is not None else None,
                                     logit_margin=top_margin(logits)))
        guard.check()
        if diagnostics is not None:
            diagnostics.record('decode_step_complete', memory=memory_sample(),
                               begin_policy_host_wall_seconds=policy_at - start,
                               forward_and_eval_wall_seconds=forward_at - policy_at,
                               commit_host_wall_seconds=committed_at - forward_at,
                               sampling_wall_seconds=start + times[-1] - committed_at,
                               timed_decode_wall_seconds=times[-1])
    cache_bytes = {}
    for entry in cache:
        name = type(entry).__name__
        cache_bytes[name] = cache_bytes.get(name, 0) + sum(
            array.nbytes for _, array in tree_flatten(entry.state) if isinstance(array, mx.array))
    memory['after_decode'] = memory_sample()
    if diagnostics is not None:
        diagnostics.set_phase('complete')
        diagnostics.record('decode_complete', memory=memory['after_decode'])
    return dict(text=tokenizer.decode(output, skip_special_tokens=True), generated_ids=output,
                greedy_next_ids=greedy_ids, teacher_forced=forced is not None, prompt_tokens=len(ids),
                generated_tokens=len(output), ttft_seconds=ttft, decode_tps=len(times) / sum(times) if times else None,
                decode_step_seconds=times, replay_steps=replay_steps,
                prefill=diff(after_prefill, before), decode=diff(stats(model), after_prefill),
                cache_bytes_by_type=cache_bytes, memory=memory,
                refinement=hot.report() if hot is not None else None)


def verify_prefill(model, ids, chunk, guard, hot):
    chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    standard_cache, layered_cache = model.make_cache(), model.make_cache()
    standard_start = time.perf_counter()
    standard = token_major_prefill(model, chunks, standard_cache, guard, hot)
    standard_seconds = time.perf_counter() - standard_start
    if hot is not None:
        standard_predictor = hot.prefill_state_digest()
        hot.reset()
    layered_start = time.perf_counter()
    layered = layer_major_prefill(model, chunks, layered_cache, guard, hot)
    layered_seconds = time.perf_counter() - layered_start
    if hot is not None and standard_predictor != hot.prefill_state_digest():
        raise RuntimeError('Layer-major prefill changed the committed predictor state')
    exact = mx.array_equal(standard, layered).item()
    state_arrays = 0
    for a, b in zip(standard_cache, layered_cache, strict=True):
        for (name, x), (other, y) in zip(tree_flatten(a.state), tree_flatten(b.state), strict=True):
            if name != other or (isinstance(x, mx.array) != isinstance(y, mx.array)):
                raise RuntimeError('Prefill cache layout differs')
            if isinstance(x, mx.array):
                state_arrays += 1
                if not mx.array_equal(x, y).item():
                    raise RuntimeError(f'Prefill cache values differ at {name}')
            elif x != y:
                raise RuntimeError(f'Prefill cache metadata differs at {name}')
    if not exact:
        raise RuntimeError('Layer-major prefill changed logits')
    return dict(status='passed', logits_identical=exact, cache_arrays_identical=state_arrays,
                predictor_state_identical=True if hot is not None else None,
                prompt_tokens=len(ids), chunk_size=chunk, standard_seconds=standard_seconds,
                layered_seconds=layered_seconds)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='Reply with exactly GLM_RESIDENT_OK and no other text.')
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, choices=(4, 8, 16), default=4)
    parser.add_argument('--rows', type=int, choices=(64, 96, 128, 160, 192, 224, 288), default=96)
    parser.add_argument('--full-base', action='store_true')
    parser.add_argument('--full-base-wired-gib', type=int, choices=(0, 32, 56, 68))
    parser.add_argument('--preload-mib-s', type=int, choices=(0, 256, 512, 1024), default=0)
    parser.add_argument('--base-budget-gib', type=int, choices=(32, 40, 44, 56), default=32)
    parser.add_argument('--row-plan', type=Path)
    parser.add_argument('--refinements', action='store_true')
    parser.add_argument('--refinement-budget-gib', type=int, choices=(4, 6), default=4)
    parser.add_argument('--hot-fraction', type=float, choices=(0.01, 0.02, 0.03, 0.05), default=0.02)
    parser.add_argument('--prefill-rows', type=int, choices=(64, 128, 160, 192, 240), default=64)
    parser.add_argument('--token-major', action='store_true')
    parser.add_argument('--verify-prefill', action='store_true')
    parser.add_argument('--fixed-json', type=Path)
    parser.add_argument('--base-prefetch', type=int, choices=(1, 2, 4))
    parser.add_argument('--base-read-mode', choices=('scatter', 'staged'), default='scatter')
    parser.add_argument('--cold-fastpath', action='store_true')
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--profile-sync', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.max_tokens <= 4096:
        parser.error('Use a fresh output and a positive token budget of at most 4096')
    if args.diagnostics is not None and (args.diagnostics.exists() or args.diagnostics.resolve() == args.output.resolve()):
        parser.error('Use a fresh diagnostic path separate from the result')
    if args.diagnostics is not None and (args.token_major or args.verify_prefill):
        parser.error('Native diagnostics require ordinary layer-major generation')
    if args.profile_sync and (args.diagnostics is None or args.full_base or args.max_tokens > 128):
        parser.error('Synchronized diagnostics require a diagnostic file, paged banks, and at most 128 output tokens')
    if args.verify_prefill and args.fixed_json:
        parser.error('The prefill equivalence check does not generate tokens')
    if args.full_base != (args.rows == 288) or (args.full_base and args.row_plan is not None):
        parser.error('Full residency requires --full-base --rows 288 and excludes a row plan')
    if args.full_base != (args.base_budget_gib == 56):
        parser.error('The full base requires a 56 GiB base budget; paged banks allow at most 44 GiB')
    if args.full_base and args.refinements:
        parser.error('Full base residency currently supports base-only execution')
    if args.full_base and args.base_read_mode != 'scatter':
        parser.error('Staged required reads support paged expert-major execution only')
    if not args.full_base and (args.full_base_wired_gib is not None or args.preload_mib_s):
        parser.error('Residency and preload pacing controls require --full-base')
    if args.refinement_budget_gib != 4 and not args.refinements:
        parser.error('A refinement budget requires --refinements')
    if args.hot_fraction == 0.05 and (not args.refinements or args.refinement_budget_gib != 6):
        parser.error('Five-percent hot allocation requires --refinements --refinement-budget-gib 6')
    if args.base_prefetch and (not args.refinements or args.verify_prefill or args.full_base):
        parser.error('Base prediction requires the committed refinement predictor during generation')
    if args.cold_fastpath and not args.refinements:
        parser.error('The cold fast path requires a mixed-precision runtime')
    if args.chunk_size * 8 > args.rows or (args.refinements and args.chunk_size * 8 > args.prefill_rows):
        parser.error('The packed row budget must cover the full prefill chunk')
    rows = [args.rows] * 42
    if args.row_plan is not None:
        rows = json.loads(args.row_plan.read_text())
        if (not isinstance(rows, list) or len(rows) != 42
                or any(type(row) is not int or not args.chunk_size * 8 <= row <= 224 for row in rows)
                or sum(rows) != args.rows * 42):
            parser.error('The row plan must contain 42 bounded counts at the selected total budget')
    validate(args.model)
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(76 * 1024**3)
    wired_gib = (68 if args.full_base_wired_gib is None else args.full_base_wired_gib) if args.full_base else 0
    if args.full_base:
        mx.set_wired_limit(wired_gib * 1024**3)
    guard = MemoryGuard(82)
    config = json.loads((args.model / 'config.json').read_text())['text_config']
    if config['swiglu_limit'] != 10:
        raise RuntimeError('Unexpected activation clamp in model configuration')
    eos = json.loads((args.model / 'generation_config.json').read_text())['eos_token_id']
    eos = {eos} if isinstance(eos, int) else set(eos)
    hot = None
    diagnostics = None
    banks = []
    base_prefetch = None
    try:
        if args.diagnostics is not None:
            diagnostics = NativeDiagnostics(args.diagnostics, args.profile_sync)
        if args.refinements:
            from native_refinements import NativeHotRefinements

            hot = NativeHotRefinements(args.model, args.hot_fraction, args.prefill_rows, args.refinement_budget_gib)
            if diagnostics is not None:
                diagnostics.record('predictor_ready', memory=memory_sample())
        start = time.perf_counter()
        model, banks = load_runtime(args.model, args.backbone, args.archive, rows, guard, hot,
                                    args.full_base, args.base_budget_gib, args.preload_mib_s, args.cold_fastpath,
                                    args.base_read_mode, diagnostics)
        if args.base_prefetch:
            from native_base_prefetch import NativeBasePrefetch

            base_prefetch = NativeBasePrefetch(banks, args.base_prefetch)
        startup_stats = stats(model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        template = (Path(__file__).resolve().parents[1] / 'spark/flash/thinking_off.jinja').read_text()
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}], tokenize=True,
                                           return_dict=False, add_generation_prompt=True, chat_template=template)
        if not ids or len(ids) + args.max_tokens > 4096:
            parser.error('Prompt and output must fit the 4096-token validation bound')
        if args.verify_prefill:
            result = verify_prefill(model, ids, args.chunk_size, guard, hot)
        else:
            forced = None
            if args.fixed_json:
                forced = json.loads(args.fixed_json.read_text())['generated_ids'][:args.max_tokens]
                if len(forced) != args.max_tokens or any(type(i) is not int or not 0 <= i < config['vocab_size'] for i in forced):
                    parser.error('The fixed token trace is invalid or too short')
            prefill = token_major_prefill if args.token_major else layer_major_prefill
            startup = time.perf_counter() - start
            result = generate(model, tokenizer, ids, args.max_tokens, eos, prefill, args.chunk_size, hot, guard, forced,
                              base_prefetch, diagnostics)
            result.update(startup_seconds=startup, rows=args.rows, chunk_size=args.chunk_size,
                          rows_by_layer=rows, prefill_order=prefill.__name__, checkpoint_revision=REVISION,
                          backbone_bytes=sum(a.nbytes for _, a in tree_flatten(model.parameters())),
                          base_bytes=sum(bank.bytes for bank in banks),
                          base_limit_bytes=args.base_budget_gib * 1024**3,
                          base_read_mode=args.base_read_mode,
                          base_staging_bytes=len(banks[0].staging.storage) if banks[0].staging is not None else 0,
                          full_base=args.full_base, startup_stats=startup_stats,
                          requested_wired_bytes=wired_gib * 1024**3, preload_mib_s=args.preload_mib_s,
                          residency_refresh_interval_ms=os.environ.get('MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS'),
                          route_timing_includes_device_sync=not args.full_base,
                          route_timing_includes_base_load=False,
                          base_prefetch=base_prefetch.report() if base_prefetch is not None else None,
                          cold_fastpath=args.cold_fastpath,
                          hot_fraction=args.hot_fraction if hot is not None else None)
            if diagnostics is not None:
                result['diagnostics'] = dict(file=str(args.diagnostics), synchronize_layers=args.profile_sync,
                                             timings_are_diagnostic=True,
                                             limitation='Instrumentation changes host timing and can change refinement readiness.')
        with args.output.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
        print(json.dumps({key: result[key] for key in ('status', 'text', 'decode_tps', 'ttft_seconds') if key in result}), flush=True)
    finally:
        try:
            try:
                if base_prefetch is not None:
                    base_prefetch.close()
            finally:
                if hot is not None:
                    hot.close()
        finally:
            try:
                for bank in banks:
                    bank.close()
            finally:
                if diagnostics is not None:
                    diagnostics.close()


if __name__ == '__main__':
    main()
