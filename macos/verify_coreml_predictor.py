"""Compare the Core ML jT predictor with the PyTorch one, offline (no GPU, no full model).

Replays predictor events (reset / preforward / commit) through a PyTorch CommittedPredictor and one
Core ML CommittedPredictor per compute-unit setting, in lockstep, and reports hot-set agreement,
block-probability error, and per-call latency with process CPU time.

Events come from NQ_PREDICTOR_RECORD (written by NativeHotRefinements during a real run) or are
synthesized. Example:
  python macos/verify_coreml_predictor.py --events /tmp/run.events.pkl --units ane cpu --output out.json
"""
import argparse
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import configure_slots, staged_predictor  # noqa: E402

HOME = Path.home()


def build(root, fraction, backend):
    os.environ.setdefault('NQ_FLASH_JT_PARAMS', json.dumps(dict(mix=0.75, hm=4.0, G=1, half_life=64.0)))
    module, staging = staged_predictor()
    predictor = module.CommittedPredictor(root, preset='spark_256K_U_2630_flat50', device='cpu',
                                          dtype=torch.float32, fixed_fraction=0.2)
    configure_slots(predictor, module, root, fraction)
    if backend.startswith('torch-noise'):
        # Sensitivity floor: PyTorch probabilities with multiplicative noise of about the Core ML fp16 error.
        scale = float(backend.partition(':')[2] or 0.003)
        rng = np.random.default_rng(7)
        predict = predictor.net.predict

        def noisy(*args):
            p = predict(*args)
            p = p * np.exp(scale * rng.standard_normal(p.shape)).astype(np.float32)
            return p / p.sum(-1, keepdims=True)

        predictor.net.predict = noisy
    elif backend != 'torch':
        from coreml_predictor import install
        install(predictor, root / 'serving/predictor/jt', compute_units=backend)
    predictor._staging = staging
    return predictor


def load_events(path):
    events = []
    with open(path, 'rb') as f:
        while True:
            try:
                events.append(pickle.load(f))
            except EOFError:
                return events


def synthetic_events(seed, prompts=2, prompt_rows=600, decode_rounds=150):
    """Prefill commits in 512-row chunks, then DFlash-like rounds: preforward one token, commit 1-8 rows.

    Routing follows a slowly drifting per-layer Zipf preference so hot sets are structured, not uniform.
    """
    rng = np.random.default_rng(seed)
    events = []
    for _ in range(prompts):
        events.append(('reset',))
        logits = rng.gumbel(size=(42, 288)) * 1.5
        position = 0
        vocab = rng.integers(0, 155008, 4000)

        def rows(n):
            nonlocal logits, position
            out = []
            for _ in range(n):
                logits = 0.98 * logits + 0.02 * rng.gumbel(size=(42, 288)) * 1.5
                p = np.exp(logits - logits.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                ids = np.stack([rng.choice(288, 8, replace=False, p=row) for row in p])
                w = rng.dirichlet(np.ones(8), 42).astype(np.float32)
                out.append((position, int(rng.choice(vocab)), ids.astype(np.int64), w,
                            np.ones(42, np.float32)))
                position += 1
            return out

        left = prompt_rows
        while left:
            take = min(left, 512)
            events.append(('commit', rows(take)))
            left -= take
        for _ in range(decode_rounds):
            accepted = rows(int(rng.integers(1, 9)))
            events.append(('preforward', accepted[0][1], accepted[0][0]))
            events.append(('commit', accepted))
    return events


class Agreement:
    def __init__(self, budgets):
        self.budgets = np.asarray(budgets)
        self.jaccard = []
        self.exact = []
        self.topb = []
        self.top8 = []
        self.abs_max = 0.0
        self.abs_sum = 0.0
        self.abs_n = 0
        self.sq_err = 0.0
        self.sq_ref = 0.0

    def wanted(self, ref, got):
        inter = (ref & got).sum(-1)
        union = np.maximum((ref | got).sum(-1), 1)
        self.jaccard.append(inter / union)
        self.exact.append((ref == got).all(-1))

    def block(self, ref, got):
        if ref is None:
            return
        diff = np.abs(ref - got)
        self.abs_max = max(self.abs_max, float(diff.max()))
        self.abs_sum += float(diff.sum())
        self.abs_n += diff.size
        self.sq_err += float((diff ** 2).sum())
        self.sq_ref += float((ref ** 2).sum())
        rank_ref = np.argsort(-ref, -1, kind='stable')
        rank_got = np.argsort(-got, -1, kind='stable')
        for layer, budget in enumerate(self.budgets):
            self.topb.append(len(np.intersect1d(rank_ref[layer, :budget], rank_got[layer, :budget])) / budget)
            self.top8.append(len(np.intersect1d(rank_ref[layer, :8], rank_got[layer, :8])) / 8)

    def report(self):
        jac = np.concatenate(self.jaccard) if self.jaccard else np.zeros(0)
        exact = np.concatenate(self.exact) if self.exact else np.zeros(0)
        return dict(
            wanted_layer_steps=int(jac.size), wanted_jaccard_mean=float(jac.mean()), wanted_jaccard_min=float(jac.min()),
            wanted_layer_steps_identical=float(exact.mean()),
            wanted_steps_all_layers_identical=float(np.mean([e.all() for e in self.exact])),
            block_topbudget_overlap_mean=float(np.mean(self.topb)), block_topbudget_overlap_min=float(np.min(self.topb)),
            block_top8_overlap_mean=float(np.mean(self.top8)), block_top8_overlap_min=float(np.min(self.top8)),
            block_abs_error_max=self.abs_max, block_abs_error_mean=self.abs_sum / max(self.abs_n, 1),
            block_relative_rms=(self.sq_err / max(self.sq_ref, 1e-30)) ** 0.5)


def replay(predictors, events):
    names = list(predictors)
    budgets = [predictors[names[0]].budgets[layer] for layer in range(3, 45)]
    agreement = {name: Agreement(budgets) for name in names[1:]}
    timing = {name: {'preforward': [], 'commit': []} for name in names}
    # Desired coverage: share of the routed (layer, expert) calls in each accepted decode round that the
    # round's wanted set (from preforward) contained, by count and by salience w^2 |x|^2.
    coverage = {name: [0, 0, 0.0, 0.0] for name in names}
    wanted = {}
    for event in events:
        kind = event[0]
        if kind == 'commit' and wanted:
            for name, mask in wanted.items():
                for _, _, ids, w, xn in event[1]:
                    hit = mask[np.arange(42)[:, None], ids]
                    salience = np.square(w.astype(np.float64)) * np.asarray(xn, np.float64)[:, None]
                    c = coverage[name]
                    c[0] += int(hit.sum()); c[1] += hit.size
                    c[2] += float(salience[hit].sum()); c[3] += float(salience.sum())
            wanted = {}
        results = {}
        for name, predictor in predictors.items():
            if kind == 'reset':
                predictor.reset()
                continue
            started, cpu = time.perf_counter(), time.process_time()
            if kind == 'preforward':
                mask = predictor.preforward(event[1], event[2])
                wanted[name] = np.asarray(mask, dtype=bool).copy()
            else:
                first = event[1][0][0]
                if first != predictor.t:
                    # native_prefill.BatchedPrefill._commit starts the clocks at the first committed row.
                    if predictor.ahead is not None or first < predictor.t:
                        raise ValueError('Recorded events skip committed rows')
                    predictor.t = first
                    predictor.prepared_t = -1
                    predictor.net.position = first
                mask = predictor.commit(event[1])
            timing[name][kind].append((time.perf_counter() - started, time.process_time() - cpu, len(event[1])
                                       if kind == 'commit' else 1))
            results[name] = (np.asarray(mask, dtype=bool), None if predictor.block is None else predictor.block.copy())
        if kind == 'reset':
            continue
        ref_mask, ref_block = results[names[0]]
        for name in names[1:]:
            agreement[name].wanted(ref_mask, results[name][0])
            agreement[name].block(ref_block, results[name][1])

    def summarize(samples, select=lambda n: True):
        chosen = [(w, c) for w, c, n in samples if select(n)]
        if not chosen:
            return None
        return dict(calls=len(chosen), wall_ms_median=1e3 * statistics.median(w for w, _ in chosen),
                    cpu_ms_median=1e3 * statistics.median(c for _, c in chosen),
                    wall_s_total=sum(w for w, _ in chosen), cpu_s_total=sum(c for _, c in chosen))

    timings = {name: dict(preforward=summarize(t['preforward']),
                          commit_decode=summarize(t['commit'], lambda n: n <= 8),
                          commit_prefill=summarize(t['commit'], lambda n: n > 8))
               for name, t in timing.items()}
    covered = {name: dict(calls=c[1], call_fraction=c[0] / max(c[1], 1), salience_fraction=c[2] / max(c[3], 1e-30))
               for name, c in coverage.items()}
    return {name: a.report() for name, a in agreement.items()}, timings, covered


def net_latency(net, repeats, warmup=3):
    """Per-call latency of the jT network alone: single row with a full window, 8-row and 7-row commits,
    and a 512-row prefill commit (predicted in 128-row chunks like CommittedPredictor.commit)."""
    rng = np.random.default_rng(1)

    def batch(n):
        return (rng.integers(0, 155008, n), rng.integers(0, 288, (n, 42, 8)),
                rng.dirichlet(np.ones(8), (n, 42)).astype(np.float32))

    def measure(n, reps, chunk=None, fresh=False):
        out = []
        for i in range(reps + warmup):
            if fresh:
                net.reset()
            else:
                net.reset()
                for _ in range(2):
                    net.predict(*batch(128))
            data = batch(n)
            started, cpu = time.perf_counter(), time.process_time()
            if chunk:
                for s in range(0, n, chunk):
                    net.predict(*(x[s:s + chunk] for x in data))
            else:
                net.predict(*data)
            if i >= warmup:
                out.append((time.perf_counter() - started, time.process_time() - cpu))
        return dict(calls=len(out), wall_ms_median=1e3 * statistics.median(w for w, _ in out),
                    cpu_ms_median=1e3 * statistics.median(c for _, c in out),
                    wall_ms_p90=1e3 * float(np.quantile([w for w, _ in out], 0.9)))

    # Steady state single steps: one window of history, then repeated one-row predictions.
    net.reset()
    for _ in range(2):
        net.predict(*batch(128))
    single = []
    for i in range(repeats * 5 + warmup):
        data = batch(1)
        started, cpu = time.perf_counter(), time.process_time()
        net.predict(*data)
        if i >= warmup:
            single.append((time.perf_counter() - started, time.process_time() - cpu))
    return dict(
        single_row=dict(calls=len(single), wall_ms_median=1e3 * statistics.median(w for w, _ in single),
                        cpu_ms_median=1e3 * statistics.median(c for _, c in single),
                        wall_ms_p90=1e3 * float(np.quantile([w for w, _ in single], 0.9))),
        commit_7_rows=measure(7, repeats), commit_8_rows=measure(8, repeats),
        prefill_512_rows=measure(512, max(repeats // 4, 3), chunk=128, fresh=True))


def compute_plan(package, units):
    import coremltools as ct
    from coreml_predictor import COMPUTE_UNITS, function_name
    counts = {}
    try:
        # Keep the MLModel alive: its compiled temporary directory is deleted with it.
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units]),
                                  function_name=function_name(1))
        compiled = model.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            compiled, compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units]))
        program = plan.model_structure.program
        for function in program.functions.values():
            for operation in function.block.operations:
                if operation.operator_name == 'const':
                    continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                device = 'unknown' if usage is None else type(usage.preferred_compute_device).__name__
                counts[device] = counts.get(device, 0) + 1
    except Exception as error:  # Placement is diagnostic only.
        counts['error'] = repr(error)
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=HOME / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit')
    p.add_argument('--events', type=Path, help='NQ_PREDICTOR_RECORD pickle; synthesized when omitted')
    p.add_argument('--max-events', type=int, default=0)
    p.add_argument('--fraction', type=float, default=0.15)
    p.add_argument('--units', nargs='+', default=['ane', 'all'],
                   help="Core ML compute units (ane, all), or torch-noise[:scale] for a perturbed-PyTorch floor")
    p.add_argument('--threads', type=int, default=4, help='PyTorch threads (the runtime uses 4)')
    p.add_argument('--repeats', type=int, default=40)
    p.add_argument('--skip-replay', action='store_true')
    p.add_argument('--skip-latency', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    predictors = {'torch': build(args.model, args.fraction, 'torch')}
    for units in args.units:
        predictors[units if units.startswith('torch') else f'coreml:{units}'] = build(args.model, args.fraction, units)
    report = dict(fraction=args.fraction, torch_threads=args.threads, events=str(args.events or 'synthetic'))
    from coreml_predictor import DEFAULT_PACKAGE
    report['compute_plan_rows1'] = {u: compute_plan(DEFAULT_PACKAGE, u) for u in args.units if not u.startswith('torch')}
    print(json.dumps(dict(compute_plan=report['compute_plan_rows1'])), flush=True)
    if not args.skip_replay:
        events = load_events(args.events) if args.events else synthetic_events(0)
        if args.max_events:
            events = events[:args.max_events]
        report['event_counts'] = {k: sum(e[0] == k for e in events) for k in ('reset', 'preforward', 'commit')}
        report['agreement_vs_torch'], report['replay_timing'], report['desired_coverage'] = replay(predictors, events)
        print(json.dumps(dict(agreement=report['agreement_vs_torch'], timing=report['replay_timing'],
                              desired_coverage=report['desired_coverage']), indent=1), flush=True)
    report['network_latency'] = {}
    for name, predictor in predictors.items():
        if name.startswith('torch-noise') or args.skip_latency:
            continue
        report['network_latency'][name] = net_latency(predictor.net, args.repeats)
        print(json.dumps({name: report['network_latency'][name]}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
