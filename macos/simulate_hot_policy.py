"""Offline (CPU-only) replay of recorded decode traces through the jT predictor, the hot-set policy and a
delivery model, to score hot-set policies by the share of routed expert calls that run at 4 bits.

Traces come from NQ_PREDICTOR_RECORD=<file> runs of native_dflash.py: predictor events (reset / preforward /
commit), the wanted sets, prompt salience, read completions, and the slot tables of every publish and
verify step with timestamps. The jT network is deterministic in its inputs, which do not depend on the policy,
so its outputs are computed once per trace (a "tape") and replayed for every policy.

The delivery model follows NativeHotRefinements with background installs:
  plan (predictor hook)  : order missing wanted experts by loading priority and submit reads, at most
                           `pending` in flight (a read holds its pending entry until the next publish);
  reader                 : `readers` threads, FIFO, each read takes `read_ms`;
  publish (step boundary): activate finished copies, then stage completed reads: reserve a slot (free, else
                           evict a resident expert outside the wanted set) and copy (`copy_ms`, 2 threads);
  verify                 : activate finished copies; the routed calls of the round run at 4 bits iff resident.
Direct mode (`direct=True`) models reads that land straight in free slots: at publish, victims are evicted
for missing wanted experts plus `spare` free slots per layer, and reads into free slots start at publish and
at plan; a read becomes resident at the first publish/verify after it completes.

Usage:
  python macos/simulate_hot_policy.py --events run.events.pkl [--events ...] [--config '{"hm": 2}'] ...
"""
import argparse
import itertools
import json
import os
import pickle
import sys
from collections import OrderedDict, deque
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hot_policy  # noqa: E402
from hot_policy import choose_victim, seed_from_prompt  # noqa: E402

ROOT = Path.home() / 'Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit'
COLD = 0xffffffff


def load_events(path):
    events = []
    with open(path, 'rb') as f:
        while True:
            try:
                events.append(pickle.load(f))
            except EOFError:
                return events


def requests(events):
    """Split a run into requests (reset to reset)."""
    out, current = [], None
    for event in events:
        if event[0] == 'reset':
            if current:
                out.append(current)
            current = []
        elif current is not None:
            current.append(event)
    if current:
        out.append(current)
    return out


# ---------------------------------------------------------------------------------------------- predictor

class TapeNet:
    """Stands in for IncrementalJT: returns recorded outputs in call order."""

    def __init__(self, tape):
        self.tape = tape
        self.index = 0
        self.position = 0
        self.cache = []
        self.model = None

    def reset(self):
        self.position = 0

    def predict(self, tokens, prev_ids, prev_q):
        out = self.tape[self.index]
        self.index += 1
        if len(out) != len(tokens):
            raise RuntimeError('Tape is out of step with the recorded events')
        self.position += len(tokens)
        return out


class Predictor:
    """One CommittedPredictor (from the checkpoint's own code) reused across configurations."""

    def __init__(self, root=ROOT, fraction=0.2):
        import torch
        macos = Path(__file__).resolve().parent
        sys.path.insert(0, str(macos))
        from predictor import configure_slots, staged_predictor
        os.environ.setdefault('NQ_FLASH_JT_PARAMS', json.dumps(BASE['jt']))
        self.module, self.staging = staged_predictor()
        self.root, self.fraction = Path(root), fraction
        self.p = self.module.CommittedPredictor(root, preset='spark_256K_U_2630_flat50', device='cpu',
                                                dtype=torch.float32, fixed_fraction=0.2)
        configure_slots(self.p, self.module, self.root, fraction)
        self.real_net = self.p.net
        self.policy_module = self.module.module(self.root / 'serving/predictor/jt/policy.py', 'nq_sim_policy')

    def configure(self, jt):
        params = self.module.policy_parameters(json.dumps(jt))
        self.p.parameters = params
        self.p.policy_cls = partial(self.policy_module.FloatingSet, **params)

    def build_tape(self, request):
        """Run the real jT network over one request's events and keep its outputs."""
        self.p.net = self.real_net
        tape = []
        predict = self.real_net.predict

        def recording(*args):
            out = predict(*args)
            tape.append(out.copy())
            return out

        self.real_net.predict = recording
        try:
            self.p.reset()
            for event in request:
                self.apply(event)
        finally:
            self.real_net.predict = predict
        return tape

    def apply(self, event):
        p = self.p
        kind = event[0]
        if kind == 'preforward':
            return p.preforward(event[1], event[2])
        if kind == 'commit':
            first = event[1][0][0]
            if first != p.t:
                # native_prefill.BatchedPrefill._commit starts the clocks at the first committed row.
                if p.ahead is not None or first < p.t:
                    raise ValueError('Recorded events skip committed rows')
                p.t = first
                p.prepared_t = -1
                p.net.position = first
            return p.commit(event[1])
        return None


def tapes_for(path, predictor, cache_dir=Path('/tmp/hotpolicy/tapes')):
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = Path(path).stat()
    target = cache_dir / f'{Path(path).name}.{stat.st_size}.{int(stat.st_mtime)}.tape.pkl'
    if target.exists():
        return pickle.loads(target.read_bytes())
    reqs = requests(load_events(path))
    tapes = [predictor.build_tape([e for e in r if e[0] in ('preforward', 'commit')]) for r in reqs]
    target.write_bytes(pickle.dumps(tapes, protocol=5))
    return tapes


# ---------------------------------------------------------------------------------------------- delivery

class Bank:
    __slots__ = ('slots', 'free', 'staging', 'capacity')

    def __init__(self, capacity, initial):
        self.capacity = capacity
        self.slots = OrderedDict((int(e), True) for e in initial)
        self.free = capacity - len(self.slots)
        self.staging = {}   # expert -> time its data is in the slot (inf while the read is in flight)


class Delivery:
    def __init__(self, budgets, defaults, cfg):
        self.cfg = cfg
        self.banks = [Bank(budgets[i], defaults[i]) for i in range(42)]
        self.pending = OrderedDict()   # (layer index, expert) -> [submit, done or None]
        self.queue = deque()           # not yet started reads (keys)
        self.threads = [0.0] * cfg['readers']
        self.copy_threads = [0.0] * cfg['copy_threads']
        self.reads = 0
        self.discarded = 0
        self.wait_seconds = 0.0
        self.last_used = np.zeros((42, 288))
        self.now = 0.0

    def resident(self):
        mask = np.zeros((42, 288), dtype=bool)
        for i, bank in enumerate(self.banks):
            if bank.slots:
                mask[i, list(bank.slots)] = True
        return mask

    def advance(self, t):
        """Start queued reads on free reader threads up to time t."""
        read = self.cfg['read_ms'] / 1000
        while self.queue:
            free = min(range(len(self.threads)), key=self.threads.__getitem__)
            start = max(self.threads[free], self.pending[self.queue[0]][0])
            if start > t:
                break
            key = self.queue.popleft()
            done = start + read
            self.threads[free] = done
            self.pending[key][1] = done
            if self.cfg['direct']:
                self.banks[key[0]].staging[key[1]] = done

    def flip(self, t):
        for bank in self.banks:
            if bank.staging:
                for expert, ready in list(bank.staging.items()):
                    if ready <= t:
                        del bank.staging[expert]
                        bank.slots[expert] = True

    def victim(self, i, wanted, priority):
        rule = self.cfg['evict']
        scores = (None if rule == 'fifo' else priority[i] if rule == 'score' else self.last_used[i])
        return choose_victim(self.banks[i].slots, set(np.flatnonzero(wanted[i]).tolist()), scores)

    def submit_missing(self, t, wanted, priority):
        """Queue reads for missing wanted experts, highest loading priority first (NativeHotRefinements._queue)."""
        cfg = self.cfg
        if cfg['direct']:
            for key in [k for k, (s, d) in self.pending.items() if d is not None and d <= t]:
                del self.pending[key]
        if cfg['cancel']:
            for key in list(self.queue):
                if not wanted[key]:
                    self.queue.remove(key)
                    del self.pending[key]
                    if cfg['direct']:
                        bank = self.banks[key[0]]
                        del bank.staging[key[1]]
                        bank.free += 1
        missing = wanted & ~self.resident()
        layers, experts = np.nonzero(missing)
        order = np.lexsort((experts, layers, -priority[layers, experts]))
        for i, e in zip(layers[order].tolist(), experts[order].tolist()):
            bank = self.banks[i]
            if e in bank.staging or (i, e) in self.pending:
                continue
            if len(self.pending) >= cfg['pending']:
                break
            if cfg['direct']:
                if bank.free <= 0:
                    continue
                bank.free -= 1
                bank.staging[e] = float('inf')
            self.pending[(i, e)] = [t, None]
            self.queue.append((i, e))
            self.reads += 1
        self.advance(t)

    def publish(self, t, wanted, priority):
        cfg = self.cfg
        self.advance(t)
        self.flip(t)
        if cfg['direct']:
            for key in [k for k, (s, d) in self.pending.items() if d is not None and d <= t]:
                del self.pending[key]
            # Free slots for missing wanted experts (plus spares), evicting outside the wanted set.
            for i, bank in enumerate(self.banks):
                need = int((wanted[i] & ~self.resident_row(i)).sum()) - len(bank.staging) + cfg['spare']
                while bank.free < need:
                    victim = self.victim(i, wanted, priority)
                    if victim is None:
                        break
                    del bank.slots[victim]
                    bank.free += 1
            self.submit_missing(t, wanted, priority)
            return
        copy = cfg['copy_ms'] / 1000
        for key in [k for k, (s, d) in self.pending.items() if d is not None and d <= t]:
            del self.pending[key]
            i, e = key
            bank = self.banks[i]
            if e in bank.slots or e in bank.staging:
                continue
            if not wanted[i, e]:
                self.discarded += 1
                continue
            if bank.free > 0:
                bank.free -= 1
            else:
                victim = self.victim(i, wanted, priority)
                if victim is None:
                    self.discarded += 1
                    continue
                del bank.slots[victim]
            thread = min(range(len(self.copy_threads)), key=self.copy_threads.__getitem__)
            done = max(t, self.copy_threads[thread]) + copy
            self.copy_threads[thread] = done
            bank.staging[e] = done

    def resident_row(self, i):
        row = np.zeros(288, dtype=bool)
        if self.banks[i].slots:
            row[list(self.banks[i].slots)] = True
        return row

    def reset(self):
        # NativeHotRefinements.reset: drain copies (activate), discard pending reads.
        for bank in self.banks:
            for expert, ready in list(bank.staging.items()):
                del bank.staging[expert]
                if ready != float('inf') and not self.cfg['direct']:
                    bank.slots[expert] = True
                else:
                    bank.free += 1
        self.pending.clear()
        self.queue.clear()


# ---------------------------------------------------------------------------------------------- simulation

# The settings of the recorded runs (commit 46e443d). Configurations are overrides of these; NEW is the
# hot_policy.DEFAULTS equivalent. Measured in-model: ~1.07 ms per install copy per thread, 4 install threads.
BASE = dict(jt=dict(mix=0.75, hm=4.0, G=1, half_life=64.0), evict='fifo', seed=0.0,
            readers=8, pending=48, read_ms=5.1, copy_ms=2.0, direct=False, spare=0, cancel=False,
            verify_wait=False, verify_wait_ms=None, copy_threads=2, replay_wanted=False, replay_resident=False)


NEW = dict(jt=dict(hot_policy.DEFAULTS['jt']), evict=hot_policy.DEFAULTS['evict'],
           verify_wait=hot_policy.DEFAULTS['verify_wait'], pending=hot_policy.DEFAULTS['pending'], copy_ms=1.07,
           copy_threads=4)


class Score:
    names = ('calls', 'ready', 'wanted', 'wanted_ready', 'late', 'outside', 'all_calls', 'all_ready',
             'salience', 'salience_ready')

    def __init__(self):
        self.v = dict.fromkeys(self.names, 0.0)

    def add(self, rows, resident, wanted):
        layer = np.arange(42)[:, None]
        for n, (_, _, ids, w, xn) in enumerate(rows):
            hit = resident[layer, ids]
            if n == 0:
                want = wanted[layer, ids]
                self.v['calls'] += hit.size
                self.v['ready'] += hit.sum()
                self.v['wanted'] += want.sum()
                self.v['wanted_ready'] += (want & hit).sum()
                self.v['late'] += (want & ~hit).sum()
                self.v['outside'] += (hit & ~want).sum()
            sal = np.square(w.astype(np.float64)) * np.asarray(xn, np.float64)[:, None]
            self.v['all_calls'] += hit.size
            self.v['all_ready'] += hit.sum()
            self.v['salience'] += sal.sum()
            self.v['salience_ready'] += sal[hit].sum()

    def merge(self, other):
        for k in self.names:
            self.v[k] += other.v[k]

    def report(self):
        v = self.v
        c = max(v['calls'], 1)
        return dict(ready=v['ready'] / c, wanted=v['wanted'] / c, wanted_ready=v['wanted_ready'] / c,
                    late=v['late'] / c, outside=v['outside'] / c, all_rows_ready=v['all_ready'] / max(v['all_calls'], 1),
                    salience_ready=v['salience_ready'] / max(v['salience'], 1e-30), rounds_calls=int(v['calls']))


def unpack(bits):
    return np.unpackbits(bits, axis=-1)[:, :288].astype(bool)


def simulate(paths_tapes, predictor, cfg, per_request=False):
    """cfg: overrides of BASE. Returns total and per-request scores."""
    cfg = {**BASE, **cfg, 'jt': {**BASE['jt'], **cfg.get('jt', {})}}
    predictor.configure(cfg['jt'])
    p = predictor.p
    total, rows = Score(), []
    reads = discarded = 0
    wait = 0.0
    for path, events, tapes in paths_tapes:
        reqs = requests(events)
        delivery = Delivery([p.budgets[L] for L in p.layers], p.defaults, cfg)
        for request, tape in zip(reqs, tapes):
            score = Score()
            p.net = TapeNet(tape)
            p.reset()
            if cfg['spare']:
                # Direct reads need free slots: the policy's pool is `spare` smaller than the bank.
                for policy in p.policies:
                    policy.inner.nf = max(int(policy.fixed.sum()) + 1, policy.inner.nf - cfg['spare'])
            delivery.reset()
            wanted = np.zeros((42, 288), dtype=bool)
            priority = np.zeros((42, 288))
            decode = False
            verifies = []          # simulated residency at each verify of the decode
            commits = 0
            stream = deque(request)
            # Timed events are handled in recorded time order; predictor events in stream order, lazily, at
            # the time of the wanted event that follows each preforward.
            while stream:
                event = stream.popleft()
                kind = event[0]
                if kind == 'decode_start':
                    decode = True
                elif kind == 'prompt_salience':
                    if cfg['seed']:
                        seed_from_prompt(p, event[1], cfg['seed'])
                elif kind == 'commit':
                    if decode:
                        score.add(event[1], verifies[commits], wanted)
                        commits += 1
                        for _, _, ids, _, _ in event[1]:
                            delivery.last_used[np.arange(42)[:, None], ids] = delivery.now
                    predictor.apply(event)
                elif kind == 'preforward':
                    # Its wanted event is the next 'wanted' in the stream (same thread).
                    t = next(e[1] for e in stream if e[0] == 'wanted')
                    delivery.now = t
                    delivery.advance(t)
                    wanted = np.asarray(predictor.apply(event), dtype=bool)
                    priority = p.loading_priority()
                elif kind == 'wanted':
                    if cfg['replay_wanted']:
                        wanted = unpack(event[2])
                    delivery.submit_missing(event[1], wanted, priority)
                elif kind == 'resident':
                    tag, t = event[1], event[2]
                    delivery.now = t
                    if tag == 'publish':
                        delivery.publish(t, wanted, priority)
                    else:
                        delivery.advance(t)
                        if cfg['verify_wait']:
                            # The verify waits for the copies started at the last publish (flip(wait=True)).
                            ready = [r for b in delivery.banks for r in b.staging.values() if r != float('inf')]
                            last = max(ready, default=t)
                            if cfg['verify_wait_ms'] is not None:
                                last = min(last, t + cfg['verify_wait_ms'] / 1000)
                            delivery.wait_seconds += max(0.0, last - t)
                            delivery.flip(max(t, last))
                        else:
                            delivery.flip(t)
                        verifies.append(unpack(event[3]) if cfg['replay_resident'] else delivery.resident())
            total.merge(score)
            rows.append(dict(path=Path(path).name, **score.report()))
        reads += delivery.reads
        discarded += delivery.discarded
        wait += delivery.wait_seconds
    out = dict(total=total.report(), reads=reads, discarded=discarded, verify_wait_seconds=wait)
    if per_request:
        out['requests'] = rows
    return out


def recorded(events_paths):
    """In-model numbers from the recorded residency snapshots (sanity check of the scoring)."""
    out = []
    for path in events_paths:
        for request in requests(load_events(path)):
            score, verifies, wanted, decode = Score(), [], None, False
            k = 0
            for e in request:
                if e[0] == 'decode_start':
                    decode = True
                elif e[0] == 'wanted':
                    wanted = unpack(e[2])
                elif e[0] == 'resident' and e[1] == 'verify':
                    verifies.append(unpack(e[3]))
                elif e[0] == 'commit' and decode:
                    score.add(e[1], verifies[k], wanted)
                    k += 1
            out.append(score)
    total = Score()
    for s in out:
        total.merge(s)
    return total.report(), [s.report() for s in out]


_STATE = {}


def _init(paths, predictor=None):
    import torch
    torch.set_num_threads(1)
    predictor = predictor or Predictor()
    _STATE['predictor'] = predictor
    _STATE['data'] = [(path, load_events(path), tapes_for(path, predictor)) for path in paths]


def _run(cfg, per_request=False):
    return dict(config=cfg, **simulate(_STATE['data'], _STATE['predictor'], cfg, per_request))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', action='append', required=True)
    parser.add_argument('--config', action='append', default=[],
                        help="JSON overrides of BASE (the recorded runs' settings), or 'new' for hot_policy.DEFAULTS; repeatable")
    parser.add_argument('--sweep', help='JSON dict of key -> list of values (cartesian product)')
    parser.add_argument('--per-request', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    predictor = Predictor()
    for path in args.events:
        tapes_for(path, predictor)
    configs = [NEW if c == 'new' else json.loads(c) for c in args.config] or [{}]
    if args.sweep:
        grid = json.loads(args.sweep)
        keys = list(grid)
        configs = []
        for values in itertools.product(*grid.values()):
            cfg = {}
            for k, v in zip(keys, values):
                if k in BASE['jt']:
                    cfg.setdefault('jt', {})[k] = v
                else:
                    cfg[k] = v
            configs.append(cfg)
    results = []
    if args.jobs > 1:
        import multiprocessing
        with multiprocessing.get_context('spawn').Pool(args.jobs, _init, (args.events,)) as pool:
            for result in pool.imap(partial(_run, per_request=args.per_request), configs):
                results.append(result)
                print(json.dumps(result), flush=True)
    else:
        _init(args.events, predictor)
        for cfg in configs:
            results.append(_run(cfg, args.per_request))
            print(json.dumps(results[-1]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
