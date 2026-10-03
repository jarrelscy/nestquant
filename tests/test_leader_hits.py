"""Leader host loop: every routed hit must reach S.step (CPU only, no GPU, no vllm).
Drives the real nq_vllm Runtime.loop (leader: F None) with a fake executor, a recording scheduler and CPU hit counters.
Replays real decode routing (GLM-5.3 routing logs, layers 3-77) as a multi-poll sequence: each model step's per-layer
hits land over 1-3 polls, split at a random layer boundary (layer 0's hits always in the first part, like the kernel's
layer order), with some polls that catch nothing and some polls spanning two steps. Asserts:
  sum of every counts matrix passed to S.step == sum of every hit written to the counters   (per layer, per expert)
  sum of ntok passed to S.step == tokens replayed
Known to FAIL on 4c9c8e0 / clean-d de22db2: the loop advances `prev` on every poll but only calls S.step when the poll
saw new layer-0 hits (ntok>0), so polls that only catch later layers' hits are dropped (nq_vllm.py loop, `if ntok>0:`).
run: CUDA_VISIBLE_DEVICES= /data/Jarrel/nqenv/bin/python tests/test_leader_hits.py         this checkout
     TREE=/path/to/nestquant ... or REF=4c9c8e0 ...                                         another tree / git ref
     NSTEPS (default 3000), SEED"""
import os, sys, types, glob, random, tempfile, subprocess, threading, logging
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
logging.basicConfig(level=logging.ERROR)
NE = 256; TOPK = 8; LAYERS = list(range(3, 78))


def tree():
    if os.environ.get('REF'):
        d = tempfile.mkdtemp(prefix='nq_ref_')
        subprocess.run(f"git -C {REPO} archive {os.environ['REF']} sm120 streaming | tar -x -C {d}", shell=True, check=True)
        return d
    return os.environ.get('TREE', REPO)


def load_nq_vllm(T):
    for n in ('vllm', 'vllm.logger'): sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules['vllm.logger'].init_logger = lambda name: logging.getLogger(name)
    for n in ('moe', 'p4rec', 'build'): sys.modules.setdefault(n, types.ModuleType(n))
    sys.path[:0] = [T + '/sm120/serve', T + '/streaming', T + '/sm120']
    os.environ.setdefault('NQ_HOME', T); os.environ.setdefault('NQ_SESSION_RESTORE', '0'); os.environ.setdefault('NQ_PREFILL_ADAPT', '0')
    import nq_vllm as NV
    assert NV.__file__.startswith(T), NV.__file__
    return NV


def decode_steps(n, seed):
    """real decode routing: list of [75, 256] int counts per model step (ntok <= 16 = decode/MTP verify)"""
    fs = sorted(glob.glob('/data/Jarrel/routing_logs/glm5.3-arvq-v2/seg-*.npz'))
    fs = [fs[i] for i in sorted(random.Random(seed).sample(range(len(fs)), min(len(fs), 12)))]
    out = []
    for f in fs:
        z = np.load(f); ex = z['experts'].astype(np.int64); st = z['step']
        for sv in np.unique(st):
            m = st == sv
            if m.sum() > 16: continue
            e = ex[m][:, 3:78, :]
            out.append(np.bincount((e + (np.arange(75) * NE)[None, :, None]).ravel(), minlength=75 * NE).reshape(75, NE))
            if len(out) >= n: return out
    return out


def polls(steps, seed):
    """-> list of per-poll hit increments [75, 256]: each step split over 1-3 polls at random layer cuts, empty polls,
    and sometimes the tail of a step merged with the head of the next one"""
    rng = random.Random(seed); P = []; carry = None
    for c in steps:
        k = rng.choice((1, 2, 2, 3)); cuts = sorted(rng.sample(range(1, 75), k - 1)); b = [0] + cuts + [75]
        parts = []
        for i in range(k):
            p = np.zeros_like(c); p[b[i]:b[i + 1]] = c[b[i]:b[i + 1]]; parts.append(p)
        if carry is not None: parts[0] = parts[0] + carry; carry = None
        if k > 1 and rng.random() < 0.2: carry = parts.pop()
        for p in parts:
            P.append(p)
            if rng.random() < 0.1: P.append(np.zeros_like(c))
    if carry is not None: P.append(carry)
    return P


class RecS:
    """scheduler stand-in: records what the loop feeds it; no ops"""
    def __init__(s, nl):
        s.fixed = np.zeros((nl, NE), bool); s.state = np.zeros((nl, NE), np.int8); s.wants_sal = False; s.P = None
        s.predictor_name = 'rec'; s.stats = dict(ups=0, downs=0); s.got = np.zeros((nl, NE), np.int64); s.ntok = 0; s.calls = 0; s.pin = None
    def step(s, c, ntok, token_ids=None, new_request=False, sal=None):
        s.got += np.asarray(c, np.int64); s.ntok += ntok; s.calls += 1; return [], []
    def level(s): return np.where(s.fixed, 4, s.state)


def main():
    T = tree(); NV = load_nq_vllm(T)
    n = int(os.environ.get('NSTEPS', '3000')); seed = int(os.environ.get('SEED', '0'))
    steps = decode_steps(n, seed); P = polls(steps, seed)
    want = np.sum(steps, 0); wtok = sum(int(c[0].sum()) // TOPK for c in steps)
    nl = len(LAYERS); S = RecS(nl)
    hits = {L: torch.zeros(NE, dtype=torch.int32) for L in LAYERS}
    rt = types.SimpleNamespace()
    it = iter(P); done = threading.Event()

    class X:                                   # the loop polls the executor first each iteration: land the next poll's hits
        lat = []
        def poll(s_, S_, issue=True):
            p = next(it, None)
            if p is None: rt.stop = True; done.set(); return
            for i, L in enumerate(LAYERS): hits[L] += torch.from_numpy(p[i].astype(np.int32))
        def apply(s_, ups, downs, S_): pass
    class Ev:
        def wait(s_, t): pass
        def clear(s_): pass
    rt.__dict__.update(dev=None, stop=False, wake=Ev(), cv=threading.Condition(), PB=None, ncap=0, in_iter=False, F=None, X=X(), S=S,
                       LA=None, SR=None, log=None, L_=LAYERS, lay={L: dict(hits=hits[L]) for L in LAYERS}, rank=0, pb_pause=0)
    torch.cuda.set_device = lambda d: None
    NV.Runtime.loop(rt)
    assert done.is_set() and getattr(rt, 'err', None) is None, getattr(rt, 'err', None)
    tot = int(want.sum()); got = int(S.got.sum()); lost = want - S.got
    per = lost.sum(1)
    print(f'{T}: {len(steps)} decode steps over {len(P)} polls, {tot} routed hits, {wtok} tokens')
    print(f'  S.step calls {S.calls}, hits reaching S.step {got} ({got / tot:.2%}), ntok {S.ntok}/{wtok}')
    if per.any():
        w = np.argsort(-per)[:5]
        print('  lost hits, worst layers: ' + ', '.join(f'L{LAYERS[i]} {int(per[i])} ({per[i] / max(1, want[i].sum()):.1%})' for i in w))
        print(f'  lost share layer 3: {per[0] / max(1, want[0].sum()):.1%}, layers 40-77: {per[37:].sum() / max(1, want[37:].sum()):.1%}')
    ok = (lost == 0).all() and S.ntok == wtok
    print('PASS' if ok else 'FAIL: hits dropped before S.step'); return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
