"""nq-tap wedge after a prefill-borrow reclaim (A/B #1, 2026-10-03): live the leader's tap scheduler issued zero reads from
a borrow reclaim until the next borrow epoch (ups = downs = 0, budget_cut only, q_todo pinned at 731, lat 76-102 tok > H 64).
Livelock: the reclaim gives S.slots / S.nf back with the normal pool full and unbalanced per layer -> nfree = 0 while
layers under nf keep emitting free-slot pairs -> those (and re-appended duplicates) pile into todo -> len(todo) feeds the
qreal latency model -> lat > H -> budget 0 -> the budget cut also blocks eager evictions, the only source of downs -> no
slot ever frees. This drives TapScheduler.step (python host loop) through warm decode, a borrow epoch (slots += X,
nf = 155) and its reclaim (borrowed-pool residents forced to state 3, slots -= X, nf back) and asserts reads resume
(for NQ_TAP_TODO_FIX=1 and =2 (no nf shrink), and that =0, the step 2 rc code, still wedges here); prints the hot share
of activated experts in the first 500 steps after the reclaim and after.
  python tests/test_tap_reclaim.py          (NFILES routing-log files, default 8)"""
import os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../streaming'); sys.path.insert(0, HERE)
import numpy as np
import test_tap_parity as TP
from test_tap_parity import routing_steps, LeaderX, cfgs, LAYERS, NE

class SlowS:
    """predictor: S = count EMA (half-life 256 tokens) + fixed per-expert prior, refreshed every 16 tokens (slowly moving
    like jF, unlike TP.StubS which redraws S each refresh)"""
    bs = None
    def __init__(s, seed, pw=0.1):
        r = np.random.default_rng(seed); s.pw = pw; s.prior = r.gamma(0.5, 1.0, (75, NE)); s.e = np.zeros((75, NE)); s.n = 0; s.S = None
    def step(s, c, ntok, tid, nr, sal=None):
        s.e = s.e * 0.5 ** (ntok / 256) + c; s.n += ntok
        if s.n < 16: return False
        s.n = 0; s.S = (100 * s.e / max(s.e.sum() / 75, 1e-9) + s.pw * s.prior).astype(np.float32); return True
    def target(s, r): return None
    def order_score(s, r): return None
    def close(s): pass


class RateX:
    """fake leader executor at a finite read rate: lands at most `cap` ups per step (FIFO; a few read errors), releases
    downs the next step"""
    def __init__(s, seed, cap=24): s.rng = np.random.default_rng(seed); s.upq = collections.deque(); s.dq = []; s.cap = cap
    def apply(s, ups, downs): s.upq.extend(ups); s.dq += downs
    def tick(s, S):
        for _ in range(min(s.cap, len(s.upq))):
            k = s.upq.popleft()
            if s.rng.random() < 0.01: S.failed(*k, read_error=True)
            else: S.landed(*k)
        for k in s.dq: S.released(*k)
        s.dq = []


ENV = dict(NQ_TAP_CTL='/nonexistent/nq_tap_ctl', NQ_TAP_H='64', NQ_TAP_QREAL='1', NQ_TAP_RATE_GBPS='2', NQ_S3_TRACK='1')
EXTRA, PF_NF = 6000, 155


def run(hl, steps, seed=0, pw=0.1, cap=8, env=None):
    k0, _ = cfgs()
    S = TP.mk_new(k0, SlowS(seed + 1, pw), dict(ENV, **(env or {})), TP.Clock(seed))
    X = RateX(seed, cap); S.xq = lambda: len(X.upq)
    init = [(L, E) for L in LAYERS for E in k0['dflt'][L] if E not in k0['fixed'][L]][:k0['slots']]
    for L, E in init: S.state[S.li[L], E] = 1
    X.apply(init, [])
    n = len(steps); b0, b1 = n // 4, n // 2                         # warm | borrow epoch | after the reclaim
    pre = collections.Counter(); post = collections.Counter(); zrun = mz = 0
    for i, (c, ntok, tid, nr) in enumerate(steps):
        if i == b0:                                                 # borrow: leader pool + lookahead width grow
            S.slots += EXTRA; S.nf = PF_NF
        if i == b1:                                                 # reclaim (nq_pb._reclaim leader side)
            for k in X.upq: S.landed(*k)                            # x_reclaim waits for every op on the pool
            X.upq.clear()
            occ = [(int(a), int(b)) for a, b in zip(*np.nonzero((S.state == 1) | (S.state == 2)))]
            rng = np.random.default_rng(seed + 7); over = len(occ) - (S.slots - EXTRA)
            for j in rng.permutation(len(occ))[:max(over, 0)]:     # the borrowed-pool residents: forced to level 2
                a, b = occ[j]; S.state[a, b] = 3; X.dq.append((S.layers[a], b))
            S.slots -= EXTRA; S.nf = k0['nf']
        c = c.astype(np.float64); S.clock.adv(ntok)
        if i >= b1:                                                 # hot share of the activated experts, before this step's I/O
            a = c > 0; h = int((a & ((S.state == 2) | S.fixed)).sum()); w = 'h500' if i < b1 + 500 else 'hrest'
            post[w] += h; post[w + 'n'] += int(a.sum())
        r0 = S.stats['refreshes']
        u, d = S.step(c, ntok, tid, nr, sal=c)
        X.apply(u, d); X.tick(S)
        if b0 // 2 <= i < b0: pre['ups'] += len(u); pre['ref'] += S.stats['refreshes'] - r0
        if i >= b1 + 50:
            post['ups'] += len(u); post['downs'] += len(d); post['ref'] += S.stats['refreshes'] - r0
            tl = len(S.todo)
            if S.stats['refreshes'] > r0: zrun = zrun + 1 if tl and not u else 0; mz = max(mz, zrun)   # todo waits, nothing issues
    tl = len(S.todo)
    return pre, post, mz, tl, dict(S.stats)


def main():
    steps = [x for x in routing_steps(int(os.environ.get('NFILES', '8'))) if x[1] <= 16]
    steps = (steps * (1 + 4000 // max(len(steps), 1)))[:4000]
    ok = True
    for fix in ('1', '2', '0'):                # 0 = step 2 rc behaviour: must wedge (the test sees the bug); 2 = no nf shrink
        for hl in ('py',):                     # clean-d: python host loop only
            pre, post, mz, tl, st = run(hl, steps, env=dict(NQ_TAP_TODO_FIX=fix))
            wedged = mz >= 50 or post['ups'] < post['ref']
            print(f'fix={fix} {hl}: warm ups {pre["ups"]} / {pre["ref"]} refreshes; after reclaim ups {post["ups"]} downs {post["downs"]} '
                  f'refreshes {post["ref"]}, longest stuck run (todo waiting, no ups) {mz}, todo {tl}, budget_cut {st["budget_cut"]} '
                  f'shrink {st.get("shrink_evict", 0)} nf_shrink {st.get("nf_shrink_evict", 0)} todo_full_skip {st.get("todo_full_skip", 0)} '
                  f'hot first 500 post-reclaim {post["h500"] / max(post["h500n"], 1):.3f} rest {post["hrest"] / max(post["hrestn"], 1):.3f} '
                  f'-> {"WEDGED" if wedged else "ok"}')
            ok &= pre['ups'] > 0 and wedged == (fix == '0')
    print('PASS' if ok else 'FAIL'); return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
