"""nq-tapad unit tests (CPU only, no GPU): boot probe (streaming/tap_probe.py) on a real record file, read-only and few reads
(PROBE_FILE, default the prod p4rec rank0.bin, also read by the live server: keep PROBE_N small), small /dev/shm file
(O_DIRECT or the buffered fallback); the ADAPT rate paths with a fake io_all (followers: saturated EWMA, unsaturated upward-only, one
sample per publish, rank-count change keeps averages); ADAPT=0 follower path vs REF (origin/spark-b175) scheduler_tap.
  python tests/test_tap_adapt_unit.py"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE) + '/streaming')
import numpy as np
import tap_adapt_sim as SIM                               # clears NQ_TAP_* and sets the sim constants
import scheduler_tap as TS, tap_probe as TP

PF = os.environ.get('PROBE_FILE', '/home/jarrelscy/nq-p4rec/hf/rank0.bin'); PRB = int(os.environ.get('PROBE_RB', '2560000'))
PN = int(os.environ.get('PROBE_N', '64')); ok = True


def chk(nm, c, msg=''):
    global ok
    ok &= bool(c); print(f'  {"PASS" if c else "FAIL"} {nm} {msg}', flush=True)


# ---- probe
if os.path.exists(PF):
    r = TP.probe(PF, PRB, n=PN, qd=8, max_s=3.0)
    chk('probe real file', r['n'] == PN and r['bytes'] == PN * PRB and r['GBps'] > 0.2 and r['s'] < 3.5,
        f"n {r['n']} {r['s']:.3f}s {r['GBps']:.2f} GB/s p50 {r['p50_ms']:.2f} ms p99 {r['p99_ms']:.2f} ms direct {r['direct']}")
    r = TP.probe(PF, PRB, n=PN, qd=4, alt=PF, qd_alt=4, max_s=3.0, seed=1)   # dual path API (same file twice: split + sum)
    chk('probe dual split', len(r['drives']) == 2 and sum(d['n'] for d in r['drives']) == r['n'] == PN,
        f"{r['GBps']:.2f} GB/s = " + ' + '.join(f"{d['GBps']:.2f} ({d['n']})" for d in r['drives']))
    r = TP.probe(PF, PRB, n=10 ** 6, qd=2, max_s=0.05)
    chk('probe max_s stops early', r['n'] < 10 ** 6 and r['s'] < 0.5, f"{r['n']} reads in {r['s']:.3f}s")
else:
    print(f'  SKIP real-file probe ({PF} missing)')
with tempfile.NamedTemporaryFile(dir='/dev/shm', prefix='nq_tapad_') as f:
    f.write(os.urandom(4 * 65536)); f.flush()
    r = TP.probe(f.name, 65536, n=16, qd=2)
    chk('probe tmpfs small file', r['n'] == 16 and r['bytes'] == 16 * 65536, f"direct {r['direct']} (tmpfs O_DIRECT ok on 6.6+, else buffered)")

# ---- probe_rate / ADAPT follower paths (fake io_all)
T = [0.0]; S = SIM.mk(TS, dict(NQ_TAP_ADAPT='1'), lambda: T[0])
rr = SIM.RBR
S.probe_rate(2.0); chk('probe_rate sets rate0', abs(S.rate0 * rr / 1e9 - 2.0) < 1e-9 and S.peak is None and abs(S.stats['ad_gbps'] - 2.0) < 1e-9)
S.probe_rate(6.0); IO = {}
S.io_all = lambda: IO
def pub(r, t, g, sat): IO[r] = dict(t_wall=t, dt_s=0.5, delivered_GBps=g, eng_waiting=8 if sat else 0, eng_reading=8 if sat else 1,
                                    ops_outstanding=4, slot_wait=0, backlog=0)
for r in range(4): pub(r, 0.0, 6.0, True)
T[0] = 0.1; S._rates(T[0], 0)
g = lambda r: S.peak[r] * (S.rb / 4) / 1e9
chk('4 ranks: peak sized from io_all', len(S.peak) == 4)
p1 = g(1)
for k in range(1, 9): pub(1, 0.5 * k, 2.0, True); T[0] = 0.1 + 0.5 * k; S._rates(T[0], 0)
chk('follower saturated 2 GB/s: EWMA down (4 s, hl 1 s)', abs(g(1) - (2.0 + 4.0 * 0.5 ** 4)) < 1e-6, f'{p1:.2f} -> {g(1):.3f} (expect 2.25)')
x = g(1); S._rates(T[0] + 0.3, 0); chk('same t_wall: no new sample', g(1) == x)
pub(2, 1.0, 1.0, False); S._rates(T[0] + 0.4, 0); chk('follower unsaturated low: ignored', abs(g(2) - 6.0) < 1e-6, f'{g(2):.3f}')
pub(2, 1.5, 9.0, False); S._rates(T[0] + 0.5, 0); chk('follower unsaturated high: raises', g(2) > 6.0, f'{g(2):.3f}')
y = g(1); del IO[3]; S._rates(T[0] + 0.6, 0); chk('rank count change keeps averages', len(S.peak) == 3 and g(1) == y)
S.tcap = 3.0e9 / (S.rb / 4); IO[2]['t_wall'] = 9; S._rates(T[0] + 0.7, 0); chk('cap applies', g(2) <= 3.0 + 1e-9)

# ---- ADAPT=0 follower path == REF (lockstep peak / lat with the same fake io_all)
RM = SIM.ref_module(os.environ.get('REF', 'origin/spark-b175')); T = [0.0]; A = SIM.mk(RM, {}, lambda: T[0]); B = SIM.mk(TS, {}, lambda: T[0])
rng = np.random.default_rng(0); IO = {}; A.io_all = B.io_all = lambda: IO; same = True
for k in range(400):
    for r in range(4): IO[r] = dict(t_wall=k // 3, dt_s=0.5, delivered_GBps=float(rng.uniform(1, 8)), eng_waiting=int(rng.integers(0, 9)),
                                   eng_reading=int(rng.integers(0, 9)), ops_outstanding=int(rng.integers(0, 300)), slot_wait=int(rng.integers(0, 50)),
                                   backlog=int(rng.integers(0, 50)), op_p50_ms=float(rng.uniform(1, 50)))
    if k == 200: del IO[3]
    T[0] = k * 0.17; A.n_land = B.n_land = k * 7
    same &= A._rates(T[0], k) == B._rates(T[0], k) and np.array_equal(A.peak, B.peak)
chk('ADAPT=0 follower _rates == REF (400 refreshes, rank drop)', same)
print('UNIT PASS' if ok else 'UNIT FAIL'); sys.exit(0 if ok else 1)
