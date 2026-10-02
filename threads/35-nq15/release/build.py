"""T35 b175 release build: streaming/repack.py (nq-res-v2 code at wt_main) per finished layer of enc_b175 into the
start.sh root layout at OUT (rank{r}.bin preallocated at full 75-layer size, records filled in place, res/rank{r}/L{L}.pt).
Layers are taken as soon as enc_b175/fin/L{L}.json has rc 0; up to --par repacks at once (repack.py merges rank{r}.json
under a lock).  Per-layer marker done/L{L}.json (sha256 of each rank's record region + res file) -> resumable.
Exits when all 75 are marked."""
import os, sys, json, time, hashlib, subprocess, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/tmp/nestquant/35-nq15/enc_b175')
ap.add_argument('--out', default='/tmp/nestquant/35-nq15/release/repo')
ap.add_argument('--done', default='/tmp/nestquant/35-nq15/release/done')
ap.add_argument('--code', default='/tmp/nestquant/35-nq15/wt_main')
ap.add_argument('--rec-bytes', type=int, default=2854912)
ap.add_argument('--par', type=int, default=4); ap.add_argument('--nt', type=int, default=4)
ap.add_argument('--order', default='7-10,40-50,11-39,51-77,3-6')
a = ap.parse_args()
PY = '/home/coder/git/glm52/.venv/bin/python'
RUN = '/tmp/nestquant/35-nq15/release/run_repack.py'
NE, L0, NL, TP = 256, 3, 75, 4
order = [L for p in a.order.split(',') for L in (range(int(p.split('-')[0]), int(p.split('-')[1]) + 1) if '-' in p else [int(p)])]
assert sorted(order) == list(range(3, 78))
for r in range(TP):                       # preallocate (sparse) at full size
    p = f'{a.out}/rank{r}.bin'
    fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size < NL * NE * a.rec_bytes:
        os.ftruncate(fd, NL * NE * a.rec_bytes)
    os.close(fd)
def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime()), m, flush=True)
def fin_ok(L):
    try: return json.load(open(f'{a.root}/fin/L{L}.json')).get('rc') == 0
    except Exception: return False
def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def mark(L, secs):
    ent = dict(layer=L, rec_bytes=a.rec_bytes, seconds=round(secs, 1), ranks={})
    for r in range(TP):
        idx = json.load(open(f'{a.out}/rank{r}.json'))
        assert str(L) in idx['layers'] and idx['rec_bytes'] == a.rec_bytes, (L, r)
        assert sorted(int(e) for e in idx['layers'][str(L)]['experts']) == list(range(NE)), (L, r)
        off, n = (L - L0) * NE * a.rec_bytes, NE * a.rec_bytes
        with open(f'{a.out}/rank{r}.bin', 'rb') as f:
            f.seek(off); b = f.read(n)
        assert len(b) == n and any(b[i] for i in range(0, n, 4096)), ('empty region', L, r)
        rp = f'{a.out}/res/rank{r}/L{L}.pt'
        ent['ranks'][r] = dict(rec_sha256=hashlib.sha256(b).hexdigest(), res_sha256=sha_file(rp), res_bytes=os.path.getsize(rp))
    json.dump(ent, open(f'{a.done}/L{L}.json.tmp', 'w'), indent=1); os.replace(f'{a.done}/L{L}.json.tmp', f'{a.done}/L{L}.json')
run = {}
while True:
    for L, (pr, t0, lf) in list(run.items()):
        if pr.poll() is not None:
            del run[L]; lf.close()
            if pr.returncode == 0:
                try: mark(L, time.time() - t0); log(f'L{L} done in {time.time()-t0:.0f}s')
                except Exception as e: log(f'L{L} MARK FAIL {e!r}')
            else: log(f'L{L} repack rc={pr.returncode}')
    done = {L for L in order if os.path.exists(f'{a.done}/L{L}.json')}
    if len(done) == NL and not run:
        log('ALL 75 layers built'); open(f'{a.done}/ALL_DONE', 'w').write(time.strftime('%FT%TZ', time.gmtime())); break
    for L in order:
        if len(run) >= a.par: break
        if L in done or L in run or not fin_ok(L): continue
        if os.path.exists(f'{a.done}/L{L}.fail'): continue
        lf = open(f'/tmp/nestquant/35-nq15/release/logs/L{L}.log', 'a')
        env = dict(os.environ, REC_BYTES=str(a.rec_bytes), NT=str(a.nt), OMP_NUM_THREADS=str(a.nt), MKL_NUM_THREADS=str(a.nt), OPENBLAS_NUM_THREADS='1')
        pr = subprocess.Popen(['nice', '-n', '10', PY, RUN, a.code, a.root, a.out, f'{L}-{L}'], stdout=lf, stderr=subprocess.STDOUT, env=env)
        sys.argv  # noqa
        run[L] = (pr, time.time(), lf); log(f'L{L} start pid {pr.pid}')
    time.sleep(10)
