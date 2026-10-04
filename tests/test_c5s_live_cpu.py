"""step 3b c5s: CPU smoke test of the live capture plumbing (sm120/serve/nq_c5s.C5Capture: runner pre/post hooks, per-layer
ring writes, prefill counts, drain thread -> streaming/c5s.py), with torch.cuda stubbed out (no GPU is touched).

Two requests (prefill step + MTP-like decode steps of 4 rows with rejected drafts + a prefilling short step) go through
the capture; the same steps are fed directly to a reference c5s.C5S (rows computed the way the capture defines them:
uint8 ids, fp16 w, fp32 sum x^2, bf16 projection -> fp16). Checks: identical finalized rows, identical feature state
and identical final mC.
usage: CUDA_VISIBLE_DEVICES= /data/Jarrel/coord/memjob.sh 16 /data/Jarrel/nq-algo/venv/bin/python tests/test_c5s_live_cpu.py"""
import os, sys, time, contextlib, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO + '/streaming'); sys.path.insert(0, REPO + '/sm120/serve')


class _Ev:
    def record(s, *a): pass
    def query(s): return True


torch.cuda.Stream = lambda *a, **k: None
torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.Event = _Ev
torch.cuda.is_current_stream_capturing = lambda: False
torch.Tensor.pin_memory = lambda s, *a, **k: s
import c5s as C5, nq_c5s as NC, nq_tfcap as TC

CK = os.environ.get('C5S_CKPT', REPO + '/streaming/ckpt/C3k.pt')
L_ = list(range(3, 78)); H = 64; rng = np.random.default_rng(0); torch.manual_seed(0)


class RT:
    ncap = 0


def main():
    Pl = C5.C5S(CK, threads=4, maxlag=10 ** 9); Pr = C5.C5S(CK, threads=4, maxlag=10 ** 9)
    os.environ['NQ_C5S_POLL_MS'] = '1'; os.environ['NQ_C5S_LOG_S'] = '0'
    cap = NC.C5Capture(RT(), L_, H, torch.device('cpu'), Pl)
    rows_l, rows_r = [], []
    for P, out in ((Pl, rows_l), (Pr, rows_r)):
        e0 = P.R.emit
        P.R.emit = (lambda e0, out: lambda i, s_, h, t: (out.append((np.array(i), np.array(s_), np.array(h), t)), e0(i, s_, h, t)))(e0, out)
    pfc = {}
    def model_step(rid, q, pos, tok, prefilling=False):
        T = len(pos)
        cap.pre(None, torch.tensor(tok, dtype=torch.int64), torch.tensor(pos, dtype=torch.int64))
        I = np.zeros((T, 75, 8), np.uint8); W = np.zeros((T, 75, 8), np.float16); X = np.zeros((T, 75), np.float32)
        HP = np.zeros((T, 8, 256), np.float16); cnt = np.zeros((75, 256), np.int64)
        for i, L in enumerate(L_):
            x = torch.randn(T, H) * 3; ids = torch.from_numpy(np.argsort(rng.random((T, 256)), 1)[:, :8].copy())
            w = torch.rand(T, 8)
            cap.layer(L, x, w, ids)
            I[:, i] = ids.numpy(); W[:, i] = w.to(torch.float16).numpy(); X[:, i] = x.float().square().sum(-1).numpy()
            k = cap.pl.get(L)
            if k is not None: HP[:, k] = (x.to(torch.bfloat16) @ cap.R[k]).to(torch.float16).numpy()
            cnt[i] = np.bincount(ids.reshape(-1).numpy(), minlength=256)
        cap.post(None, rid=rid, prefilling=prefilling)
        if T > TC.CAP_MAXT: Pr.R.prefill(q, T, cnt)
        else: Pr.R.step(q, prefilling, np.asarray(pos), np.asarray(tok), I, W, X, HP)
    nrow = 0
    for q, (rid, n) in enumerate((('req-a', 150), ('req-b', 211))):
        model_step(rid, q, list(range(100)), list(rng.integers(0, 150000, 100)))           # prefill (counts only)
        model_step(rid, q, list(range(100, 103)), [5, 6, 7], prefilling=True)            # short prefilling step: dropped
        p = 103
        while p < 103 + n:
            T = min(4, 103 + n - p); acc = int(rng.integers(0, T))
            tok = [154842 if (q == 0 and p == 160) else int(rng.integers(0, 150000)) for _ in range(T)]
            model_step(rid, q, list(range(p, p + T)), tok)
            p += acc + 1
        nrow += n
    Pr.R._flush()
    t0 = time.time()
    while time.time() - t0 < 60:                       # drain thread catches up (last request's rows stay pending there)
        time.sleep(0.05)
        if len(rows_l) + len(Pl.R.pend) >= nrow and not cap.pend: break
    time.sleep(0.2); Pl.R._flush()
    while Pl.ready: Pl.run_pending()
    Pr.run_pending()
    ok = len(rows_l) == len(rows_r) == nrow
    ok &= all(all(np.array_equal(a, b) for a, b in zip(x[:3], y[:3])) and x[3] == y[3] for x, y in zip(rows_l, rows_r))
    F, G = Pl.F, Pr.F
    ok &= F.n == G.n and F.seg == G.seg == 2 and F.starts == G.starts and np.array_equal(F.last, G.last)
    ok &= all(np.array_equal(F.Es[h], G.Es[h]) and np.array_equal(F.Ec[h], G.Ec[h]) for h in C5.HLS)
    ok &= np.array_equal(F.ring_s, G.ring_s) and np.array_equal(F.pf, G.pf)
    ok &= Pl.cur is not None and Pr.cur is not None and Pl.cur[1] == Pr.cur[1] and np.array_equal(Pl.cur[0], Pr.cur[0])
    print(f'rows live {len(rows_l)} ref {len(rows_r)} expected {nrow}; refreshes {Pl.st["refresh"]} fwd live {Pl.st["fwd"]} '
          f'(skipped {Pl.st["skipped"]}); last refresh row {Pl.cur[1]}; pf row-sum {float(F.pf.sum(1).mean()):.2f} (8 = all prefill tokens); lost {cap.lost}')
    print('PASS' if ok else 'FAIL'); return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
