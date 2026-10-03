"""NQ_S3_EXT model hook (streaming/scheduler_tap.py) vs p-online pols.Combo e1 arms (CPU, needs the p-online data + E1 preds).
The live TapScheduler is driven as in test_s3_adm.py (olib.budget_sim policy); the hook replays the E1 prediction file
p-online's Combo(e1=, e1f=) reads (tests/s3x_replay.py:replay), so live keys must equal Combo.score at every refresh and the
budget_sim numbers must equal res_stack:
  env NQ_S3=1 NQ_S3_X=1 NQ_S3_XW=0.15 NQ_S3_RESET=0 NQ_S3_EXT=..:replay        -> 'e1mix.15+g25.9.25+B5n22'
  env NQ_S3_TRACK=1 NQ_S3_EXT=..:replay, ctl 's3=1 s3x=1 s3xw=0.2 s3p=0.5 s3rst=0' -> 'e1mix.2+g25.9.5+B5n22'
  history check: NQ_S3=1 (per-request reset) with a hook that checks ctx.ids / sal / rows / new_request against the stream
  and returns None (-> = ComboR g25.9.25+B5n22); then the hook cost (none / feats probes, s3xt) per refresh.
  python tests/test_s3_ext.py [tb-<task> | gen-<key> ...]   (default tb-cad-model gen-lean4_explain0)"""
import os, sys, json, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import test_s3_adm as A
from test_s3_adm import OL, PL, ARM, PO
E1D = '/data/Jarrel/nq-step3/e1/out/c64_base_xl'
HOOK = HERE + '/s3x_replay.py'
A.S3ENV = A.S3ENV + ('NQ_S3_X', 'NQ_S3_XW', 'NQ_S3_EXT', 'NQ_S3_XRING')


def multi_tok():
    """a step with ntok > 1 routed rows: split into ntok rows, same multiset per layer, no expert twice in a row, sal kept"""
    rng = np.random.default_rng(3); SS = np.zeros((4, 75, 256), np.float32)
    T = A.boot(SS, dict(NQ_S3='1', NQ_S3_EXT=HOOK + ':none')); ok = True
    for ntok in (2, 3, 5):
        rows = np.stack([np.argsort(rng.random((75, 256)), 1)[:, :8] for _ in range(ntok)])
        c = np.zeros((75, 256)); sl = np.zeros((75, 256))
        for r in rows: c[OL.ar, r] += 1; sl[OL.ar, r] += rng.random((75, 8))
        T._s3_add(c, ntok, sl); Z = T.s3st; got = Z['xi'][Z['xp'] - ntok:Z['xp']]; gs = Z['xs'][Z['xp'] - ntok:Z['xp']]
        for l in range(75):
            ok &= sorted(got[:, l].ravel().tolist()) == sorted(rows[:, l].ravel().tolist())
            ok &= all(len(set(got[k, l].tolist())) == 8 for k in range(ntok))
        ok &= np.allclose(gs.sum(0).sum(1), sl.sum(1), rtol=1e-5)
    T._s3_add(c * 0, 1, None); ok &= T.s3xbad == 1                   # inconsistent counts: skipped + counted
    print(f'multi-token steps split into rows (ntok 2/3/5) + bad-step skip: {ok}', flush=True)
    return ok


def main():
    names = sys.argv[1:] or ['tb-cad-model', 'gen-lean4_explain0']; ok = multi_tok()
    for nm in names:
        st = A.load(nm); SS = st['S']; res = json.load(open(f'{PO}/res_stack_{st["name"]}.json'))
        kind, _, key = nm.partition('-'); e1f = f'{E1D}/pred_{kind}_{key}.npy'; os.environ['NQ_S3X_REPLAY'] = e1f
        ctl = tempfile.NamedTemporaryFile('w', suffix='.nq_tap_ctl', delete=False); ctl.write('s3=1 s3x=1 s3xw=0.2 s3p=0.5 s3rst=0\n'); ctl.close()
        print(f'{st["name"]}: {len(st["ids"])} rows, {len(st["rstart"])} requests, E1 {e1f}', flush=True)
        cases = [('e1mix.15+g25.9.25+B5n22 env', A.boot(SS, dict(NQ_S3='1', NQ_S3_X='1', NQ_S3_XW='0.15', NQ_S3_RESET='0', NQ_S3_EXT=HOOK + ':replay')),
                  PL.Combo(st, **dict(ARM, e1=.15, e1f=e1f)), 'e1mix.15+g25.9.25+B5n22'),
                 ('e1mix.2+g25.9.5+B5n22 ctl', A.boot(SS, dict(NQ_S3_TRACK='1', NQ_S3_EXT=HOOK + ':replay', NQ_TAP_CTL=ctl.name)),
                  PL.Combo(st, **dict(ARM, p=.5, e1=.2, e1f=e1f)), 'e1mix.2+g25.9.5+B5n22')]
        for cn, T, ref, k in cases:
            r, L, dt = A.run(st, T, ref, None); e = res[k]['all']; dd = (r[0] - e[0], r[1] - e[1])
            good = L.bad == 0 and L.n > 0 and max(abs(dd[0]), abs(dd[1])) <= 0.02 and T.s3xt[0] == L.n and T.s3xbad == 0
            print(f'  {cn:32s} live {r[0]:.2f}/{r[1]:.2f}  keys vs Combo: {L.n} refreshes, max rel {L.worst:.1e}; p-online {e[0]:.2f}/{e[1]:.2f} '
                  f'(d {dd[0]:+.3f}/{dd[1]:+.3f}); hook calls {T.s3xt[0]}, {1e3 * T.s3xt[1] / max(T.s3xt[0], 1):.3f} ms/call  [{dt:.0f}s] '
                  + ('ok' if good else 'FAIL'), flush=True); ok &= good
        os.unlink(ctl.name)
        # history check: ctx.ids / sal / rows / new_request vs the stream (per-request reset), hook returns None
        T = A.boot(SS, dict(NQ_S3='1', NQ_S3_X='1', NQ_S3_EXT=HOOK + ':none', NQ_S3_XRING='256'))
        ids = st['ids']; sal = st['sal']; rs = sorted(int(x) for x in st['rstart']); chk = dict(n=0, bad=0, nr=0)
        def hist(cx):
            t = cx.tok; r0 = max(x for x in rs if x < t); n = min(t - r0, 256)
            want = np.sort(ids[t - n:t], axis=2).astype(np.int16)
            ws = np.take_along_axis(sal[t - n:t], np.argsort(ids[t - n:t], axis=2, kind='stable'), 2)
            good = (cx.rows == t - r0 and cx.ids.shape == want.shape and np.array_equal(cx.ids, want)
                    and np.allclose(cx.sal, ws, rtol=1e-6, atol=0))
            chk['n'] += 1; chk['bad'] += not good; chk['nr'] += cx.new_request
            if not good and chk['bad'] <= 3: print('   history DIFF at row', t, cx.rows, t - r0, cx.ids.shape, want.shape)
            return None
        T.s3s = None; T.s3['xf'] = 'inline'; T.s3xf0 = 'inline'; T.s3xfn = hist; T.s3xstate = {}
        r, L, dt = A.run(st, T, A.ComboR(st, **ARM), None)
        nrq = sum(1 for x in rs if 0 < x < (len(ids) // 16) * 16) + 1
        good = chk['bad'] == 0 and chk['n'] == L.n and L.bad == 0 and chk['nr'] == nrq
        print(f'  history (ring 256, reset)       {chk["n"]} calls, {chk["bad"]} diffs, new_request {chk["nr"]}/{nrq}; keys vs ComboR max rel {L.worst:.1e} '
              f'live {r[0]:.2f}/{r[1]:.2f}  ' + ('ok' if good else 'FAIL'), flush=True); ok &= good
        # hook cost in the host loop (ms per refresh): plain arm vs none / feats probes (ring 1024)
        for fn in ('-', 'none', 'feats'):
            env = dict(NQ_S3='1') if fn == '-' else dict(NQ_S3='1', NQ_S3_X='1', NQ_S3_EXT=HOOK + ':' + fn)
            T = A.boot(SS, env); N = min(len(ids), 4096); ts = 0.0; nref = 0
            for t in range(N):
                c = np.zeros((75, 256)); c[OL.ar, ids[t]] = 1.0; sl = np.zeros((75, 256)); sl[OL.ar, ids[t]] = sal[t]
                r0 = T.stats['refreshes']; t0 = time.perf_counter(); T.step(c, 1, None, t in rs, sal=sl); d = time.perf_counter() - t0
                if T.stats['refreshes'] != r0: ts += d; nref += 1
            print(f'  cost hook={fn:5s}: tap step at refresh {1e3 * ts / max(nref, 1):.2f} ms (incl. tap plan/pairs), hook '
                  f'{1e3 * T.s3xt[1] / max(T.s3xt[0], 1):.3f} ms/call over {T.s3xt[0]} calls', flush=True)
    return ok


if __name__ == '__main__':
    ok = main(); print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
