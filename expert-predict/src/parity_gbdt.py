"""Parity test (CPU only): online GBDTPredictor (nestquant/streaming/gbdt_predictor.py) fed token-by-token from a held-out
decode stream reproduces the offline pipeline's score matrices and C-sim share (cap 24 aggregate, lead 13, hm 0.5, R16)."""
import sys, time, json; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from gbdt_feats import *
sys.path.insert(0, '/data/Jarrel/nestquant/streaming'); import fixed_set; from gbdt_predictor import GBDTPredictor
import lightgbm as lgb
task = sys.argv[1] if len(sys.argv) > 1 else 'sound-change-cascade'; NCHK = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MODEL = '/data/Jarrel/nestquant/streaming/gbdt_p64_s5.txt'; HM = 0.5
d = load(task); ex = d['ex']; tok = d['tok']; req = d['req']; N = len(ex); nref = -(-N // 16)
fx, _, _ = fixed_set.load(); layers = list(range(L0, L0 + NL))
ps = GBDTPredictor(layers, fx, MODEL, hm=HM, mode='sync'); pa = GBDTPredictor(layers, fx, MODEL, hm=HM, mode='next_refresh')
Ss = np.zeros((nref + 1, NL, NE), np.float32); Sa = np.zeros_like(Ss); lofs = np.arange(NL)[:, None] * NE
t0 = time.time(); tref = 0.0; nr = 0
import os
if os.environ.get('REUSE'): Ss = np.load(f'{W}/feat/parity_Ss.npy'); Sa = np.load(f'{W}/feat/parity_Sa.npy'); N0 = 0
else: N0 = N
for t in range(N0):
    cnt = np.bincount((ex[t].astype(np.int64) + lofs).ravel(), minlength=NL * NE).reshape(NL, NE)
    nreq = t == 0 or req[t] != req[t - 1]
    t1 = time.perf_counter(); r1 = ps.step(cnt, 1, [int(tok[t])], nreq); dt = time.perf_counter() - t1
    if r1: tref += dt; nr += 1
    if pa.step(cnt, 1, [int(tok[t])], nreq): Sa[(t + 1) // 16] = pa.S
    if r1: Ss[(t + 1) // 16] = ps.S
pa.close(); np.save(f'{W}/feat/parity_Ss.npy', Ss); np.save(f'{W}/feat/parity_Sa.npy', Sa)
if N0: print(f'online pass {time.time() - t0:.0f}s, {nr} refreshes, sync step() at refresh {1e3 * tref / max(nr, 1):.2f} ms (features+predict, 4 thr); async late_waits {pa.stats["late_waits"]}', flush=True)
# offline reference for the first NCHK refreshes
bst = lgb.Booster(model_file=MODEL); cols = [FNAMES.index(f) for f in bst.feature_name()]; mx = 0.0
for b, cand, X, _, _, e256, top, _ in gen(d, np.zeros((3, NL, NE, KP), np.int64), 1, want_y=False):
    if b > NCHK: break
    s = np.zeros((NL, NE), np.float32); np.put_along_axis(s, cand, bst.predict(X.reshape(-1, X.shape[-1])[:, cols], num_threads=4).reshape(NL, -1).astype(np.float32), 1)
    np.put_along_axis(s, top, 1e3 + e256[np.arange(NL)[:, None], top], 1); mx = max(mx, float(np.abs(s - Ss[b]).max()))
print(f'max |S_online - S_offline| over first {NCHK} refreshes: {mx:.3g}', flush=True)
SL = np.zeros_like(Ss); SL[1:] = Ss[:-1]
print('async == sync shifted by one refresh:', bool(np.array_equal(SL[:nref], Sa[:nref])), flush=True)
res = {}
for nm, S in (('sync', Ss), ('async_next_refresh', Sa)):
    o = sim(ex, S, R=16, hm=HM, cap_gbps=24); res[nm] = dict(share=round(o['share'], 5), gbps=round(o['gbps'], 2))
off = [json.loads(l) for l in open(f'{W}/results/gbdt_eval.jsonl')]
ref = [r for r in off if r['task'] == task and r['tgt'] == 's5p64' and r['cap'] == 24 and r['hm'] == HM]
print('online sim:', res, ' offline gbdt_eval.jsonl:', [(round(r['share'], 5), round(r['gbps'], 2)) for r in ref], flush=True)
