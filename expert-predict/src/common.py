import ctypes, json, sys, os, time
import numpy as np
sys.path.insert(0, '/data/Jarrel/nestquant/streaming'); import fixed_set   # read-only import
W = '/data/Jarrel/expert-predict'; DATA = W + '/data'
NE, NL, K, L0, NF = 256, 75, 8, 3, 51
MB = 9.97e6            # bytes per upgrade, all 4 ranks (2.56 MB per rank); scheduler rec_bytes convention
TPS = 95.0; CAP_GBPS = 6.0; BURST = 64
DEC_TASKS = ['formal-crypto', 'embedding-drift-monitor', 'fin-saccr-rwa', 'pretrain-shard-corruption',
             'freight-dispatch-shift', 'sound-change-cascade']
PRE_TASKS = ['ks-solver-cpp', 'lake-temp-glm', 'layout-config-recreation2', 'photonic-waveguide-routing',
             'react-lead-form', 'satb-audio-transcription', 'takens-embedding-lean']
VAL = ['formal-crypto', 'embedding-drift-monitor']
TEST = ['fin-saccr-rwa', 'pretrain-shard-corruption', 'freight-dispatch-shift', 'sound-change-cascade']
SPECIAL = dict(think=154841, ethink=154842, tc=154843, etc=154844, tr=154845, etr=154846, eot=154820,
               user=154827, asst=154828, obs=154829)

_fx, _, _ = fixed_set.load()
FIXED = np.zeros((NL, NE), np.uint8)
for l in range(NL): FIXED[l, _fx[l + L0]] = 1
_fj = json.load(open('/data/Jarrel/nestquant/threads/22-boundary-experts/fixed_set.json'))
INIT = np.zeros((NL, NE), np.uint8)
for l in range(NL):
    nr = np.array(_fj['n_routed'][str(l + L0)], float)
    INIT[l, np.argsort(-np.where(FIXED[l], -1, nr), kind='stable')[:NF]] = 1

# slot weights for the salience proxies (static, per (layer, expert), from the calibration capture / encode manifests)
REAP = np.array([_fj['reap_mean'][str(l + L0)] for l in range(NL)], np.float64)       # mean p*||y|| per routed row
def _gain():
    g = np.zeros((NL, NE))
    for l in range(NL):
        m = json.load(open(f'/rawdata/Jarrel/nq-glm53-hf/layers/L{l + L0}/manifest.json'))['per_expert']
        for e in range(NE):
            pj = m[str(e)]['proj']; d = {k: pj[k]['proxy_rot']['2'] - pj[k]['proxy_rot']['4'] for k in pj}
            g[l, e] = (d['gate'] + d['up']) / 2 + d['down']
    return g
GAINREL = _gain()                     # relative-error drop 2->4 bit (encode proxy_rot; gate/up mean + down)
GAIN = REAP * GAINREL                 # ~ mean p*||y4 - y2|| proxy per routed row
WTS_NAMES = ['sal_share', 'gain_share']
WTS = np.ascontiguousarray(np.stack([REAP.ravel(), GAIN.ravel()]).astype(np.float32))
_lib = ctypes.CDLL(W + '/src/libsim.so')
_lib.run.restype = ctypes.c_longlong
P = ctypes.c_void_p
_lib.run.argtypes = [P, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, P, P, P, P, ctypes.c_double, ctypes.c_double, ctypes.c_int, P, P]

def load(task, decode_only=True):
    z = np.load(f'{DATA}/{task}.npz')
    ex, tok, dec, req = z['ex'], z['tok'], z['dec'], z['req']
    if decode_only:
        m = dec
        return dict(ex=np.ascontiguousarray(ex[m]), tok=tok[m], req=req[m], full_idx=np.nonzero(m)[0],
                    ex_all=ex, tok_all=tok, dec_all=dec, req_all=req)
    return dict(ex=ex, tok=tok, req=req, full_idx=np.arange(len(tok)), ex_all=ex, tok_all=tok, dec_all=dec, req_all=req)

def block_counts(ex, G=16, chunk=1 << 16):
    """uint8 [nb, NL, NE] routed counts per G-token block (G<=31)."""
    N = len(ex); nb = -(-N // G); out = np.zeros((nb, NL * NE), np.uint8)
    lofs = (np.arange(NL, dtype=np.int64) * NE)[None, :, None]
    for a in range(0, N, chunk):
        b = min(N, a + chunk); x = ex[a:b].astype(np.int64) + lofs
        blk = (np.arange(a, b) // G - a // G)[:, None, None]
        c = np.bincount((blk * (NL * NE) + x).ravel(), minlength=(-(-(b - a) // G) + 1) * NL * NE)
        nbb = (b - 1) // G - a // G + 1
        out[a // G:a // G + nbb] += c[:nbb * NL * NE].reshape(nbb, NL * NE).astype(np.uint8)
    return out.reshape(nb, NL, NE)

def window_counts(ex, R, lead, nref):
    """float32 [nref, NL, NE]: exact counts in [rR+lead, rR+lead+R) (oracle target)."""
    N = len(ex); S = np.zeros((nref, NL * NE), np.float32); lofs = (np.arange(NL) * NE)[None, :, None]
    for r in range(nref):
        a = r * R + lead; b = min(N, a + R)
        if a < N: S[r] = np.bincount((ex[a:b].astype(np.int64) + lofs).ravel(), minlength=NL * NE)
    return S.reshape(nref, NL, NE)

def sim(ex, S, R=64, lead=13, cap_gbps=CAP_GBPS, lazy=0, tps=TPS, nf=NF, fixed=FIXED, init=INIT, burst=BURST, hm=0.0, ha=0.0, wts=None):
    N = len(ex); ex = np.ascontiguousarray(ex, np.uint8); S = np.ascontiguousarray(S, np.float32)
    assert S.shape[0] >= -(-N // R), (S.shape, N, R)
    hits = np.zeros(N, np.int16); ups = np.zeros(N, np.int32); out = np.zeros(3); wout = np.zeros(2 * len(WTS))
    per = cap_gbps * 1e9 / tps
    fixed = np.ascontiguousarray(fixed, np.uint8); init = np.ascontiguousarray(init, np.uint8)
    t0 = time.time()
    _lib.run(ex.ctypes.data, N, NL, K, fixed.ctypes.data, S.ctypes.data, R, nf, lead, per, per * burst, MB, lazy,
             init.ctypes.data, hits.ctypes.data, ups.ctypes.data, out.ctypes.data, hm, ha, len(WTS), WTS.ctypes.data, wout.ctypes.data)
    r = dict(share=float(hits.sum()) / (N * NL * K), ups_tok=out[0] / N, gbps=out[0] / N * MB * tps / 1e9,
                stall=out[2], N=N, hits=hits, secs=time.time() - t0)
    for j, k in enumerate(WTS_NAMES): r[k] = wout[2 * j + 1] / wout[2 * j]
    return r

def ema_scores(C, R, hl, G=16):
    """EMA(half-life hl tokens) of block counts, sampled at refresh points t=rR (history < t). C: [nb,NL,NE] G-blocks."""
    nb = C.shape[0]; step = R // G; nref = -(-nb // step) + 1
    a = 0.5 ** (G / hl); s = np.zeros((NL, NE), np.float32); S = np.zeros((nref, NL, NE), np.float32); norm = (1 - a) / G
    for b in range(nb):
        if b % step == 0: S[b // step] = s
        s = s * a + C[b]
    if nb % step == 0: S[nb // step] = s
    return S * np.float32(norm)
