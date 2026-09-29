"""Per-refresh feature generator for (layer, expert) score models on the decode stream.
All features at refresh t use tokens < t only (plus prefill tokens computed before t)."""
import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
G = 16
def seg_state(tok, req):
    """0 = thinking, 1 = after </think> (answer), per decode token; a new request starts in thinking."""
    st = np.zeros(len(tok), np.int8)
    newreq = np.r_[True, req[1:] != req[:-1]]
    e = tok == SPECIAL['ethink']; b = tok == SPECIAL['think']
    # running state via cumulative last-event
    ev = np.where(newreq | b, 0, np.where(e, 1, -1)).astype(np.int8)
    idx = np.where(ev >= 0, np.arange(len(tok)), 0); idx = np.maximum.accumulate(idx)
    st = ev[idx].copy(); return st, newreq

def prep(d):
    """block-level arrays for a task's decode stream."""
    ex, tok, req = d['ex'], d['tok'], d['req']; N = len(ex)
    st, newreq = seg_state(tok, req)
    C = block_counts(ex, G)
    Ca = block_counts(ex[st == 1], G) if False else None
    # per-state block counts: answer tokens only
    exa = ex.copy(); exa_mask = st == 1
    Can = np.zeros_like(C)
    # count answer tokens per block by masking: route non-answer rows to a dummy by using a weight
    nb = C.shape[0]; lofs = (np.arange(NL, dtype=np.int64) * NE)[None, :, None]
    ia = np.nonzero(exa_mask)[0]
    for a in range(0, len(ia), 1 << 16):
        s = ia[a:a + (1 << 16)]; x = ex[s].astype(np.int64) + lofs; blk = (s // G)[:, None, None]
        u, c = np.unique((blk * (NL * NE) + x).ravel(), return_counts=True)
        Can.reshape(nb, -1).ravel()[u] += c.astype(np.uint8)
    Cth = C - Can
    bst = st[np.minimum(np.arange(nb) * G + G - 1, N - 1)]     # state at end of block
    nans = np.bincount(np.arange(N) // G, weights=exa_mask, minlength=nb)   # answer tokens per block
    # tokens since request start at end of block; request-start flag per block
    rs = np.nonzero(newreq)[0]; last_rs = rs[np.searchsorted(rs, np.arange(N), 'right') - 1]; since = np.arange(N) - last_rs
    bsince = since[np.minimum(np.arange(nb) * G + G - 1, N - 1)]
    # prefill context: prefill rows of the same task computed between decode rows (full stream order)
    fi = d['full_idx']; pre = ~d['dec_all']
    # for each decode block, the prefill rows computed before its first token and after the previous block's first token
    return dict(C=C, Cth=Cth, Can=Can, bst=bst, nans=nans, bsince=bsince, N=N, st=st)

HL = (32, 128, 512, 2048, 8192)
def iter_feats(P, R, prior, hl=HL, shl=(256, 2048)):
    """yields (r, X[NL,NE,F]) at t=r*R. F = len(hl) global EMA rates + 2*len(shl) per-state EMA rates (own-state clock)
    + prior[state_now] + prior[other state] + 1 bias. EMA rates are counts per token (0..1 per slot)."""
    C, Cth, Can, bst, nans = P['C'], P['Cth'], P['Can'], P['bst'], P['nans']
    nb = C.shape[0]; step = R // G; nref = -(-nb // step) + 1
    a = np.array([0.5 ** (G / h) for h in hl], np.float32)[:, None, None]
    E = np.zeros((len(hl), NL, NE), np.float32)
    sa = np.array([0.5 ** (1 / h) for h in shl], np.float32)
    Et = np.zeros((len(shl), NL, NE), np.float32); Ea = np.zeros_like(Et)
    Wt = np.zeros(len(shl), np.float32); Wa = np.zeros(len(shl), np.float32)   # normalizers (sum of weights)
    F = len(hl) + 2 * len(shl) + 3
    for b in range(nb + 1):
        if b % step == 0:
            s = int(bst[b - 1]) if b > 0 else 0
            X = np.empty((NL, NE, F), np.float32)
            X[..., :len(hl)] = np.moveaxis(E * (1 - a) / G, 0, -1)
            cur, oth = (Et, Ea) if s == 0 else (Ea, Et); wc, wo = (Wt, Wa) if s == 0 else (Wa, Wt)
            k = len(hl)
            for j in range(len(shl)):
                X[..., k] = cur[j] / max(wc[j], 1e-6); X[..., k + len(shl)] = oth[j] / max(wo[j], 1e-6); k += 1
            k += len(shl)
            X[..., k] = prior[s]; X[..., k + 1] = prior[1 - s]; X[..., k + 2] = 1.0
            yield b // step, s, X
        if b == nb: break
        E = E * a + C[b]
        na = nans[b]; nt = G - na if b < nb - 1 else G - na   # tokens per state in block (last block may be short)
        dt = sa ** nt; da = sa ** na
        Et = Et * dt[:, None, None] + Cth[b]; Wt = Wt * dt + nt
        Ea = Ea * da[:, None, None] + Can[b]; Wa = Wa * da + na

def fut_target(C, R, F, nref, lead_blocks=1):
    """future counts per token in blocks [r*step+lead_blocks, +F/G)."""
    cs = np.concatenate([np.zeros((1, NL, NE), np.float32), np.cumsum(C, 0, dtype=np.float32)])
    nb = C.shape[0]; step = R // G
    i0 = np.minimum(np.arange(nref) * step + lead_blocks, nb); i1 = np.minimum(i0 + F // G, nb)
    return cs, i0, i1

def priors(tasks):
    """per-state decode usage rate per (l,e) on the given tasks: [2, NL, NE] (counts per token)."""
    tot = np.zeros((2, NL, NE)); n = np.zeros(2)
    for t in tasks:
        d = load(t); P = prep(d)
        tot[0] += P['Cth'].sum(0); tot[1] += P['Can'].sum(0); n[1] += (P['st'] == 1).sum(); n[0] += (P['st'] == 0).sum()
    return (tot / n[:, None, None]).astype(np.float32)
