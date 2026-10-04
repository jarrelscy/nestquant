"""step 3b c5s: live port of the p-salnet c5s admission score (NQ_C5S=<C3k ckpt>; nothing here is imported when unset).

  score = min(exp(lsh(0.6*lsh(log max(jF,1e-4)) + 0.6*lsh(mC))) * sum_e exp(mC), 65000)     (lsh = v - logsumexp_e v, per layer)
  mC    = C3k head 0 (log expected salience of the next 64 rows), clamped to (-30, 11) and rounded through fp16 (= the
          offline score cache /rawdata/Jarrel/nq-step3b-salnet/sc/C3k_<stream>.npy)

Pieces (pure numpy/torch, no vLLM):
  SeqPred / MHA   verbatim copy of nq-step3/p-seq/model.py (the C3k architecture; the ckpt is {'cfg','sd',...})
  Feat            incremental copy of nq-step3/p-seq/feats.build: rows are fed one at a time in stream order; the block
                  [t-16, t) is closed when row t arrives (the request of row t gives the request-relative scalars and pf,
                  exactly build()'s q = req_of_row[t]); same ops, dtypes and order, so the features match build()
  Rows            tfcap row finalizer: per-step decode rows (MTP verify rows incl. rejected drafts) -> one row per
                  (request, position), keeping the last one written, in position order = the offline prep (prep_tfcap /
                  nq-tfpred prep_cap) dedup; pos 0 / prefilling steps dropped; a request change starts a new request
  C5S             Feat + C3k forward + the published latest mC; score(S) blends it with the current jF S

Timing (live): the forward for refresh b (row t = 16b) runs on the c5s drain thread once row t is drained, so the tap
scheduler always blends the LATEST published mC with the current jF S. A late score (forward not done yet, or rows not
drained yet) simply reuses the previous mC; when several refreshes are pending at once only the newest is forwarded (the
features of every block are still accumulated). The score falls back to plain jF (the prod admission) when no mC exists
for the current request yet, or the newest mC is more than NQ_C5S_MAXLAG finalized rows old."""
import contextlib, math, os, time, threading
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

G = 16; NL = 75; NE = 256; NF = 54; LE = NL * NE
HLS = (16, 64, 256, 1024)
OFF = (np.arange(NL) * NE)[:, None]
THINK, ETHINK = 154841, 154842
_null = contextlib.nullcontext
WJ, WM = 0.6, 0.6            # c5s arm: 0.6 * lsh(log jF) + 0.6 * lsh(mC)


# ------------------------------------------------------------------ C3k architecture (verbatim nq-step3/p-seq/model.py)
class MHA(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)
    def forward(s, x):
        B, T, d = x.shape; h = s.nh
        q, k, v = s.qkv(x).view(B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)
        return s.o(Fn.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, d))


class SeqPred(nn.Module):
    def __init__(s, d=64, feats=None, use_hp=True, use_ctx=True, expert_attn=True, layer_attn=True, nh=4, use_eemb=True):
        super().__init__()
        feats = feats if feats is not None else list(range(NF))
        s.cfg = dict(d=d, feats=list(feats), use_hp=use_hp, use_ctx=use_ctx, expert_attn=expert_attn, layer_attn=layer_attn, nh=nh, use_eemb=use_eemb)
        s.register_buffer('fidx', torch.tensor(feats, dtype=torch.long))
        s.enc = nn.Sequential(nn.Linear(len(feats), 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.eemb = nn.Parameter(torch.zeros(NL, NE, d), requires_grad=use_eemb); s.lemb = nn.Parameter(torch.randn(NL, 1, d) * 0.02)
        s.use_hp = use_hp; s.use_ctx = use_ctx
        if use_ctx: s.ctx = nn.Linear(2, d)
        if use_hp:
            s.hp = nn.Sequential(nn.Linear(2 * 8 * 256, 256), nn.GELU())
            s.hpl = nn.Linear(256, NL * d)
            s.hpk = nn.Linear(256, NL * 16); s.ek = nn.Parameter(torch.randn(NL, NE, 16) * 0.02)
            nn.init.zeros_(s.hpl.weight); nn.init.zeros_(s.hpl.bias)
        s.ea = expert_attn; s.la = layer_attn
        if expert_attn: s.n1 = nn.LayerNorm(d); s.sa = MHA(d, nh)
        if layer_attn: s.n2 = nn.LayerNorm(d); s.xa = MHA(d, nh)
        s.n3 = nn.LayerNorm(d); s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.nf = nn.LayerNorm(d); s.head = nn.Linear(d, 3)
        nn.init.zeros_(s.head.weight); nn.init.constant_(s.head.bias, -2.0)

    def forward(s, x, sc, hp):
        """x [B, NL*NE, NF] , sc [B,4], hp [B,2,8,256]. returns [B, NL, NE, 3] (log sal64, log cnt64, log nu)."""
        B = x.shape[0]; d = s.cfg['d']
        h = s.enc(x[..., s.fidx].view(B, NL, NE, -1)) + s.eemb[None] + s.lemb[None]
        if s.use_ctx: h = h + s.ctx(sc[:, :2])[:, None, None]
        extra = 0
        if s.use_hp:
            z = s.hp(hp.reshape(B, -1))
            h = h + s.hpl(z).view(B, NL, 1, d)
            extra = (s.hpk(z).view(B, NL, 1, 16) * s.ek[None]).sum(-1)
        if s.ea:
            u = s.n1(h).view(B * NL, NE, d); h = h + s.sa(u).view(B, NL, NE, d)
        if s.la:
            m = s.n2(h.mean(2)); h = h + s.xa(m)[:, :, None]
        h = h + s.ff(s.n3(h))
        o = s.head(s.nf(h))
        if s.use_hp: o = o + torch.stack([extra, extra, torch.zeros_like(extra)], -1)
        return o


def load_net(path, dev='cpu'):
    c = torch.load(path, map_location='cpu', weights_only=False)
    net = SeqPred(**c['cfg']); net.load_state_dict(c['sd']); net.eval()
    return net.to(dev)


# ------------------------------------------------------------------ blend (= p-salnet/export.py, arm c5s)
def lsh(v):
    m = v.max(-1, keepdims=True)
    return v - np.log(np.exp(v - m).sum(-1, keepdims=True)) - m


def blend(S, mC):
    """S jF [75,256] (count units), mC [75,256] float32 (clamped, fp16-rounded log sal64) -> c5s score float32 [75,256]"""
    l = WJ * lsh(np.log(np.maximum(np.asarray(S, np.float32), 1e-4))) + WM * lsh(mC)
    m = np.exp(mC).sum(-1, keepdims=True)
    return np.minimum(np.exp(lsh(l)) * m, 65000).astype(np.float16).astype(np.float32)   # export stores f16


# ------------------------------------------------------------------ incremental feats.build
class Feat:
    """feats.build(ids, sal, hp, think, rstart, pf) fed row by row. add() returns True when the row closed a block and
    a refresh (b = n/16, row t = 16b, rows < t) is ready (s.pend); inputs() then gives X fp16 [LE,NF], S f32 [4], HP fp16
    [2,8,256] (X is built lazily, so a superseded refresh costs only the block update). Think = olib.phase_of per request
    (the tb traces mark the </think> row itself as think; one row per request differs). S[2:4] (valid64/128) are unused
    by the model and set to 1."""
    def __init__(s):
        s.dec = {h: 0.5 ** (1 / h) for h in HLS}
        s.w16 = {h: s.dec[h] ** np.arange(15, -1, -1) for h in HLS}
        s.Es = {h: np.zeros(LE, np.float32) for h in HLS}; s.Ec = {h: np.zeros(LE, np.float32) for h in HLS}
        s.ring_s = np.zeros((64, LE), np.float32); s.ring_c = np.zeros((64, LE), np.float32)
        s.last = np.full(LE, -10 ** 9, np.int64)
        s.bi = np.zeros((G, NL, 8), np.int64); s.bs = np.zeros((G, NL, 8), np.float32)
        s.bh = np.zeros((G, 8, 256), np.float16); s.bt = np.zeros(G, np.float64); s.has_hp = True
        s.n = 0                      # rows seen
        s.rs = 0                     # start row of the current request
        s.starts = []                # request starts > 0 not yet consumed by a block close
        s.pf = None                  # current request prefill fraction [75,256] float32 (None -> feature 53 = 0)
        s.seg = 0                    # request counter
        s.ph = 0                     # phase_of segment state (0 = think)
        s.pend = None

    def new_request(s, pf=None):
        """the next row starts a new request; pf = prefill expert counts / prefill tokens [75,256] (or None)"""
        if s.n > 0: s.starts.append(s.n)
        s.rs = s.n; s.pf = None if pf is None else np.asarray(pf, np.float32).reshape(NL, NE); s.seg += 1; s.ph = 0

    def _close(s, t):
        r0 = t - G
        ix = (s.bi + OFF[None]).reshape(G, -1)
        sv = s.bs.reshape(G, -1)
        Wb = np.zeros((G, LE), np.float32); Cb = np.zeros((G, LE), np.float32)
        np.put_along_axis(Wb, ix, sv, 1); np.put_along_axis(Cb, ix, 1.0, 1)
        for h in HLS:
            s.Es[h] = s.Es[h] * s.dec[h] ** G + s.w16[h] @ Wb; s.Ec[h] = s.Ec[h] * s.dec[h] ** G + s.w16[h] @ Cb
        s.ring_s = np.concatenate([s.ring_s[G:], Wb]); s.ring_c = np.concatenate([s.ring_c[G:], Cb])
        for k in range(G):
            s.last[ix[k]] = r0 + k
        # build(): reset if a request starts in (r0, t) or at t (row t's own request start)
        if any(r0 < r <= t for r in s.starts):
            for h in HLS: s.Es[h][:] = 0; s.Ec[h][:] = 0
            s.ring_s[:] = 0; s.ring_c[:] = 0; s.last[:] = -10 ** 9
        s.starts = [r for r in s.starts if r > t]

    def add(s, ids, sal, hp, tok=None, think=None):
        """one finalized row: ids [75,8] int, sal [75,8] float32, hp [8,256] fp16|None, tok = input token (phase_of think)
        or think given explicitly (bool). Returns True if this row made a refresh ready (s.pend)."""
        t = s.n; ready = False
        if t > 0 and t % G == 0:
            s._close(t)
            th = float(s.bt.mean())
            hp_ = s.bh.astype(np.float32)
            s.pend = (t, np.array([th, np.log1p(t - s.rs) / 8, 1.0, 1.0], np.float32),
                      np.stack([hp_.mean(0), hp_[-1]]).astype(np.float16) if s.has_hp else np.zeros((2, 8, 256), np.float16),
                      None if s.pf is None else s.pf.reshape(-1) * 16, s.seg)
            ready = True
        if think is None:
            if tok == THINK: s.ph = 0
            elif tok == ETHINK: s.ph = 1
            think = s.ph == 0
        k = t % G
        s.bi[k] = ids; s.bs[k] = sal; s.bt[k] = bool(think)
        if hp is None: s.has_hp = False
        else: s.bh[k] = hp
        s.n = t + 1
        return ready

    def inputs(s):
        """X fp16 [LE,NF], S f32 [4], HP fp16 [2,8,256], t, seg of the newest ready refresh (state = right after its block close;
        valid until the next block closes)"""
        t, S, HP, pf, seg = s.pend
        x = np.empty((LE, NF), np.float32)
        x[:, 0:16] = np.log1p(20 * s.ring_s[48:]).T
        x[:, 16:28] = np.log1p(20 * s.ring_s[:48].reshape(12, 4, LE).sum(1)).T
        x[:, 28:44] = s.ring_c.reshape(16, 4, LE).sum(1).T / 4
        for j, h in enumerate(HLS):
            x[:, 44 + j] = np.log1p(200 * s.Es[h] * (1 - s.dec[h])); x[:, 48 + j] = s.Ec[h] * (1 - s.dec[h]) * 4
        x[:, 52] = np.where(s.last < 0, 1.2, np.log1p(np.maximum(t - s.last, 0)) / 7)
        x[:, 53] = 0 if pf is None else pf
        return x.astype(np.float16), S, HP, t, seg


# ------------------------------------------------------------------ tfcap row finalizer
class Rows:
    """steps (request q, its rows' pos/tok/ids/w/xn/hp, prefilling flag) -> finalized rows in (request, position) order.
    Within a request positions only advance (an MTP step starts after the last accepted token), so every pending row
    with pos < the new step's first pos is final; the step's own rows (pos >= first) stay pending and are superseded by
    the next step's rows at the same positions (= offline keep-last-per-position)."""
    def __init__(s, emit, new_request):
        s.emit = emit; s.newreq = new_request
        s.q = None; s.pend = {}; s.pfc = {}; s.pfn = {}

    def prefill(s, q, T, counts):
        s.pfc[q] = counts.astype(np.int64) if q not in s.pfc else s.pfc[q] + counts; s.pfn[q] = s.pfn.get(q, 0) + T

    def _flush(s, below=None):
        for p in sorted(s.pend):
            if below is not None and p >= below: break
            s.emit(*s.pend.pop(p))

    def step(s, q, prefilling, pos, tok, ids, w, xn, hp):
        if prefilling: return
        if q != s.q:
            s._flush(); s.q = q
            pf = None
            if q in s.pfc: pf = (s.pfc[q] / max(s.pfn[q], 1)).astype(np.float32)
            s.newreq(pf)
        ok = np.nonzero(pos > 0)[0]
        if not len(ok): return
        s._flush(int(pos[ok].min()))
        for j in ok:
            sal = w[j].astype(np.float32) ** 2 * xn[j].astype(np.float32)[:, None]
            s.pend[int(pos[j])] = (ids[j], sal, None if hp is None else hp[j], int(tok[j]))


# ------------------------------------------------------------------ live predictor
class C5S:
    def __init__(s, ckpt, dev='cpu', threads=0, maxlag=64):
        s.dev = torch.device(dev); s.net = load_net(ckpt, s.dev); s.threads = int(threads); s.maxlag = int(maxlag)
        s.F = Feat(); s.R = Rows(s._emit, s.F.new_request)
        s.cur = None                 # (mC float32 [75,256], t, seg, version)
        s.ver = 0; s.ready = False
        s.st = dict(rows=0, refresh=0, fwd=0, skipped=0, t_feat=0.0, t_fwd=0.0, t_blk=0.0, fallback=0, used=0)
        s._tset = False

    def _emit(s, ids, sal, hp, tok):
        t0 = time.perf_counter()
        if s.F.add(ids, sal, hp, tok=tok):
            if s.ready: s.st['skipped'] += 1
            s.ready = True; s.st['refresh'] += 1
        s.st['rows'] += 1; s.st['t_blk'] += time.perf_counter() - t0

    def forward(s, x, S, HP):
        if s.threads > 0 and not s._tset and s.dev.type == 'cpu':
            torch.set_num_threads(s.threads); s._tset = True
        st = getattr(s, 'stream', None) if s.dev.type == 'cuda' else None
        with torch.no_grad(), (torch.cuda.stream(st) if st is not None else _null()):
            o = s.net(torch.from_numpy(x.astype(np.float32))[None].to(s.dev), torch.from_numpy(S)[None].to(s.dev),
                      torch.from_numpy(HP.astype(np.float32))[None].to(s.dev))[0, ..., 0]
            return o.clamp(-30, 11).to(torch.float16).float().cpu().numpy()

    def run_pending(s):
        """forward the newest ready refresh (if any) and publish it"""
        if not s.ready: return False
        s.ready = False
        t0 = time.perf_counter(); x, S, HP, t, seg = s.F.inputs(); t1 = time.perf_counter()
        mC = s.forward(x, S, HP); t2 = time.perf_counter()
        s.ver += 1; s.cur = (mC, t, seg, s.ver)
        s.st['fwd'] += 1; s.st['t_feat'] += t1 - t0; s.st['t_fwd'] += t2 - t1
        return True

    def latest(s):
        """newest usable mC (float32 [75,256]) and its version, or (None, None): none yet for this request, or too old"""
        c = s.cur
        if c is None or c[2] != s.F.seg or s.F.n - c[1] > s.maxlag: return None, None
        return c[0], c[3]

    def score(s, S):
        """c5s admission for the current jF S, or None (-> caller uses jF)"""
        mC, v = s.latest()
        if mC is None: s.st['fallback'] += 1; return None
        s.st['used'] += 1; return blend(S, mC)
