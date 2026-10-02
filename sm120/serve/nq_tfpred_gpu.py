"""nq-tfpred serve path: the multi-window transformer expert-use predictor (threads/36-tfpred/model.py) as a drop-in for
GPUJointPredictor (NQ_PREDICTOR=tf, NQ_TF_CKPT=<ckpt.pt>).  Off unless selected; nq_vllm.py never imports it otherwise.

  p = TFGPUPredictor(layers, fixed, ckpt, n_float=77, hm=0.7, device='cuda', graph=True)
  p.step(counts[NL,NE], ntok, token_ids=None, new_request=False, sal=None) -> bool   (True at a 16-token refresh)
  p.target(resident) -> want bool [NL,NE] ; p.order_score(resident) ; p.S (np f32 [NL,NE])
  p.value(a, h) -> np f32 [NL,NE]: expected hits in [a, a+h) tokens after the last refresh (piecewise-uniform windows)

Inputs are exactly the trainer's (threads/36-tfpred/data.py Blocks.inputs): raw per-block hits for the last F=32 blocks,
raw sums of the last C=32 completed 8-block chunks that end before the fine window (chunks aligned to the block counter),
the answer fraction of the last F blocks (needs token_ids; 0 when the caller passes none), blocks since the request's
first decode block, and the request's prefill hit fraction per token.  Request boundaries: new_request=True, or (serve,
where the host loop only sees polled counts) a step with ntok > max_decode_ntok after decode steps = a new prefill.
S = value(NQ_TF_LAT, NQ_TF_H): predicted hits after a read issued now lands (default 64 tokens) over the next 256 tokens,
which the Scheduler's top-n_float target uses.  The scoring core (gather + transformer + window integration) is one
CUDA graph; per refresh the host copies 4 index vectors and reads back S (77 KB)."""
import os, sys
import numpy as np, torch

import importlib.util as _ilu                                   # load by path: a bare 'import model' may collide in the serve
_sp = _ilu.spec_from_file_location('nq_tf_model', os.path.join(os.environ.get('NQ_HOME', '/nq'), 'threads/36-tfpred/model.py'))
TM = _ilu.module_from_spec(_sp); _sp.loader.exec_module(TM)

G, F, C, CK = 16, 32, 32, 8
RING_F, RING_C = 64, 64
THINK_ID, ETHINK_ID = 154841, 154842


class TFGPUPredictor:
    def __init__(self, layers, fixed, ckpt, n_float=77, hm=0.7, ha=0.0, device='cuda', graph=True, max_decode_ntok=16,
                 lat=None, horizon=None):
        self.layers = list(layers); NL = self.NL = len(self.layers); NE = self.NE = 256
        assert NL == TM.NL, (NL, TM.NL)
        self.nf, self.hm, self.ha, self.dev, self.max_ntok = n_float, hm, ha, torch.device(device), max_decode_ntok
        self.fixed = np.zeros((NL, NE), bool)
        for i, L in enumerate(self.layers):
            self.fixed[i, list(fixed[L])] = True
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.win = list(ck['win']); self.use_sal = bool(ck['args'].get('sal')); self.noans = bool(ck['args'].get('noans'))
        self.net = TM.TFPred(**ck['cfg'], scale=ck.get('scale')).to(self.dev).eval(); self.net.load_state_dict(ck['state'])
        self.content = bool(ck['cfg']['content'])
        assert not self.content, 'content checkpoints need token/hidden-state plumbing (offline only)'
        self.params = sum(p.numel() for p in self.net.parameters())
        dv = self.dev; z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dv)   # noqa: E731
        self.fr = z(RING_F, NL, NE); self.cr = z(RING_C, NL, NE); self.ar = z(RING_F)
        self.cur = z(NL, NE); self.chk = z(NL, NE); self.pf = z(NL, NE); self.pf_acc = z(NL, NE)
        self.fidx = torch.zeros(F, dtype=torch.long, device=dv); self.fm = z(F)
        self.cidx = torch.zeros(C, dtype=torch.long, device=dv); self.cm = z(C)
        self.rpos_t = z(1); self.pf_n = 0
        self.wlen = torch.tensor(np.diff(self.win), dtype=torch.float32, device=dv)
        self.lat = float(os.environ.get('NQ_TF_LAT', '64') if lat is None else lat)
        self.H = float(os.environ.get('NQ_TF_H', '256') if horizon is None else horizon)
        self.wA = torch.from_numpy(self._wint(self.lat, self.H)).to(dv)
        self.h_f = torch.zeros(2 * F + 2 * C + 1, dtype=torch.float32).pin_memory()
        self.d_f = torch.zeros_like(self.h_f, device=dv)
        self.nblk = 0; self.btok = 0; self.bans = 0; self.seg = 0; self.req_blk0 = 0; self.in_prefill = False
        self.S = None; self.mu = None; self.stats = {}
        self.graph = None
        if graph and self.dev.type == 'cuda':
            st = torch.cuda.Stream(dv); st.wait_stream(torch.cuda.current_stream(dv))
            with torch.cuda.stream(st):
                for _ in range(3):
                    self._core()
            torch.cuda.current_stream(dv).wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.g_S, self.g_mu = self._core()

    def _wint(self, a, h):
        """[W] weights: overlap of [a, a+h) with each window / window length (the last window's rate extends beyond)."""
        e = np.array(self.win, float); w = np.zeros(len(e) - 1, np.float32)
        for i in range(len(e) - 1):
            w[i] = max(0.0, min(a + h, e[i + 1]) - max(a, e[i])) / (e[i + 1] - e[i])
        if a + h > e[-1]:
            w[-1] += (a + h - max(a, e[-1])) / (e[-1] - e[-2])
        return w

    def _t(self, a):
        return a.to(self.dev, torch.float32) if torch.is_tensor(a) else torch.as_tensor(np.asarray(a, np.float32), device=self.dev)

    # ------------------------------------------------------------------ host bookkeeping
    def step(self, counts, ntok=1, token_ids=None, new_request=False, sal=None):
        c = self._t(sal if self.use_sal else counts)
        if ntok > self.max_ntok:                      # prefill (or a polled step mixing one): request prefill counts
            if not self.in_prefill:
                self.pf_acc.zero_(); self.pf_n = 0; self.in_prefill = True
            self.pf_acc += self._t(counts); self.pf_n += ntok
            return False
        if new_request or self.in_prefill:
            if self.in_prefill:
                self.pf.copy_((self.pf_acc / max(self.pf_n, 1)).half())   # match trainer fp16 pf rounding (data.blocks_from_stream)
            elif new_request:
                self.pf.zero_()
            self.in_prefill = False; self.req_blk0 = self.nblk; self.seg = 0
        if token_ids is not None:
            for t in token_ids:
                if t == THINK_ID: self.seg = 0
                elif t == ETHINK_ID: self.seg = 1
            if self.seg == 1 and not self.noans: self.bans += ntok
        self.cur += c; self.btok += ntok
        if self.btok < G:
            return False
        self._close_block(); self.S = self._score(); return True

    def _close_block(self):
        b = self.nblk
        self.fr[b % RING_F].copy_(self.cur); self.ar[b % RING_F] = min(self.bans, G) / G
        self.chk += self.cur
        if b % CK == CK - 1:
            self.cr[(b // CK) % RING_C].copy_(self.chk); self.chk.zero_()
        self.cur.zero_(); self.btok -= G; self.bans = 0; self.nblk += 1
        fi = b - (F - 1) + np.arange(F)                                  # data.Blocks.inputs, stream start = block 0
        c_end = (b - F + 1) // CK; ci = c_end - C + np.arange(C)
        h = self.h_f.numpy()
        h[:F] = fi % RING_F; h[F:2 * F] = fi >= 0
        h[2 * F:2 * F + C] = ci % RING_C; h[2 * F + C:2 * F + 2 * C] = (ci >= 0)
        h[-1] = b - self.req_blk0
        self.d_f.copy_(self.h_f, non_blocking=True)

    @torch.no_grad()
    def _core(self):
        d = self.d_f
        fidx = d[:F].long(); fm = d[F:2 * F]; cidx = d[2 * F:2 * F + C].long(); cm = d[2 * F + C:2 * F + 2 * C]
        fine = (self.fr[fidx] * fm[:, None, None])[None]
        coarse = (self.cr[cidx] * cm[:, None, None])[None]
        ans = (self.ar[fidx] * fm)[None]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
            lr = self.net(fine=fine, coarse=coarse, pf=self.pf[None], ans=ans, rpos=d[-1:].clamp(min=0))
        mu = torch.exp(lr.float()[0].clamp(max=20)) * self.wlen                 # [NL,NE,W] hits per window
        return (mu * self.wA).sum(-1), mu

    @torch.no_grad()
    def _score(self):
        if self.graph is not None:
            self.graph.replay(); self.mu = self.g_mu; return self.g_S.cpu().numpy()
        S, self.mu = self._core(); return S.cpu().numpy()

    def value(self, a, h):
        if self.mu is None:
            return None
        return (self.mu * torch.from_numpy(self._wint(a, h)).to(self.dev)).sum(-1).cpu().numpy()

    def _adj(self, resident):
        v = np.where(self.fixed, -np.inf, self.S).astype(np.float32)
        r = np.asarray(resident, bool) & ~self.fixed
        return np.where(r, v * np.float32(1 + self.hm) + np.float32(self.ha), v), r

    def target(self, resident):
        if self.S is None:
            return None
        v, r = self._adj(resident)
        top = np.argsort(-v, 1, kind='stable')[:, :self.nf]
        want = np.zeros((self.NL, self.NE), bool); np.put_along_axis(want, top, True, 1)
        return want

    def order_score(self, resident):
        return self._adj(resident)[0] if self.S is not None else None

    def close(self):
        pass
