"""GBDT floating-set predictor (expert-predict study, 2026-09-29). Pure policy helper for scheduler.Scheduler; does not
touch the budget, big-step guard, executor feedback or kv_pressure logic.

What it predicts: for each MoE layer, which 51 non-fixed experts to hold at level 4 for the next ~64 decode tokens.
Model: LightGBM Poisson on next-64 routed hits (60 trees x 15 leaves), trained on fin-saccr-rwa + pretrain-shard-
corruption decode streams, early-stopped on formal-crypto + embedding-drift-monitor (held-out results in
/data/Jarrel/expert-predict/PROGRESS.md). Five features per (layer, expert), all from the per-step routing counts:
  ema32, ema128        decayed routed-hit rates (half-life 32 / 128 tokens, per token)
  mem_cur_state        own-clock memory (half-life 2048 tokens of that state) of the current segment type:
                       think (after <think> / request start) or answer (after </think>), from the emitted token ids
  tok_since_hit        tokens since the expert was last routed (16-token resolution, capped at 1e5)
  hits16               routed hits in the last 16-token block
Only the EMA256 ranks 20..120 of the non-fixed experts are scored by the model (7575 rows per refresh); ranks <20 are
always kept; ranks >120 get score 0.

Cadence: features are updated per 16-token block; a refresh happens at every block boundary (every 16 decode tokens).
Scoring is exactly the offline pipeline (/data/Jarrel/expert-predict/src/gbdt_feats.py + gbdt_eval.py); parity test:
/data/Jarrel/expert-predict/src/parity_gbdt.py.

  p = GBDTPredictor(layers, fixed, model_path=None, n_float=51, hm=0.5, mode='next_refresh', num_threads=4)
  p.step(counts[len(layers), NE], ntok=1, token_ids=None, new_request=False) -> bool   once per DECODE step, same
        counts the Scheduler gets. Returns True when a new score matrix was applied (i.e. a refresh is due).
        Steps with ntok > 16 (prefill chunks) are ignored, matching the offline decode-only stream.
  p.target(resident[len(layers), NE] bool) -> want[len(layers), NE] bool or None
        resident = floating experts in flight or at level 4 (Scheduler.state in (1, 2)). Hysteresis: a resident
        expert's score is multiplied by (1 + hm). Layers with no information keep their resident set. None before the
        first score matrix exists (keep floating_default).
  p.order_score(resident) -> float32 [len(layers), NE]   hysteresis-adjusted score; issue wanted upgrades in
        descending order of this (the offline sim does), instead of Scheduler.score.
  p.close()   stop the worker thread.

mode='sync': predict inline at the block boundary (~2 ms on 4 threads + ~1-2 ms numpy features; offline numbers).
mode='next_refresh': features are snapshotted at the block boundary and predicted on a worker thread (LightGBM
releases the GIL); the result is applied at the NEXT block boundary (one refresh = 16 tokens late). The held-out cost
of the late application is reported in PROGRESS.md.

Wiring sketch (NQ serving agent): keep Scheduler as-is, set Scheduler(refresh=16); after each s.step(...) call
p.step(counts, ntok, token_ids, new_request); when the Scheduler refreshes, replace s.want[has] with
p.target(np.isin(s.state, (1, 2))) where available, and order candidates by p.order_score(...).
Assumes a single decode stream (c1); with several concurrent requests the think/answer state is per request and
the counts are mixed, so pass token_ids=None (state stays 'think') or feed the dominant request's tokens."""
import os, threading, queue
import numpy as np

THINK_ID, ETHINK_ID = 154841, 154842
G = 16
FEATS = ('ema32', 'ema128', 'mem_cur_state', 'tok_since_hit', 'hits16')
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gbdt_p64_s5.txt')

class GBDTPredictor:
    def __init__(self, layers, fixed, model_path=None, NE=256, n_float=51, hm=0.5, ha=0.0, rlo=20, rhi=121,
                 mode='next_refresh', num_threads=4, think_id=THINK_ID, ethink_id=ETHINK_ID, max_decode_ntok=16):
        import lightgbm as lgb
        self.layers = list(layers); self.NL = len(self.layers); self.NE = NE; self.nf = n_float; self.hm = hm; self.ha = ha
        self.rlo, self.rhi = rlo, rhi; self.mode = mode; self.nthr = num_threads
        self.think_id, self.ethink_id = think_id, ethink_id; self.max_ntok = max_decode_ntok
        self.fixed = np.zeros((self.NL, NE), bool)
        for i, L in enumerate(self.layers): self.fixed[i, list(fixed[L])] = True
        self.bst = lgb.Booster(model_file=model_path or DEFAULT_MODEL)
        names = self.bst.feature_name(); assert tuple(names) == FEATS, names
        NL = self.NL
        # --- state (identical arithmetic to gbdt_feats.gen) ---
        self.ag = [np.float32(0.5 ** (G / h)) for h in (32, 128, 256)]
        self.E = [np.zeros((NL, NE), np.float32) for _ in self.ag]
        self.sa = 0.5 ** (1 / 2048); self.Et = np.zeros((NL, NE), np.float32); self.Ea = np.zeros_like(self.Et); self.wt = self.wa = 0.0
        self.last = np.full((NL, NE), -10 ** 6, np.int64)
        self.nblk = 0                                     # completed blocks
        self.bc = np.zeros((NL, NE), np.float32); self.bca = np.zeros((NL, NE), np.float32); self.btok = 0; self.bans = 0
        self.seg = 0                                      # 0 think, 1 answer (per token, from emitted ids)
        self.S = None                                     # applied score matrix
        self.li = np.arange(NL)[:, None]
        self._q = self._res = self._th = None
        if mode == 'next_refresh':
            self._q = queue.Queue(maxsize=1); self._res = queue.Queue(maxsize=1)
            self._th = threading.Thread(target=self._worker, daemon=True); self._th.start()
        self.stats = dict(refreshes=0, late_waits=0)

    # ---------------------------------------------------------------- per step
    def step(self, counts, ntok=1, token_ids=None, new_request=False):
        if ntok > self.max_ntok: return False            # prefill chunk: not part of the decode stream
        c = np.asarray(counts, np.float32)
        if new_request: self.seg = 0
        if token_ids is not None:
            for t in token_ids:
                if t == self.think_id: self.seg = 0
                elif t == self.ethink_id: self.seg = 1
        # a step's counts go to the segment of its last token (exact when ntok == 1)
        self.bc += c
        if self.seg == 1: self.bca += c; self.bans += ntok
        self.btok += ntok
        if self.btok < G: return False
        self._close_block(); return self._refresh()

    def _close_block(self):
        c = self.bc; Can = self.bca; Cth = c - Can; na = np.float64(self.bans); j = self.nblk   # float64 like the offline nans
        for k in range(len(self.ag)): self.E[k] = self.E[k] * self.ag[k] + c
        nt = G - na; dt = np.float32(self.sa ** nt); da = np.float32(self.sa ** na)
        self.Et = self.Et * dt + Cth; self.wt = self.wt * dt + nt; self.Ea = self.Ea * da + Can; self.wa = self.wa * da + na
        self.last = np.where(c > 0, j, self.last); self.h16 = c
        self.bst_state = self.seg; self.nblk += 1
        self.btok -= G; self.bc = np.zeros_like(c); self.bca = np.zeros_like(c); self.bans = 0

    def _features(self):
        b = self.nblk; ag = self.ag
        e256 = self.E[2] * ((1 - ag[2]) / G)
        sc = np.where(self.fixed, -np.inf, e256); order = np.argsort(-sc, 1, kind='stable'); cand = order[:, self.rlo:self.rhi]
        cur = self.Et / max(self.wt, 1e-6) if self.bst_state == 0 else self.Ea / max(self.wa, 1e-6)
        g = lambda A: np.take_along_axis(A, cand, 1)
        X = np.empty((self.NL, cand.shape[1], 5), np.float32)
        X[..., 0] = g(self.E[0]) * ((1 - ag[0]) / G); X[..., 1] = g(self.E[1]) * ((1 - ag[1]) / G); X[..., 2] = g(cur)
        X[..., 3] = np.minimum(G * (b - g(self.last)), 1e5); X[..., 4] = g(self.h16)
        return X.reshape(-1, 5), cand, order[:, :self.rlo], e256

    def _score(self, X, cand, top, e256):
        pr = self.bst.predict(X, num_threads=self.nthr).reshape(self.NL, -1)
        S = np.zeros((self.NL, self.NE), np.float32)
        np.put_along_axis(S, cand, pr.astype(np.float32), 1); np.put_along_axis(S, top, 1e3 + e256[self.li, top], 1)
        return S

    def _refresh(self):
        self.stats['refreshes'] += 1
        f = self._features()
        if self.mode == 'sync':
            self.S = self._score(*f); return True
        applied = False
        if self._pending:                                  # result of the previous boundary -> apply now
            if self._res.empty(): self.stats['late_waits'] += 1
            self.S = self._res.get(); applied = True
        self._q.put(f); self._pending = True
        return applied
    _pending = False

    def _worker(self):
        while True:
            f = self._q.get()
            if f is None: return
            self._res.put(self._score(*f))

    def close(self):
        if self._th is not None: self._q.put(None); self._th.join(timeout=5)

    # ---------------------------------------------------------------- selection
    def _adj(self, resident):
        v = np.where(self.fixed, -np.inf, self.S).astype(np.float32)
        r = np.asarray(resident, bool) & ~self.fixed
        return np.where(r, v * np.float32(1 + self.hm) + np.float32(self.ha), v), r

    def target(self, resident):
        if self.S is None: return None
        v, r = self._adj(resident)
        tot = np.where(self.fixed, 0, np.maximum(self.S, 0)).sum(1)
        if np.ndim(self.nf) == 0:
            top = np.argsort(-v, 1, kind='stable')[:, :self.nf]
            want = np.zeros((self.NL, self.NE), bool); np.put_along_axis(want, top, True, 1)
        else:                                        # nq-lalloc per-layer n_float
            from scheduler import nf_topmask; want = nf_topmask(v, self.nf)
        nz = tot <= 0; want[nz] = r[nz]
        return want

    def order_score(self, resident):
        return self._adj(resident)[0] if self.S is not None else None
