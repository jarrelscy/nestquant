"""GBDT floating-set predictor v2 (T32): gbdt_predictor.GBDTPredictor + salience inputs.  SERVE CHANGE: step() also
takes the per-step salience per expert, sal[len(layers), NE] = sum over the step's routed slots of w^2 * xn, where
  w  = final gate weight used in the MoE combine (incl. routed_scaling_factor)
  xn = sum(x^2) in fp32 of x = post_attention_layernorm(h) (the normalised MoE input; one scalar per token per layer)
Two uses:
  model with 9 features (FEATS + FEATS_V2, e.g. gbdt_v2sal_p64.txt): salience-target Tweedie model on the 5 count
      features + 4 salience features;
  scale='mps' with any 5-feature model (e.g. gbdt_p64_s5.txt): predicted hits x mps128 (the expert's EMA128 salience
      per hit relative to the layer's); forced EMA256-top-20 keep their 1e3+ scores.
Salience features, per 16-token block (same cadence/decays as ema32/ema128), scale-free per layer with
norm_L = sum_e EMA256 sal / sum_e EMA256 hits (causal):
  sema32, sema128  EMA rates of block salience / norm_L        sal16  last block salience / norm_L
  mps128           (EMA128 sal / EMA128 hits) / norm_L, 1.0 where EMA128 hits <= 1e-3
Offline definition + parity: threads/32-gbdt-sal/t32lib.v2_features, parity_v2.py."""
import numpy as np

from gbdt_predictor import GBDTPredictor, DEFAULT_MODEL, FEATS, G

FEATS_V2 = ("sema32", "sema128", "sal16", "mps128")


class GBDTPredictorV2(GBDTPredictor):
    def __init__(self, layers, fixed, model_path=None, scale=None, **kw):
        import lightgbm as lgb
        super().__init__(layers, fixed, model_path=DEFAULT_MODEL, **kw)
        self.bst = lgb.Booster(model_file=model_path or DEFAULT_MODEL)
        names = tuple(self.bst.feature_name())
        assert names in (FEATS, FEATS + FEATS_V2), names
        self.v2 = names == FEATS + FEATS_V2
        assert scale in (None, "mps"), scale
        self.scale = scale
        self.sag = [0.5 ** (G / h) for h in (32, 128, 256)]
        z = lambda: np.zeros((self.NL, self.NE), np.float64)  # noqa: E731
        self.Es = [z() for _ in self.sag]
        self.Ec = [z() for _ in self.sag]
        self.bs = z()
        self.s16 = z()

    def step(self, counts, ntok=1, token_ids=None, new_request=False, sal=None):
        if ntok > self.max_ntok:
            return False
        if sal is not None:
            self.bs += np.asarray(sal, np.float64)
        return super().step(counts, ntok, token_ids, new_request)

    def _close_block(self):
        s = self.bs.astype(np.float32).astype(np.float64)       # blocks are float32 in the offline pipeline
        c = self.bc.astype(np.float64)
        for k, a in enumerate(self.sag):
            self.Es[k] = self.Es[k] * a + s
            self.Ec[k] = self.Ec[k] * a + c
        self.s16 = s
        self.bs = np.zeros_like(self.bs)
        super()._close_block()

    def _salfeats(self, cand):
        norm = self.Es[2].sum(1) / np.maximum(self.Ec[2].sum(1), 1e-30)
        norm = np.where(norm > 0, norm, 1.0)[:, None]
        g = lambda A: np.take_along_axis(A, cand, 1)            # noqa: E731
        a = self.sag
        h = g(self.Ec[1])
        mps = np.where(h > 1e-3, g(self.Es[1]) / np.maximum(h, 1e-30) / norm, 1.0)
        return np.stack([g(self.Es[0]) * ((1 - a[0]) / G) / norm, g(self.Es[1]) * ((1 - a[1]) / G) / norm,
                         g(self.s16) / norm, mps], -1).astype(np.float32)

    def _features(self):
        X, cand, top, e256 = super()._features()
        F = self._salfeats(cand)
        if self.v2:
            X = np.concatenate([X.reshape(self.NL, cand.shape[1], 5), F], -1).reshape(-1, 9)
        return (X, cand, top, e256, F[..., 3]) if self.scale == "mps" else (X, cand, top, e256)

    def _score(self, X, cand, top, e256, mps=None):
        pr = self.bst.predict(X, num_threads=self.nthr).reshape(self.NL, -1)
        if mps is not None:
            pr = pr * mps
        S = np.zeros((self.NL, self.NE), np.float32)
        np.put_along_axis(S, cand, pr.astype(np.float32), 1)
        np.put_along_axis(S, top, 1e3 + e256[self.li, top], 1)
        return S
