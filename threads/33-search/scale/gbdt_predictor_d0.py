"""GBDT floating-set predictor v2 + dyn0 features (T33j scale; features = T33g glib.dyn_feats 'dyn0' set, no priors).
Drop-in for streaming/gbdt_predictor_v2.GBDTPredictorV2: same constructor, same step(counts, ntok, token_ids,
new_request, sal) / target / order_score / close.  Needs a model whose feature_name() is
  FEATS (5) + FEATS_V2 (4) + DYN0 (9)   e.g. models/rALL_dyn0_cftf.txt (60 trees x 15 leaves, tweedie 1.5)
k=0 layout: GBDTPredictorD0(layers, {L: [] ...}, model_path=..., n_float=77, rlo=0, rhi=256).

dyn0 features, per (layer, expert), updated per 16-token block with the same decays/cadence as v2; all use the same
per-layer normaliser as v2 but from the float32 EMA256 arrays (glib), nrm = sum_e EMA256 sal / sum_e EMA256 hits:
  e256, ema64, ema512             EMA hit rates (half-life 256 / 64 / 512 tokens, per token)
  sema256, sema64, sema512        EMA salience rates / nrm
  r_sema128, r_e256               rank (0 = largest) of EMA128 salience / EMA256 hits among all 256 experts
  sal_share                       EMA128 salience / sum over the layer's experts
Extra state vs v2: 4 float64 [NL, NE] EMAs (hits and salience at h=64, 512).  Extra per-refresh work: ~8 [NL,256]
EMA scalings, 2 argsort-ranks and 2 row sums, then the GBDT on 18 instead of 9 columns.
Offline reference: threads/33-search/gbdt/glib.dyn_feats (+ scale/scalelib.feats for the base 9); parity:
threads/33-search/scale/parity_d0.py (exact scores and top-77 sets, correct-flag and stuck-flag)."""
import sys

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/streaming")
from gbdt_predictor import GBDTPredictor, DEFAULT_MODEL, FEATS, G  # noqa: E402
from gbdt_predictor_v2 import GBDTPredictorV2, FEATS_V2  # noqa: E402

DYN0 = ("e256", "sema256", "ema64", "sema64", "ema512", "sema512", "r_sema128", "r_e256", "sal_share")


def _rank(A):
    """== argsort(argsort(-A, stable)) as float32, via the inverse permutation (one sort)."""
    o = np.argsort(-A, 1, kind="stable")
    r = np.empty(A.shape, np.float32)
    np.put_along_axis(r, o, np.arange(A.shape[1], dtype=np.float32)[None], 1)
    return r


class GBDTPredictorD0(GBDTPredictorV2):
    def __init__(self, layers, fixed, model_path=None, scale=None, **kw):
        import lightgbm as lgb
        GBDTPredictor.__init__(self, layers, fixed, model_path=DEFAULT_MODEL, **kw)
        self.bst = lgb.Booster(model_file=model_path)
        names = tuple(self.bst.feature_name())
        assert names == FEATS + FEATS_V2 + DYN0, names
        assert scale is None
        self.v2, self.scale = True, None
        self.sag = [0.5 ** (G / h) for h in (32, 128, 256)]
        self.dag = [0.5 ** (G / h) for h in (64, 512)]
        z = lambda: np.zeros((self.NL, self.NE), np.float64)  # noqa: E731
        self.Es = [z() for _ in self.sag]; self.Ec = [z() for _ in self.sag]
        self.Ds = [z() for _ in self.dag]; self.Dc = [z() for _ in self.dag]
        self.bs = z(); self.s16 = z()

    def _close_block(self):
        s = self.bs.astype(np.float32).astype(np.float64)
        c = self.bc.astype(np.float64)
        for k, a in enumerate(self.dag):
            self.Ds[k] = self.Ds[k] * a + s
            self.Dc[k] = self.Dc[k] * a + c
        super()._close_block()                                  # v2: Es/Ec(32,128,256), s16, then base block close

    def _dyn(self):
        """[NL, NE, 9] float32 in DYN0 order, all experts (glib.dyn_feats arithmetic)."""
        f = lambda E, a: (E * ((1 - a) / G)).astype(np.float32)  # noqa: E731   glib.ema_seg output (float32)
        a32, a128, a256 = self.sag; a64, a512 = self.dag
        e256, s256 = f(self.Ec[2], a256), f(self.Es[2], a256)
        nrm = s256.sum(1) / np.maximum(e256.sum(1), 1e-30)
        nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
        s128 = f(self.Es[1], a128)
        D = [e256, s256 / nrm, f(self.Dc[0], a64), f(self.Ds[0], a64) / nrm, f(self.Dc[1], a512), f(self.Ds[1], a512) / nrm,
             _rank(s128), _rank(e256), s128 / np.maximum(s128.sum(1, keepdims=True), 1e-30)]
        return np.stack(D, -1).astype(np.float32)

    def _features(self):
        X, cand, top, e256 = GBDTPredictor._features(self)
        F = self._salfeats(cand)
        fi = (np.arange(self.NL)[:, None] * self.NE + cand).ravel()
        Dy = self._dyn().reshape(-1, 9)[fi]
        X = np.concatenate([X, F.reshape(-1, 4), Dy], 1)
        return X, cand, top, e256
