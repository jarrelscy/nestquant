"""T37 jF floating-set predictor, numpy reference (port of threads/33-search/joint/joint_predictor.py for
GLM-5.3-Flash: 288 experts, layers 3-44, 19 fixed + 48 floating).  Self-contained: the GLM-5.3 version subclasses
streaming/gbdt_predictor_v2.GBDTPredictorV2, whose base constructor always loads the GLM-5.3 v1 model
(gbdt_p64_s5.txt, 5 features) that has no Flash counterpart; the v1/v2 serve-state arithmetic is therefore inlined
here line for line (float32 serve EMAs / mem_cur_state, float64 salience), mode='sync' only (the jF serve mode).
Offline definition: jlib37.features / net_inputs + train37.Net; parity: parity37.py.

  p = JointPredictor(layers, fixed, "jF.pt", hm=0.7, v2_model="v2_sal_tweedie1.5.txt")    # n_float / NE from jF.pt
  p.step(counts[NL, 288], ntok, token_ids, new_request, sal[NL, 288]) -> bool (True at a refresh, every 16 tokens)
  p.target(resident) -> want bool [NL, 288] floating set (fixed excluded) ; p.order_score(resident) ; p.S
  p.step_chunk(counts[nb, NL, 288], sal[nb, NL, 288]) ; p.target_seed()   prefill->decode handoff seeder (SERVE.md)
counts = routed-slot hits of this step's top-8; sal = sum over this step's routed slots of w^2 * xn, w = final combine
weight INCLUDING routed_scaling_factor (2.5), xn = fp32 sum(x^2) of the normalised MoE input.  Steps with ntok > 16
(prefill chunks) are ignored.  Build a fresh instance per request (features start at zero, set = floating_default).
Score = exp(log v2(all 288 experts) + r), r = 2-layer transformer residual over the 288 experts of each layer."""
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
try:
    import train37 as TR                                       # noqa: E402
    import jlib37 as J                                         # noqa: E402
except ImportError:                                            # release layout: train.py / jlib.py
    import train as TR                                         # noqa: E402
    import jlib as J                                           # noqa: E402

G = 16
THINK_ID, ETHINK_ID = 154841, 154842
V2_MODEL = os.path.join(_here, "v2_sal_tweedie1.5.txt")
HC = (8, 32, 64, 128, 256, 512, 2048)
HS = HC
V2_FEATS = ("ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128")


def load_net(net_path, device="cpu"):
    ck = torch.load(net_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = TR.Net(ck["K"], a["arch"], a["d"], a["nl"], nf=ck["nf"], ne=ck["ne"], nlayers=len(ck["layers"]))
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck


class JointPredictor:
    def __init__(self, layers, fixed, net_path, n_float=None, hm=0.7, ha=0.0, device="cpu", v2_model=None,
                 num_threads=4, think_id=THINK_ID, ethink_id=ETHINK_ID, max_decode_ntok=16, mode="sync", **_):
        import lightgbm as lgb
        assert mode == "sync", "jF reference supports mode='sync' only"
        self.net, ck = load_net(net_path, device)
        self.layers = list(layers); NL = self.NL = len(self.layers); NE = self.NE = ck["ne"]
        self.nf = n_float or ck["nf"]; self.hm, self.ha, self.dev, self.nthr = hm, ha, device, num_threads
        self.think_id, self.ethink_id, self.max_ntok = think_id, ethink_id, max_decode_ntok
        self.fixed = np.zeros((NL, NE), bool)
        for i, L in enumerate(self.layers):
            self.fixed[i, list(fixed[L])] = True
        self.bst = lgb.Booster(model_file=v2_model or V2_MODEL)
        assert tuple(self.bst.feature_name()) == V2_FEATS, self.bst.feature_name()
        self.lidx = torch.tensor([ck["layers"].index(L) for L in self.layers], device=device)
        self.fxm = torch.from_numpy(self.fixed).to(device)
        z64 = lambda: np.zeros((NL, NE), np.float64)  # noqa: E731
        # serve v1/v2 state (= gbdt_predictor.GBDTPredictor + GBDTPredictorV2)
        self.ag = [np.float32(0.5 ** (G / h)) for h in (32, 128, 256)]
        self.E = [np.zeros((NL, NE), np.float32) for _ in self.ag]
        self.sa = 0.5 ** (1 / 2048); self.Et = np.zeros((NL, NE), np.float32); self.Ea = np.zeros_like(self.Et)
        self.wt = self.wa = 0.0
        self.last = np.full((NL, NE), -10 ** 6, np.int64)
        self.nblk = 0
        self.bc = np.zeros((NL, NE), np.float32); self.bca = np.zeros((NL, NE), np.float32); self.btok = 0; self.bans = 0
        self.bs = z64(); self.s16 = z64(); self.h16 = np.zeros((NL, NE), np.float32)
        self.seg = 0; self.bst_state = 0
        self.S = None
        # jF extra EMAs (float64)
        self.ac = {h: 0.5 ** (G / h) for h in HC}
        self.Hc = {h: z64() for h in HC}
        self.Hs = {h: z64() for h in HS}

    def step(self, counts, ntok=1, token_ids=None, new_request=False, sal=None):
        if ntok > self.max_ntok:
            return False
        if sal is not None:
            self.bs += np.asarray(sal, np.float64)
        c = np.asarray(counts, np.float32)
        if new_request:
            self.seg = 0
        if token_ids is not None:
            for t in token_ids:
                if t == self.think_id: self.seg = 0
                elif t == self.ethink_id: self.seg = 1
        self.bc += c                                           # a step's counts go to the segment of its last token
        if self.seg == 1:
            self.bca += c; self.bans += ntok
        self.btok += ntok
        if self.btok < G:
            return False
        self._close_block()
        self.S = self._score(*self._features())
        return True

    def step_chunk(self, counts, sal, counts_ans=None, n_ans=None, seg_last=None):
        """HANDOFF SEEDER (Mac layer-major prefill, SPEC sec. 4).  Folds the prompt tail's routing into the state, one
        16-token block at a time, with NO per-block scoring and NO set changes; scores once at the end.  Then call
        target_seed() once (= refresh at hm 0) for the floating set of the first decode block; decode then proceeds
        with step() / target() as usual.  Use the last ~4096 prompt tokens (256 blocks; the longest EMA is 2048).
          counts, sal [nb, NL, NE]: per 16-token prompt block, routed-slot hits / sum w^2 xn (as in step());
          counts_ans [nb, NL, NE], n_ans [nb], seg_last [nb]: optional answer-segment split (after </think>) of each
          block, = the offline block format (counts of answer-segment tokens, their number, state at the block's last
          token); default all think.
        Must start on a block boundary of a fresh instance (or after whole blocks)."""
        assert self.btok == 0, "step_chunk must start on a 16-token block boundary"
        nb = len(counts)
        for b in range(nb):
            self._load_block(counts[b], sal[b], None if counts_ans is None else counts_ans[b],
                             0 if n_ans is None else int(n_ans[b]), 0 if seg_last is None else int(seg_last[b]))
            self._close_block()
        self.seg = self.bst_state
        if nb:
            self.S = self._score_now()
        return nb > 0

    def target_seed(self):
        """the handoff refresh: top-n_float non-fixed experts at hm 0 (no resident set).  None if nothing scored;
        a layer with no positive score keeps no floating expert (caller: use floating_default there)."""
        return self.target(np.zeros((self.NL, self.NE), bool))

    def _load_block(self, c, s, ca, na, sl):
        self.bc = np.asarray(c, np.float32).copy()
        self.bca = np.zeros_like(self.bc) if ca is None else np.asarray(ca, np.float32).copy()
        self.bans = na; self.bs = np.asarray(s, np.float64).copy(); self.seg = sl; self.btok = G

    def _score_now(self):
        return self._score(*self._features())

    def _close_block(self):
        s = self.bs.astype(np.float32).astype(np.float64)       # blocks are float32 in the offline pipeline
        c64 = self.bc.astype(np.float64)
        for h in HC:
            self.Hc[h] = self.Hc[h] * self.ac[h] + c64
            self.Hs[h] = self.Hs[h] * self.ac[h] + s
        self.s16 = s; self.bs = np.zeros_like(self.bs)
        c = self.bc; Can = self.bca; Cth = c - Can; na = np.float64(self.bans); j = self.nblk
        for k in range(len(self.ag)):
            self.E[k] = self.E[k] * self.ag[k] + c
        nt = G - na; dt = np.float32(self.sa ** nt); da = np.float32(self.sa ** na)
        self.Et = self.Et * dt + Cth; self.wt = self.wt * dt + nt; self.Ea = self.Ea * da + Can; self.wa = self.wa * da + na
        self.last = np.where(c > 0, j, self.last); self.h16 = c
        self.bst_state = self.seg; self.nblk += 1
        self.btok -= G; self.bc = np.zeros_like(c); self.bca = np.zeros_like(c); self.bans = 0

    def _features(self):
        NL, NE, ag = self.NL, self.NE, self.ag
        sc = lambda h: (1 - self.ac[h]) / G                    # noqa: E731
        Ec = {h: self.Hc[h] * sc(h) for h in HC}
        Es = {h: self.Hs[h] * sc(h) for h in HS}
        nrm = Es[256].sum(1) / np.maximum(Ec[256].sum(1), 1e-30)
        nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
        F = {f"ema{h}": Ec[h] for h in HC}
        F["ema32"] = self.E[0] * ((1 - ag[0]) / G)             # serve float32 recursion (= v2 inputs)
        F["ema128"] = self.E[1] * ((1 - ag[1]) / G)
        for h in HS:
            F[f"sema{h}"] = Es[h] / nrm
        F["hits16"] = self.h16.astype(np.float64)
        F["sal16"] = self.s16 / nrm
        for h in (128, 512):
            a = sc(h)
            F[f"mps{h}"] = np.where(Ec[h] / a > 1e-3, Es[h] / np.maximum(Ec[h], 1e-30) / nrm, 1.0)
        F["tok_since_hit"] = np.minimum(G * (self.nblk - self.last), 1e5)
        F["mem_cur_state"] = self.Et / max(self.wt, 1e-6) if self.bst_state == 0 else self.Ea / max(self.wa, 1e-6)
        F["_pos"] = np.full((NL, NE), np.log1p(self.nblk - 1))
        F = {k: np.asarray(v, np.float32) for k, v in F.items()}
        Xv = np.stack([F[n] for n in V2_FEATS], -1).reshape(-1, 9)
        P = self.bst.predict(Xv, num_threads=self.nthr).reshape(NL, NE).astype(np.float32)
        X = J.net_inputs(F, P)                                 # [NL, NE, 22] fp16 (offline identical)
        return X, P

    @torch.no_grad()
    def _score(self, X, P):
        x = torch.from_numpy(X).to(self.dev)
        lp = torch.from_numpy(np.log(np.maximum(P, 1e-30)).astype(np.float32)).to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
            r = self.net(x, lp, self.lidx, self.fxm).float()
        return torch.exp((lp + r).clamp(max=30)).cpu().numpy().astype(np.float32)

    # ---------------------------------------------------------------- selection (= GBDTPredictor)
    def _adj(self, resident):
        v = np.where(self.fixed, -np.inf, self.S).astype(np.float32)
        r = np.asarray(resident, bool) & ~self.fixed
        return np.where(r, v * np.float32(1 + self.hm) + np.float32(self.ha), v), r

    def target(self, resident):
        if self.S is None:
            return None
        v, r = self._adj(resident)
        tot = np.where(self.fixed, 0, np.maximum(self.S, 0)).sum(1)
        top = np.argsort(-v, 1, kind="stable")[:, :self.nf]
        want = np.zeros((self.NL, self.NE), bool); np.put_along_axis(want, top, True, 1)
        nz = tot <= 0; want[nz] = r[nz]
        return want

    def order_score(self, resident):
        return self._adj(resident)[0] if self.S is not None else None

    def close(self):
        pass
