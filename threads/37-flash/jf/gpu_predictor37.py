"""T37 (GLM-5.3-Flash: NE 288, layers 3-44, 19 fixed + 48 floating; all read from jF.pt) port of the T33i GPU serve path for the joint predictor: all per-block state, the 22 inputs, v2's 60 LightGBM trees and the
transformer residual run as torch ops on one device (1 CUDA stream); only the think/answer token bookkeeping and two
scalar weights stay on the host.  Same interface as streaming/gbdt_predictor_v2.GBDTPredictorV2 (mode='sync'):
  p = GPUJointPredictor(layers, fixed, 'jF.pt', hm=0.7, device='cuda'|'cpu'|'mps')   # n_float / NE from jF.pt
  p.step(counts[NL,NE] (np or torch), ntok, token_ids, new_request, sal[NL,NE]) -> bool   (True at a refresh)
  p.target(resident) -> want bool [NL,NE] (np) ; p.order_score(resident) ; p.S (np f32 scores)
Arithmetic mirrors joint_predictor(37).JointPredictor (numpy): float32 for the serve-v2 EMAs / mem_cur_state, float64 for
the salience + extra EMAs and the tree comparisons (LightGBM compares double(x) <= threshold).  Parity:
parity_gpu.py.  Fresh instance per chain (offline features restart at every chain)."""
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
G, THINK_ID, ETHINK_ID = 16, 154841, 154842                    # = streaming/gbdt_predictor (self-contained here)
try:
    import train37 as TR                                       # noqa: E402
except ImportError:                                            # release layout
    import train as TR                                         # noqa: E402

V2_MODEL = os.path.join(_here, "v2_sal_tweedie1.5.txt")
HC = (8, 32, 64, 128, 256, 512, 2048)
RATE = [f"ema{h}" for h in HC] + [f"sema{h}" for h in HC] + ["hits16", "sal16"]


class TorchTrees:
    """LightGBM numerical '<=' trees (missing None) -> batched torch evaluation; tweedie -> exp(sum)."""

    def __init__(self, model_file, device):
        import lightgbm as lgb
        m = lgb.Booster(model_file=model_file).dump_model()
        assert m["objective"].startswith("tweedie") and m["num_tree_per_iteration"] == 1
        self.names = m["feature_names"]
        T = len(m["tree_info"]); N = max(2 * t["num_leaves"] - 1 for t in m["tree_info"])
        feat = np.zeros((T, N), np.int64); thr = np.full((T, N), np.inf); L = np.zeros((T, N), np.int64)
        R = np.zeros((T, N), np.int64); val = np.zeros((T, N)); depth = 0
        for ti, t in enumerate(m["tree_info"]):
            nxt = [0]

            def add(n, d):
                nonlocal depth
                i = nxt[0]; nxt[0] += 1
                if "leaf_value" in n:
                    L[ti, i] = R[ti, i] = i; val[ti, i] = n["leaf_value"]; depth = max(depth, d); return i
                assert n["decision_type"] == "<=" and n["missing_type"] == "None"
                feat[ti, i] = n["split_feature"]; thr[ti, i] = n["threshold"]
                L[ti, i] = add(n["left_child"], d + 1); R[ti, i] = add(n["right_child"], d + 1)
                return i
            add(t["tree_structure"], 0)
        tt = lambda a: torch.from_numpy(a).to(device)            # noqa: E731
        self.feat, self.thr, self.L, self.R, self.val = tt(feat), tt(thr), tt(L), tt(R), tt(val)
        self.depth, self.T = depth, T
        self.tix = torch.arange(T, device=device)

    def __call__(self, X):                                        # X [n, F] float32 -> [n] float64
        x = X.double()
        node = torch.zeros(X.shape[0], self.T, dtype=torch.long, device=X.device)
        for _ in range(self.depth):
            f = self.feat[self.tix, node]; th = self.thr[self.tix, node]
            go = torch.gather(x, 1, f) <= th
            node = torch.where(go, self.L[self.tix, node], self.R[self.tix, node])
        return torch.exp(self.val[self.tix, node].sum(1))


class GPUJointPredictor:
    def __init__(self, layers, fixed, net_path, n_float=None, hm=0.7, ha=0.0, device="cuda", v2_model=V2_MODEL,
                 think_id=THINK_ID, ethink_id=ETHINK_ID, max_decode_ntok=16, graph=False, bf16=True):
        ck = torch.load(net_path, map_location="cpu", weights_only=False); a = ck["args"]
        self.layers = list(layers); NL = self.NL = len(self.layers); NE = self.NE = ck["ne"]
        n_float = n_float or ck["nf"]
        self.nf, self.hm, self.ha, self.dev = n_float, hm, ha, device
        self.think_id, self.ethink_id, self.max_ntok = think_id, ethink_id, max_decode_ntok
        self.fixed = np.zeros((NL, NE), bool)
        for i, L in enumerate(self.layers):
            self.fixed[i, list(fixed[L])] = True
        self.trees = TorchTrees(v2_model, device)
        self.net = TR.Net(ck["K"], a["arch"], a["d"], a["nl"], nf=ck["nf"], ne=NE, nlayers=len(ck["layers"])).to(device).eval(); self.net.load_state_dict(ck["state"])
        self.lidx = torch.tensor([ck["layers"].index(L) for L in self.layers], device=device)
        self.fxm = torch.from_numpy(self.fixed).to(device)
        z32 = lambda: torch.zeros(NL, NE, dtype=torch.float32, device=device)    # noqa: E731
        z64 = lambda: torch.zeros(NL, NE, dtype=torch.float64, device=device)    # noqa: E731
        self.ag = [np.float32(0.5 ** (G / h)) for h in (32, 128)]
        self.E = [z32(), z32()]
        self.sa = 0.5 ** (1 / 2048); self.Et, self.Ea = z32(), z32(); self.wt = self.wa = 0.0
        self.ac = {h: 0.5 ** (G / h) for h in HC}
        self.Hc = {h: z64() for h in HC}; self.Hs = {h: z64() for h in HC}
        self.last = torch.full((NL, NE), -10 ** 6, dtype=torch.long, device=device)
        self.bc, self.bca, self.bs = z32(), z32(), z64()
        self.h16, self.s16 = z32(), z64()
        self.btok = self.bans = self.nblk = 0; self.seg = 0; self.bst_state = 0
        self.S = None
        self.bf16 = bf16 and str(device).startswith("cuda")
        # device scalars read by the (graph-capturable, pure) scoring core; host scalars mirror into them per block
        self.b_wt = torch.ones((), dtype=torch.float64, device=device); self.b_wa = torch.ones_like(self.b_wt)
        self.b_state = torch.zeros((), dtype=torch.bool, device=device)
        self.b_nblk = torch.zeros((), dtype=torch.long, device=device)
        self.b_pos = torch.zeros((), dtype=torch.float32, device=device)
        self.graph = None
        if graph:
            assert device != "cpu"
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    self._core()
            torch.cuda.current_stream().wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.g_out = self._core()

    def _t(self, a, dt):
        return a.to(self.dev, dt) if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)

    def step(self, counts, ntok=1, token_ids=None, new_request=False, sal=None):
        if ntok > self.max_ntok:
            return False
        c = self._t(counts, torch.float32)
        if new_request:
            self.seg = 0
        if token_ids is not None:
            for t in token_ids:
                if t == self.think_id: self.seg = 0
                elif t == self.ethink_id: self.seg = 1
        self.bc += c
        if self.seg == 1:
            self.bca += c; self.bans += ntok
        if sal is not None:
            self.bs += self._t(sal, torch.float64)
        self.btok += ntok
        if self.btok < G:
            return False
        self._close_block(); self.S = self._score(); return True

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
        self.bc.copy_(self._t(c, torch.float32))
        if ca is None:
            self.bca.zero_()
        else:
            self.bca.copy_(self._t(ca, torch.float32))
        self.bans = na; self.bs.copy_(self._t(s, torch.float64)); self.seg = sl; self.btok = G

    def _score_now(self):
        return self._score()

    def _close_block(self):
        """in-place state updates (the captured graph reads these exact tensors)."""
        c = self.bc; Can = self.bca; na = np.float64(self.bans); j = self.nblk
        s = self.bs.float().double(); c64 = c.double()
        for h in HC:
            self.Hc[h].mul_(self.ac[h]).add_(c64); self.Hs[h].mul_(self.ac[h]).add_(s)
        for k in range(2):
            self.E[k].mul_(float(self.ag[k])).add_(c)
        nt = G - na; dt = np.float32(self.sa ** nt); da = np.float32(self.sa ** na)
        self.Et.mul_(float(dt)).add_(c - Can); self.wt = self.wt * dt + nt
        self.Ea.mul_(float(da)).add_(Can); self.wa = self.wa * da + na
        self.last.masked_fill_(c > 0, j)
        self.h16.copy_(c); self.s16.copy_(s); self.bst_state = self.seg; self.nblk += 1
        self.b_wt.fill_(float(max(self.wt, 1e-6))); self.b_wa.fill_(float(max(self.wa, 1e-6)))
        self.b_state.fill_(bool(self.bst_state)); self.b_nblk.fill_(self.nblk)
        self.b_pos.fill_(float(np.float32(np.log1p(self.nblk - 1))))
        self.btok -= G; self.bc.zero_(); self.bca.zero_(); self.bs.zero_()
        self.bans = 0

    @torch.no_grad()
    def features(self):
        sc = lambda h: (1 - self.ac[h]) / G                    # noqa: E731
        Ec = {h: self.Hc[h] * sc(h) for h in HC}; Es = {h: self.Hs[h] * sc(h) for h in HC}
        nrm = Es[256].sum(1) / Ec[256].sum(1).clamp(min=1e-30)
        nrm = torch.where(nrm > 0, nrm, torch.ones_like(nrm))[:, None]
        F = {f"ema{h}": Ec[h].float() for h in HC}
        F["ema32"] = self.E[0] * np.float32((1 - self.ag[0]) / G)
        F["ema128"] = self.E[1] * np.float32((1 - self.ag[1]) / G)
        for h in HC:
            F[f"sema{h}"] = (Es[h] / nrm).float()
        F["hits16"] = self.h16; F["sal16"] = (self.s16 / nrm).float()
        for h in (128, 512):
            F[f"mps{h}"] = torch.where(Ec[h] / sc(h) > 1e-3, Es[h] / Ec[h].clamp(min=1e-30) / nrm,
                                       torch.ones_like(Ec[h])).float()
        F["tok_since_hit"] = (G * (self.b_nblk - self.last)).clamp(max=100000).float()
        F["mem_cur_state"] = torch.where(self.b_state, self.Ea.double() / self.b_wa,
                                         self.Et.double() / self.b_wt).float()      # numpy: f32 / f64 scalar -> f64
        F["_pos"] = self.b_pos.expand(self.NL, self.NE)
        X9 = torch.stack([F[n] for n in self.trees.names], -1).reshape(-1, 9)
        P = self.trees(X9).float().reshape(self.NL, self.NE)
        mx = lambda v: v.clamp(min=0)                          # noqa: E731
        cols = [torch.log1p(mx(F[n]) * (16.0 if n not in ("hits16", "sal16") else 1.0)) for n in RATE]
        cols += [torch.log1p(mx(F["mem_cur_state"]) * 16.0), torch.log1p(F["tok_since_hit"]) / 5.0,
                 torch.log(F["mps128"].clamp(min=1e-3)), torch.log(F["mps512"].clamp(min=1e-3)), F["_pos"] / 5.0,
                 torch.log(P.clamp(min=1e-4))]
        return torch.stack(cols, -1).half(), P

    @torch.no_grad()
    def _core(self):
        X, P = self.features()
        lp = torch.log(P.clamp(min=1e-30))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            r = self.net(X, lp, self.lidx, self.fxm).float()
        return torch.exp((lp + r).clamp(max=30))

    @torch.no_grad()
    def _score(self):
        if self.graph is not None:
            self.graph.replay(); return self.g_out.cpu().numpy()
        return self._core().cpu().numpy()

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
