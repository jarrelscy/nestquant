"""T33i joint floating-set predictor (streaming).  Same interface as streaming/gbdt_predictor_v2.GBDTPredictorV2:
  p = JointPredictor(layers, fixed, net_path, n_float=77, hm=0.6, mode='sync'|'next_refresh', device='cuda')
  p.step(counts, ntok, token_ids, new_request, sal) -> bool ; p.target(resident) ; p.order_score(resident) ; p.close()
Score = exp(log v2(all 256 experts) + r), r = residual of a 2-layer transformer over the 256 experts of a layer
(d96, 4 heads, no expert-identity embedding: permutation-equivariant across experts, only a per-layer embedding).
Inputs per (layer, expert), 22 = jlib.INPUTS: hit EMAs h in {8,32,64,128,256,512,2048}, salience EMAs same half-lives
/ norm_L, hits16, sal16, mem_cur_state, tok_since_hit, mps128, mps512, log1p(block pos in chain), v2 prediction.
Offline definition: jlib.features / jlib.net_inputs / train.Net; parity: parity_stream.py.
Chain semantics: offline features start from zero at every 4x2048-token chain -> build a fresh predictor
at the start of each chain / request, exactly like floating_default resets."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/coder/git/nestquant/streaming")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbdt_predictor import G                                   # noqa: E402
from gbdt_predictor_v2 import GBDTPredictorV2                  # noqa: E402
import train as TR                                             # noqa: E402  (before jlib)
import jlib as J                                               # noqa: E402

V2_MODEL = "/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt"
HC = (8, 32, 64, 128, 256, 512, 2048)
HS = HC


class JointPredictor(GBDTPredictorV2):
    def __init__(self, layers, fixed, net_path, n_float=77, hm=0.6, device="cuda", v2_model=V2_MODEL, **kw):
        super().__init__(layers, fixed, model_path=v2_model, n_float=n_float, hm=hm, **kw)
        assert self.v2
        ck = torch.load(net_path, map_location="cpu")
        a = ck["args"]
        self.dev = device
        self.net = TR.Net(ck["K"], a["arch"], a["d"], a["nl"]).to(device).eval()
        self.net.load_state_dict(ck["state"])
        self.lidx = torch.tensor([TR.LAYERS.index(L) for L in self.layers], device=device)
        self.fxm = torch.from_numpy(self.fixed).to(device)
        z = lambda: np.zeros((self.NL, self.NE), np.float64)  # noqa: E731
        self.ac = {h: 0.5 ** (G / h) for h in HC}
        self.Hc = {h: z() for h in HC}
        self.Hs = {h: z() for h in HS}

    def _close_block(self):
        s = self.bs.astype(np.float32).astype(np.float64)
        c = self.bc.astype(np.float64)
        for h in HC:
            self.Hc[h] = self.Hc[h] * self.ac[h] + c
            self.Hs[h] = self.Hs[h] * self.ac[h] + s
        super()._close_block()

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
        Xv = np.stack([F[n] for n in self.bst.feature_name()], -1).reshape(-1, 9)
        P = self.bst.predict(Xv, num_threads=self.nthr).reshape(NL, NE).astype(np.float32)
        X = J.net_inputs(F, P)                                 # [NL, NE, 22] fp16 (offline identical)
        return X, P

    @torch.no_grad()
    def _score(self, X, P):
        x = torch.from_numpy(X).to(self.dev)
        lp = torch.from_numpy(np.log(np.maximum(P, 1e-30)).astype(np.float32)).to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            r = self.net(x, lp, self.lidx, self.fxm).float()
        return torch.exp((lp + r).clamp(max=30)).cpu().numpy().astype(np.float32)
