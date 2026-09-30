"""T35 Part B: rate-b base emulation inside the T18 e2e harness (py: plug-in, no harness edits).

AdaptEmu = quantisers.Adapt (jF floating set, per-token levels) whose COLD (level-2) experts are replaced by
    W_b = W_ref + s * (W_nq2 - W_ref),   s = sqrt(r(b) / r(2))
W_ref = the harness's FP8 reference dequant (ref(), the same weights the fp8 arm uses), W_nq2 = the real predecoded
NestQuant L2 decode (keeps its LDLQ/Hessian error shaping).  Hot (level-4) experts are the real L4 decode, untouched.
Math in fp32, stored in the predecode dtype (fp16); at s = 1 this returns W_nq2 bit-exactly.
  --cand NAME=py:/home/coder/git/nestquant/threads/35-nq15/emu_adapt.py:AdaptEmu:s=1.41,lo=...,hi=...,<Adapt args>
"""
import sys
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import quantisers as Q


class AdaptEmu(Q.Adapt):
    def __init__(self, s="1.0", b=None, **kw):
        super().__init__(**kw)
        self.s = float(s)
        self.b = b
        self.n_emu = 0

    def expert_level(self, layer, expert, ref, level):
        W = super().expert_level(layer, expert, ref, level)
        if level == 4 or W is None:
            return W
        R = ref()
        self.n_emu += 1
        out = {}
        for p, w in W.items():
            r = R[p].float()
            out[p] = (r + self.s * (w.float() - r)).to(w.dtype)          # fp16 like the nq2 predecode
        return out
