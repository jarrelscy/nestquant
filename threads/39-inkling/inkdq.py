"""Inkling-2-Small expert dequant: NVFP4 (layers 0-39, e2m1 nibbles + e4m3 per-16 scale + per-expert amax/denominator)
and MXFP8 (layers 40-47, e4m3 + ue8m0 per 32x32 block). Returns bf16 [gate; up] (2F x D) and down (D x F)."""
import json, torch
from safetensors import safe_open

CK = "/tmp/nq_inkling/model"
D, F, NE = 4608, 2304, 384
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
_IDX = None; _FH = {}


def idx():
    global _IDX
    if _IDX is None:
        _IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
    return _IDX


def raw(k):
    fn = idx()[k]
    if fn not in _FH:
        _FH[fn] = safe_open(f"{CK}/{fn}", "pt", device="cpu")
    return _FH[fn].get_tensor(k)


def has(k):
    return k in idx()


def deinterleave(w):  # checkpoint rows [g0,u0,g1,u1,...] -> [g; u]
    return torch.cat([w[0::2], w[1::2]], 0)


def fp4_unpack(u8, lut):
    lo = (u8 & 0x0F).long(); hi = (u8 >> 4).long()
    return torch.stack([lut[lo], lut[hi]], -1).flatten(-2)   # low nibble = first element


class LayerExperts:
    """All 384 experts of one layer resident on `dev` in the source format; .w(e) -> (Wgu bf16 [2F,D], Wd bf16 [D,F])."""

    def __init__(self, L, dev):
        p = f"model.llm.layers.{L}.mlp.experts."
        self.dev = dev
        self.fp4 = has(p + "w13_weight_amax")
        self.w13 = raw(p + "w13_weight").to(dev); self.s13 = raw(p + "w13_weight_scale").to(dev)
        self.w2 = raw(p + "w2_weight").to(dev); self.s2 = raw(p + "w2_weight_scale").to(dev)
        if self.fp4:
            self.a13 = raw(p + "w13_weight_amax").float().to(dev); self.a2 = raw(p + "w2_weight_amax").float().to(dev)
            self.d13 = float(raw(p + "w13_weight_outer_scale_denominator")); self.d2 = float(raw(p + "w2_weight_outer_scale_denominator"))
            self.lut = E2M1.to(dev)

    def _nv(self, w, s, amax, den):
        x = fp4_unpack(w, self.lut)                         # [R, C]
        sc = s.float().repeat_interleave(16, -1) * (amax / den)
        return x * sc

    def _mx(self, w, s):
        sf = torch.exp2(s.view(torch.uint8).float() - 127.) if s.dtype in (torch.uint8, torch.float8_e8m0fnu) else s.float()
        r, c = w.shape
        return (w.float().view(sf.shape[0], r // sf.shape[0], sf.shape[1], c // sf.shape[1]) * sf[:, None, :, None]).view(r, c)

    def w(self, e, dtype=torch.bfloat16):
        if self.fp4:
            gu = self._nv(self.w13[e], self.s13[e], self.a13[e], self.d13)
            dn = self._nv(self.w2[e], self.s2[e], self.a2[e], self.d2)
        else:
            gu = self._mx(self.w13[e], self.s13[e]); dn = self._mx(self.w2[e], self.s2[e])
        return deinterleave(gu).to(dtype), dn.to(dtype)
