"""Inkling-2-Mini environment (from the Inkling-2-Small one: D 2304, F 1152, 128 experts, NSH 1, 24 layers) for the T12/T19/T25/T35 encode stack; import before any of it.

  - nq19: D 6144 -> 4608, F 2048 -> 2304, NEXP 256 -> 384 (nq19_load does `from nq19 import D, F`, so this must run
    before nq19_load is imported).
  - orbit_duet.source.weights: Inkling native expert weights (NVFP4 L0-39 / MXFP8 L40-47) via inkdq, [gate, up, down]
    as fp32 [F,D], [F,D], [D,F].
  - TP shard layout: NSH = 2 (target 2x RTX PRO 6000).  A shard = F/2 = 1152 intermediate channels:
    gate/up = 72 strips of 16 x all 36 chunks, down = 9 chunks of 128 x all 288 strips -> 2592 units per shard.
    (GLM: 256 channels = 16 strips / 2 chunks, 768 units, NSH 8.  2304/8 = 288 is not a multiple of 128, so TP8/TP4
    shards cannot hold whole 128-row down chunks; TP2 can.)
  - nq_encode.BASE_K from NQ35_BASE_K (nq15 pattern-rate base; Inkling b175 = 1.75).
No T29 (Had512 down) anywhere: plain Had128."""
import os, sys, json
import torch

NQ = "/home/coder/git/nestquant/threads"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (f"{NQ}/12-reference-encoder", f"{NQ}/19-full-capture", f"{NQ}/25-campaign", f"{NQ}/26-vision-calib",
          f"{NQ}/35-nq15", f"{NQ}/05-exl3-harness", "/home/coder/git/orbit-duet", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")

import inkdq as Q                             # noqa: E402

D, F, NEXP = Q.D, Q.F, Q.NE
NSH = 1                                       # one GPU (12 GB target): no TP split
LAYERS = list(range(0, 24))
NFIX = int(os.environ.get("INK_NFIX", "0"))

import nq19                                   # noqa: E402
nq19.D, nq19.F, nq19.NEXP, nq19.SRC = D, F, NEXP, Q.CK
import nq19_load                              # noqa: E402
nq19_load.D, nq19_load.F = D, F

import orbit_duet.source as OS                # noqa: E402

_LE = {}


def weights(source, layer, expert, device="cuda"):
    """[gate [F,D], up [F,D], down [D,F]] fp32 from the native checkpoint (layer kept resident on CPU between calls)."""
    if layer not in _LE:
        _LE.clear()
        _LE[layer] = Q.LayerExperts(layer, "cpu")
    gu, dn = _LE[layer].w(expert, torch.float32)
    out = [gu[:F].to(device), gu[F:].to(device), dn.to(device)]
    if [list(w.shape) for w in out] != [[F, D], [F, D], [D, F]]:
        raise ValueError(f"Inkling expert shapes {[list(w.shape) for w in out]}")
    for w in out:
        if not torch.isfinite(w).all():
            raise ValueError("Nonfinite source weights")
    return out


OS.weights = weights

import nq15                                   # noqa: E402,F401  base-K extension over nq_encode / nq_decode
import nq_encode as NE                        # noqa: E402
import nq_decode as ND                        # noqa: E402
NE.BASE_K = float(os.environ.get("NQ35_BASE_K", "1.75"))

SH_STRIPS, SH_CHUNKS = F // NSH // 16, F // NSH // 128          # 72, 9


def unit_order(tk, tn, shard_axis, device="cpu"):
    """nq_decode.unit_order with the Inkling TP2 shard width (F/2 channels)."""
    a = torch.arange(tk, device=device).repeat_interleave(tn)
    c = torch.arange(tn, device=device).repeat(tk)
    shard = c // SH_STRIPS if shard_axis == "n" else a // SH_CHUNKS
    key = (shard * tn + c) * tk + a
    return torch.argsort(key), shard


ND.unit_order = unit_order
NE.D.unit_order = unit_order if getattr(NE, "D", None) is ND else NE.D.unit_order
NE.PROD["units_per_shard"] = (D // 128) * SH_STRIPS                  # Mini: 18 * 72 = 1296 (= 9 * 144 for down)

import nq_layer as NL                         # noqa: E402
NL.NSH = NSH
NL.FIXED_RULE = dict(NL.FIXED_RULE, n=NFIX,
                     rule=f"top {NFIX} experts per layer by token-weighted REAP = always level 4 (default allocation; "
                          "0 = k0 layout, every hot slot floating)")
try:
    import nq25_st                            # noqa: E402
    nq25_st.NSH = NSH
except Exception:
    pass
