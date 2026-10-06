"""T37 (GLM-5.3-Flash) environment for the T12/T19/T25/T26 encode stack, imported before any of it.

  - nq19: D 6144 -> 4096, NEXP 256 -> 288, SRC -> the Flash FP8 checkpoint (nq19_load does `from nq19 import D`, so
    this must run before nq19_load is imported).
  - orbit_duet.source.weights: Flash keys (model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj), block
    FP8 dequant via orbit's own dequantize, shape check [2048,4096]x2 + [4096,2048] (orbit's version demands 6144).
  - nq_layer.FIXED_RULE: n 19 (T37 REAP fixed set).
  - nq_encode.BASE_K from NQ35_BASE_K (nq15 pattern-rate base; T37 = 1.5).
No T29 (Had512 down) anywhere: Flash layers are plain Had128."""
import os, sys, json
import torch

NQ = "/home/coder/git/nestquant/threads"
for p in (f"{NQ}/12-reference-encoder", f"{NQ}/19-full-capture", f"{NQ}/25-campaign", f"{NQ}/26-vision-calib",
          f"{NQ}/35-nq15", f"{NQ}/37-flash", f"{NQ}/05-exl3-harness", "/home/coder/git/orbit-duet"):
    if p not in sys.path:
        sys.path.insert(0, p)

SRC = os.environ.get("NQ37_SRC", "/tmp/nestquant/37-flash/fp8")
D, F, NEXP, NFIX = 4096, 2048, 288, 19
LAYERS = list(range(3, 45))

import nq19                                   # noqa: E402
nq19.D, nq19.NEXP, nq19.SRC = D, NEXP, SRC
import nq19_load                              # noqa: E402
nq19_load.D = D

import orbit_duet.source as OS                # noqa: E402
from safetensors import safe_open             # noqa: E402

_idx, _fh = {}, {}


def _map(source):
    if source not in _idx:
        _idx[source] = json.load(open(f"{source}/model.safetensors.index.json"))["weight_map"]
    return _idx[source]


def _get(source, key):
    f = f"{source}/{_map(source)[key]}"
    if f not in _fh:
        _fh[f] = safe_open(f, framework="pt", device="cpu")
    return _fh[f].get_tensor(key)


def weights(source, layer, expert, device="cuda"):
    source = str(source or SRC)
    out = []
    for pn in ("gate_proj", "up_proj", "down_proj"):
        k = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{pn}"
        w = _get(source, k + ".weight").to(device)
        s = _get(source, k + ".weight_scale_inv").to(device)
        x = OS.dequantize(w, s, (128, 128))
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite source weights")
        out.append(x)
    if [list(w.shape) for w in out] != [[F, D], [F, D], [D, F]]:
        raise ValueError(f"Flash expert shapes {[list(w.shape) for w in out]}")
    return out


OS.weights = weights

import nq15                                   # noqa: E402,F401  base-K extension over nq_encode / nq_decode
import nq_encode as NE                        # noqa: E402
NE.BASE_K = float(os.environ.get("NQ35_BASE_K", "1.5"))
import nq_layer as NL                         # noqa: E402
NL.FIXED_RULE = dict(NL.FIXED_RULE, n=NFIX,
                     rule=f"top {NFIX} experts per layer by token-weighted REAP (text 0.75 boundary-weighted + vision "
                          "0.25) = always level 4 (default allocation); the rest level 2 until a runtime allocation "
                          "overrides")
