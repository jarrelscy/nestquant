"""Candidate-quantiser plug-ins for nq_e2e.py (thread 18).

Every candidate is one object with this interface:

    class Quantiser:
        name: str
        def begin_layer(self, layer: int, device) -> None          # optional; load this layer's artefacts
        def expert(self, layer: int, expert: int, ref) -> dict | None
            # ref() -> {"gate_proj": [2048,6144], "up_proj": [2048,6144], "down_proj": [6144,2048]}
            #          bf16 on device = FP8 reference dequant (lazy; only read if you call it)
            # return the same keys (any float dtype, [out, in], ORIGINAL un-rotated basis, on device)
            # or None -> the reference expert is used (counted as "fallback" in the report)
        def end_layer(self, layer: int) -> None                    # optional; free memory

Spec strings (``--cand NAME=SPEC``):
    ref                               identity (sanity: KLD must be exactly 0)
    rtn:bits=4,group=128              round-to-nearest asym min/max per (row, group) of the FP8 reference
    dir:/path                         dequantised safetensors: any *.safetensors under /path with keys
                                      model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight  or
                                      layer{L}.expert{E}.{gate,up,down}_proj ; missing -> reference
    nestquant:root=/path,level=4      thread-12 artefacts {root}/layer_{L:03d}/expert_{E:03d}.pt
                                      decoded with threads/12-reference-encoder/nq_decode.decode_expert
    exl3:root=/path,bits=4            {root}/layer_{L:03d}/expert_{E:03d}/expert_{bits}.bin (orbit-duet
                                      legacy .bin; decoded by exllamav3 via orbit_duet.exl3_adapter)
    nvfp4:root=/path                  {root}/layer_{L:03d}/expert_{E:03d}/weights.pt (orbit-duet ModelOpt
                                      payload; decoded by orbit_duet.nvfp4_reference.decode)
    py:/file.py:Class[:k=v,...]       any external class implementing the interface
Optional ``layers=a-b`` in any spec restricts quantisation to those layers (others -> reference).
"""
import importlib
import importlib.util
import json
import os
import sys

import torch

PROJ = ("gate_proj", "up_proj", "down_proj")
NQ12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
ORBIT = "/home/coder/git/orbit-duet"


def _kv(s):
    out = {}
    for part in filter(None, s.split(",")):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def _layers(kv):
    if "layers" not in kv:
        return None
    a, _, b = kv.pop("layers").partition("-")
    return set(range(int(a), int(b or a) + 1))


class Base:
    name = "?"
    only_layers = None

    def begin_layer(self, layer, device):
        self.dev = device

    def end_layer(self, layer):
        pass

    def active(self, layer):
        return self.only_layers is None or layer in self.only_layers


class Ref(Base):
    is_ref = True

    def expert(self, layer, expert, ref):
        return None


class RTN(Base):
    """Asymmetric min/max RTN per (row, group-of-inputs). Smoke-test quantiser only."""

    def __init__(self, bits=4, group=128):
        self.bits, self.group = int(bits), int(group)

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        out = {}
        for p, w in ref().items():
            n, k = w.shape
            g = w.float().view(n, k // self.group, self.group)
            lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
            qmax = 2 ** self.bits - 1
            s = (hi - lo).clamp_min(1e-12) / qmax
            q = ((g - lo) / s).round().clamp(0, qmax)
            out[p] = (q * s + lo).view(n, k).to(torch.bfloat16)
        return out


class Dir(Base):
    """Directory of dequantised safetensors (any sharding; header-indexed, pread, no mmap)."""

    def __init__(self, path):
        from nq_io import SafeIndex
        self.idx = SafeIndex(path)

    def _key(self, layer, expert, p):
        for k in (f"model.layers.{layer}.mlp.experts.{expert}.{p}.weight",
                  f"layer{layer}.expert{expert}.{p}"):
            if k in self.idx:
                return k
        return None

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        keys = [self._key(layer, expert, p) for p in PROJ]
        if any(k is None for k in keys):
            return None
        return {p: self.idx.get(k, self.dev) for p, k in zip(PROJ, keys)}


class NestQuant(Base):
    """Rotated-basis reconstructions are shared across levels: with nq2 and nq4 streams in one pass the
    expensive nq_decode.rotated_levels runs once per expert (0.37 s), each extra level costs ~0.02 s."""
    _shared = {}                       # class-level: {"key": (file, dev), "art": ..., "rot": {proj: rot}}

    def __init__(self, root, level=4):
        self.root, self.level = root, int(level)
        if NQ12 not in sys.path:
            sys.path.insert(0, NQ12)
        import nq_decode
        self.nqd = nq_decode

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        f = f"{self.root}/layer_{layer:03d}/expert_{expert:03d}.pt"
        if not os.path.exists(f):
            return None
        c = NestQuant._shared
        if c.get("key") != (f, str(self.dev)):
            c.clear()
            art = torch.load(f, map_location="cpu", weights_only=False)
            c.update(key=(f, str(self.dev)), art=art,
                     rot={p: self.nqd.rotated_levels(art[p], self.dev) for p in ("gate", "up", "down")})
        art, rot = c["art"], c["rot"]
        W = [self.nqd.decode_matrix(art[p], self.level, self.dev, rot=rot[p]) for p in ("gate", "up", "down")]
        perm = art.get("meta", {}).get("inter_perm")          # same un-permute as nq_decode.decode_expert
        if perm is not None:
            inv = torch.argsort(torch.as_tensor(perm, device=self.dev))
            W = [W[0][inv], W[1][inv], W[2][:, inv]]
        return {"gate_proj": W[0], "up_proj": W[1], "down_proj": W[2]}

    def end_layer(self, layer):
        NestQuant._shared.clear()


class EXL3(Base):
    def __init__(self, root, bits=4):
        self.root, self.bits = root, int(bits)
        if ORBIT not in sys.path:
            sys.path.insert(0, ORBIT)
        # exllamav3_ext needs libcudart.so.12 (thread 05 ships one)
        import ctypes
        lib = "/home/coder/git/nestquant/threads/05-exl3-harness/lib/libcudart.so.12"
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        from orbit_duet.exl3_adapter import EXL3Expert
        self.cls = EXL3Expert

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        f = f"{self.root}/layer_{layer:03d}/expert_{expert:03d}/expert_{self.bits}.bin"
        if not os.path.exists(f):
            return None
        g, u, d = self.cls(f).decoded_weights()
        return {"gate_proj": g, "up_proj": u, "down_proj": d}


class NVFP4(Base):
    def __init__(self, root):
        self.root = root
        if ORBIT not in sys.path:
            sys.path.insert(0, ORBIT)
        from orbit_duet.nvfp4_reference import decode
        self.decode = decode

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        f = f"{self.root}/layer_{layer:03d}/expert_{expert:03d}/weights.pt"
        if not os.path.exists(f):
            return None
        pl = torch.load(f, weights_only=True, map_location="cpu")
        ws = [self.decode({k: v.to(self.dev) for k, v in q.items()}) for q in pl["weights"]]
        return dict(zip(PROJ, ws))


def make(name, spec):
    kind, _, rest = spec.partition(":")
    if kind == "py":
        path, _, tail = rest.partition(":")
        cls, _, args = tail.partition(":")
        kv = _kv(args)
        only = _layers(kv)
        if path.endswith(".py"):
            sp = importlib.util.spec_from_file_location(f"nqplug_{name}", path)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
        else:
            mod = importlib.import_module(path)
        q = getattr(mod, cls)(**kv)
    else:
        kv = _kv(rest) if kind != "dir" else {}
        if kind == "dir":
            path, _, tail = rest.partition(",")
            kv = _kv(tail)
        only = _layers(kv)
        if kind == "ref":
            q = Ref()
        elif kind == "rtn":
            q = RTN(**kv)
        elif kind == "dir":
            q = Dir(path)
        elif kind == "nestquant":
            q = NestQuant(**kv)
        elif kind == "exl3":
            q = EXL3(**kv)
        elif kind == "nvfp4":
            q = NVFP4(**kv)
        else:
            raise SystemExit(f"unknown quantiser kind {kind!r}")
    q.name = name
    q.spec = spec
    q.only_layers = only
    for m in ("begin_layer", "end_layer"):
        if not hasattr(q, m):
            setattr(q, m, (lambda *a, **k: None))
    return q
