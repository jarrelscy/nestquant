"""Thread 19: shared pieces of the GLM-5.3 full-model calibration capture.

Paths, the FP8-source adapter (duck-typed to orbit-duet's glm_reference.Source so its Layer class is reused
unchanged), triangle packing, and the Hessian recipe reconstruction.  See FORMAT.md.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NQ = os.path.dirname(os.path.dirname(HERE))
ORBIT = "/home/coder/git/orbit-duet"
for p in (ORBIT, f"{NQ}/threads/18-e2e-eval"):
    if p not in sys.path:
        sys.path.append(p)

SRC = os.environ.get("NQ19_SRC", "/tmp/nestquant/src/glm53-fp8")
OUT = os.environ.get("NQ19_OUT", "/tmp/nestquant/19-capture")
CORPUS = f"{ORBIT}/runs/glm53_training_15m_v2"
MATCHED = f"{ORBIT}/runs/glm53_matched_context_pilot_v1"
CTX_SEED = 20260925            # orbit_duet.glm_calibration.CalibrationBatches context seed
CTX_FRACTION = 4               # context rows = T_fit // 4  (pilot: 65536 -> 16384)
TAIL_FIRST_WINDOW = 28784      # thread 18 nq-tail = last 262144 tokens; fits must stay below
VAL_WINDOWS_RESERVED = 128     # windows 28656..28783 = held-out val rows; fits use windows < 28656
SHARD_WINDOWS = 2048           # progressive shard k = fit windows [2048 k, min(2048 (k + 1), 28656))
D, F = 6144, 2048
NEXP = 256


def gpu_cap():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("NQ19_GPU_GB", "12")) / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Src:
    """FP8 GLM-5.3 checkpoint with the same `.config/.weight/.tensor/.embed` semantics as
    orbit-duet benchmarks.ood.glm_reference.Source (weights: FP8 * block scale in fp32, then cast)."""

    def __init__(self, root=SRC):
        from nq_io import SafeIndex
        self.root = root
        self.idx = SafeIndex(root)
        self.config = json.load(open(f"{root}/config.json"))
        self.block = tuple(self.config["quantization_config"]["weight_block_size"])
        self._embed = None

    def tensor(self, key, device="cpu"):
        return self.idx.get(key, device)

    def weight(self, key, device="cuda", dtype=torch.bfloat16):
        from orbit_duet.source import dequantize
        v = self.tensor(key, device)
        if v.dtype == torch.float8_e4m3fn:
            s = self.tensor(key.removesuffix("weight") + "weight_scale_inv", device)
            v = dequantize(v, s, self.block)
        elif v.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"unexpected dtype {v.dtype} for {key}")
        return v.to(dtype)

    def embed(self, tokens):
        if self._embed is None:
            self._embed = self.tensor("model.embed_tokens.weight", "cpu")
        return self._embed[tokens.cpu()].to(tokens.device, dtype=torch.bfloat16)



PROJ = ("gate_proj", "up_proj", "down_proj")


class ExpertCache:
    """All 256 routed experts of one layer as raw FP8 + fp32 scale_inv in ONE pinned host buffer
    (~9.7 GB, allocated once, refilled per layer with 16 pread threads)."""

    def __init__(self, src, layer=None):
        self.src = src
        self.keys = []
        self.nbytes = 0
        ref = self._entries(3)
        self.size = sum(b for (_, _, _, _, b) in ref)
        self.buf = torch.empty(self.size, dtype=torch.uint8, pin_memory=True)
        self.layer = None
        if layer is not None:
            self.load(layer)

    def _entries(self, layer):
        out = []
        for e in range(NEXP):
            for p in PROJ:
                for suf in (".weight", ".weight_scale_inv"):
                    k = f"model.layers.{layer}.mlp.experts.{e}.{p}{suf}"
                    f, dt, shape, off, nb = self.src.idx.map[k]
                    out.append((k, f, (dt, shape), off, nb))
        return [(k, f, m, off, nb) for (k, f, m, off, nb) in out]

    def load(self, layer):
        from concurrent.futures import ThreadPoolExecutor
        ents = self._entries(layer)
        if sum(e[4] for e in ents) != self.size:
            raise ValueError("expert byte size differs across layers")
        self.views = {}
        pos = 0
        jobs = []
        for (k, f, (dt, shape), off, nb) in ents:
            self.views[k] = (pos, nb, dt, shape)
            jobs.append((f, off, pos, nb))
            pos += nb
        mv = memoryview(self.buf.numpy())

        def rd(j):
            f, off, pos, nb = j
            fd = os.open(f, os.O_RDONLY)
            try:
                got = 0
                while got < nb:
                    got += os.preadv(fd, [mv[pos + got:pos + nb]], off + got)
            finally:
                os.close(fd)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(rd, jobs))
        self.layer = layer

    def raw(self, e, p, suf, device="cuda"):
        from nq_io import DT
        pos, nb, dt, shape = self.views[f"model.layers.{self.layer}.mlp.experts.{e}.{p}{suf}"]
        return self.buf[pos:pos + nb].to(device, non_blocking=False).view(DT[dt]).view(shape)

    def expert(self, e, device="cuda", dtype=torch.bfloat16):
        """[gate, up, down] dequantised exactly like glm_reference.Source.weight (fp32 product, then cast)."""
        return [dequant(self.raw(e, p, ".weight", device), self.raw(e, p, ".weight_scale_inv", device),
                        self.src.block).to(dtype) for p in PROJ]


def dequant(w, s, block=(128, 128)):
    from orbit_duet.source import dequantize
    return dequantize(w, s, block)          # fp32


# ------------------------------------------------------------------------------------------------ packing
_TRIU = {}


def triu(n, device):
    k = (n, str(device))
    if k not in _TRIU:
        _TRIU[k] = torch.triu_indices(n, n, device=device)
    return _TRIU[k]


def pack(A):
    """Row-major upper triangle (i <= j) of a symmetric [n, n] matrix -> [n(n+1)/2]."""
    i, j = triu(A.shape[0], A.device)
    return A[i, j]


def unpack(v, n):
    i, j = triu(n, v.device)
    A = torch.zeros(n, n, dtype=v.dtype, device=v.device)
    A[i, j] = v
    A = A + A.T
    A.diagonal().mul_(0.5)
    return A


def npk(n):
    return n * (n + 1) // 2


# ------------------------------------------------------------------------------------------------ recipe
def nt(G):
    return G / G.diagonal().mean()


def recipe_H(routed_w, routed_u, ctx, cp2, alpha=0.25):
    """Thread-08 selected recipe: alpha*nt(W) + (1-alpha)*nt(U) with
    W = routed_w + cp2*ctx  (p-weighted routed + context at weight cp),  U = routed_u + ctx (unweighted)."""
    W = routed_w.double() + cp2 * ctx.double()
    U = routed_u.double() + ctx.double()
    return (alpha * nt(W) + (1 - alpha) * nt(U)).float()
