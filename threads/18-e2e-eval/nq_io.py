"""Low-RSS safetensors access (thread 18): header index + pread into a small pinned staging buffer.

No mmap: host RSS stays ~staging size (default 256 MiB) no matter how large the checkpoint is; the
kernel page cache (shared, reclaimable) does the caching across ranks.
"""
import glob
import json
import os
import struct

import torch

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "F8_E4M3": torch.float8_e4m3fn, "I32": torch.int32, "I64": torch.int64,
      "U8": torch.uint8, "I16": torch.int16, "BOOL": torch.bool}
STAGE = int(os.environ.get("NQ_STAGE_MB", "256")) << 20


class SafeIndex:
    def __init__(self, root):
        self.root = root
        self.map = {}
        files = sorted(glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no safetensors under {root}")
        for f in files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.map[k] = (f, v["dtype"], tuple(v["shape"]), base + a, b - a)
        self._fh = {}
        self._stage = None

    def __contains__(self, k):
        return k in self.map

    def names(self):
        return self.map.keys()

    def _file(self, f):
        if f not in self._fh:
            self._fh[f] = os.open(f, os.O_RDONLY)
        return self._fh[f]

    def get(self, name, device="cpu"):
        f, dt, shape, off, nbytes = self.map[name]
        fd = self._file(f)
        out = torch.empty(nbytes, dtype=torch.uint8, device=device)
        if str(device) == "cpu":
            mv = memoryview(out.numpy())
            got = 0
            while got < nbytes:
                got += os.preadv(fd, [mv[got:]], off + got)
        else:
            if self._stage is None:
                self._stage = torch.empty(STAGE, dtype=torch.uint8).pin_memory()
            mv = memoryview(self._stage.numpy())
            done = 0
            while done < nbytes:
                n = min(STAGE, nbytes - done)
                got = 0
                while got < n:
                    got += os.preadv(fd, [mv[got:n]], off + done + got)
                out[done:done + n].copy_(self._stage[:n])   # sync copy: staging reuse is safe
                done += n
        return out.view(DT[dt]).view(shape)


def fp8_dequant(w, s_inv, block=128, dtype=torch.bfloat16):
    """Block-FP8 (e4m3, [ceil(N/128), ceil(K/128)] fp32 scale_inv) -> dtype, same device."""
    N, K = w.shape
    if N % block == 0 and K % block == 0:
        return (w.float().view(N // block, block, K // block, block)
                * s_inv.float()[:, None, :, None]).view(N, K).to(dtype)
    s = s_inv.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (w.float() * s).to(dtype)


class FP8Model:
    """Tensor access for the zai-org/GLM-5.3 FP8 checkpoint; weights come back dequantised bf16."""

    def __init__(self, root):
        self.idx = SafeIndex(root)

    def weight(self, name, device):
        """name without '.weight'; handles FP8 (+ weight_scale_inv) and plain tensors."""
        w = self.idx.get(f"{name}.weight", device)
        si = f"{name}.weight_scale_inv"
        if si in self.idx:
            return fp8_dequant(w, self.idx.get(si, device))
        return w

    def tensor(self, name, device):
        if name.endswith(".weight") and name[:-7] + ".weight_scale_inv" in self.idx:
            return self.weight(name[:-7], device)
        return self.idx.get(name, device)

    def expert(self, layer, e, device):
        p = f"model.layers.{layer}.mlp.experts.{e}"
        return {k: self.weight(f"{p}.{k}", device) for k in ("gate_proj", "up_proj", "down_proj")}
