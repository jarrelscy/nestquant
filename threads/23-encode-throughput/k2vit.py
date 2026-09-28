"""Thread 23 exact K2 mul1 Viterbi (csrc/nq_k2vit.cu): drop-in for exllamav3 quantize_tiles(rings, K=2, mul1) states.

    import k2vit; st = k2vit.states(rings)        # rings [R,256] fp32 -> int64 states & 0xFFFF (== NE.viterbi(rings, 2))
"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = "/tmp/nestquant/23-encode-throughput/ext"
_EXT = None
_HIST = {}


def ext():
    global _EXT
    if _EXT is None:
        import exllamav3
        import torch.utils.cpp_extension as CE
        from torch.utils.cpp_extension import load
        os.makedirs(EXT_DIR, exist_ok=True)
        os.environ.setdefault("CUDA_HOME", "/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        if CE.CUDA_HOME is None:
            CE.CUDA_HOME = os.environ["CUDA_HOME"]
        nb = "/tmp/nestquant/12-reference-encoder/bin"                      # ninja (static binary symlink)
        if os.path.isdir(nb) and nb not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nb + ":" + os.environ.get("PATH", "")
        inc = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
        _EXT = load(name="nq_k2vit", sources=[os.path.join(HERE, "csrc", "nq_k2vit.cu")], extra_include_paths=[inc],
                    build_directory=EXT_DIR, extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


VARIANT = int(os.environ.get("NQ23_K2VAR", "3"))       # 3 = register-cached decode + hmin2 select


def _hist(dev, variant):
    key = (str(dev), variant)
    if key not in _HIST:
        nb = ext().blocks(torch.device(dev).index or 0, variant)
        _HIST[key] = torch.empty((nb, 256, 2048), dtype=torch.short, device=dev)   # 1 MB per resident block
    return _HIST[key]


def states(rings, variant=None):
    variant = VARIANT if variant is None else variant
    rings = rings.float().contiguous()
    out = torch.empty(rings.shape, dtype=torch.short, device=rings.device)
    ext().k2vit(rings, out, None, _hist(rings.device, variant), variant)
    return out.long() & 0xFFFF


def quantize_tiles(tiles, variant=None):
    """== exllamav3 quantize_tiles(tiles, {"K": 2, "mul1": True}) -> (fp32 decoded tiles, int16 indices)."""
    variant = VARIANT if variant is None else variant
    tiles = tiles.contiguous()
    assert tiles.dim() == 2 and tiles.shape[1] == 256 and tiles.dtype == torch.float
    q = torch.zeros_like(tiles); idx = torch.zeros_like(tiles, dtype=torch.short)
    ext().k2vit(tiles, idx, q, _hist(tiles.device, variant), variant)
    return q, idx


def free():
    _HIST.clear()
