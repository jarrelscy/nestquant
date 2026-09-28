"""Thread 23 exact pattern-rate (KA = 2) Viterbi (csrc/nq_frac23.cu): drop-in for T12 nq_patvit.patq_cuda at
K in {2 + popcount/16} (production: down residual K 2.3125) == exllamav3 quantize_tiles_frac_kernel<2, mask>.

    import frac23; q, st = frac23.patq_cuda(rings, 2.3125)     # == PV.patq_cuda(rings, 2.3125)
"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = "/tmp/nestquant/23-encode-throughput/ext_f23"      # own build dir, never shared with k2vit
_EXT = None
_HIST = {}
CALLS = [0, 0]                  # patq_cuda launches, rings (usage proof for gates)
VARIANT = int(os.environ.get("NQ23_FRACVAR", "3"))       # 3 = register-cached decode + hmin2 select


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
        for nb in ("/tmp/nestquant/12-reference-encoder/bin", "/home/coder/git/glm52/.venv/bin"):   # ninja
            if os.path.exists(os.path.join(nb, "ninja")):
                if nb not in os.environ.get("PATH", "").split(":"):
                    os.environ["PATH"] = nb + ":" + os.environ.get("PATH", "")
                break
        inc = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
        _EXT = load(name="nq_frac23", sources=[os.path.join(HERE, "csrc", "nq_frac23.cu")], extra_include_paths=[inc],
                    build_directory=EXT_DIR, extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


def _hist(dev, variant):
    key = (str(dev), variant)
    if key not in _HIST:
        nb = ext().blocks(torch.device(dev).index or 0, variant)
        _HIST[key] = torch.empty((nb, 256, 2048), dtype=torch.short, device=dev)   # 1 MB per resident block
    return _HIST[key]


def supports(K):
    import nq_decode as D
    KA, _ = D.PATTERNS.get(float(K), (None, None))
    return KA == 2 and float(K) in (2.25, 2.3125)      # patterns verified against the reference kernel


def quantize(tiles, step_mask, variant=None):
    """tiles [R,256] fp32 -> (fp32 decoded tiles, int16 states); step_mask in EXL3 Viterbi-step convention."""
    variant = VARIANT if variant is None else variant
    tiles = tiles.float().contiguous()
    assert tiles.dim() == 2 and tiles.shape[1] == 256
    q = torch.zeros_like(tiles); idx = torch.zeros_like(tiles, dtype=torch.short)
    ext().frac23(tiles, idx, q, _hist(tiles.device, variant), int(step_mask), variant)
    return q, idx


def patq_cuda(tiles, K, variant=None):
    """== nq_patvit.patq_cuda(tiles, K) for KA = 2 patterns: (values, states & 0xFFFF as int64)."""
    import nq_decode as D
    import nq_patvit as PV
    KA, MASK = D.PATTERNS[float(K)]
    assert KA == 2
    q, idx = quantize(tiles, PV.step_mask(MASK), variant)
    CALLS[0] += 1; CALLS[1] += tiles.shape[0]
    return q, idx.long() & 0xFFFF


def free():
    _HIST.clear()
