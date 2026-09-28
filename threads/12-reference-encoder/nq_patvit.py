"""Pattern-rate residual trellis (thread 14's patvit/t12pat, vendored so the production encoder does not depend on T14's
working dir). Step i (Viterbi order) shifts in D(i) bits, state_i = ((state_{i-1} << D(i)) | b_i) & 0xFFFF,
value = mul1 LUT, 256-step tail-biting ring (EXL3 two-pass trick). Kernel convention: w_p = KA + ((MASK >> (p%16)) & 1)
at ring position p = (-i) mod 256, so Viterbi step i uses w_{(-i) mod 256}. Ring bits = 16 * K (whole u16 words).
"""
import torch
import harness as h
import nq_decode as D

# T14-verified patterns (kernel masks) + LDLQ drift factors (thread 14 fits)
NEW_PATTERNS = {1.875: (1, 0xFEFE), 1.9375: (1, 0xFFFE), 2.3125: (2, 0x9248)}      # 2.25 = (2, 0x8888) already present
DRIFT = {1.875: 1.02225, 1.9375: 1.0201, 2.25: 1.0135, 2.3125: 1.0124}
_LUT = None


def install():
    for K, v in NEW_PATTERNS.items():
        D.PATTERNS.setdefault(K, v)
    Qm = h._ex()
    for K, d in DRIFT.items():
        Qm.LDLQ_DRIFT.setdefault(K, d)


def is_pat(K):
    """True if exllamav3's CUDA Viterbi cannot do K (only integers and (KA, 0xAAAA) half rates)."""
    K = float(K)
    return not (K.is_integer() or (2 * K).is_integer())


def vsteps(K):
    KA, MASK = D.PATTERNS[float(K)]
    return [KA + ((MASK >> ((-i) % 16)) & 1) for i in range(256)]


@torch.no_grad()
def _run(w, Dst):
    global _LUT
    if _LUT is None:
        _LUT = h.codebook_lut("mul1").float().cuda()
    V = _LUT
    T, L = w.shape
    dev = w.device
    inf = float("inf")
    backs = [None] * L

    def forward(roll, start):
        if start is None:
            cost = torch.zeros(T, 65536, device=dev)
        else:
            cost = torch.full((T, 65536), inf, device=dev)
            cost.scatter_(1, start[:, None], 0.)
        for i in range(L):
            ri = (i + roll) % L
            k = Dst[ri]
            E = 1 << (16 - k)
            mn, top = cost.view(T, 1 << k, E).min(1)
            backs[ri] = top.to(torch.uint8)
            d = (V[None] - w[:, ri, None]).square()
            cost = (d.view(T, E, 1 << k) + mn[:, :, None]).view(T, 65536)
        return cost

    def trace(roll, s, stop_at_zero):
        out = torch.empty((T, L), dtype=torch.int64, device=dev)
        tt = torch.arange(T, device=dev)
        for i in range(L - 1, -1, -1):
            ri = (i + roll) % L
            k = Dst[ri]
            out[:, ri] = s
            e = s >> k
            top = backs[ri][tt, e].long()
            s = (top << (16 - k)) | e
            if stop_at_zero and ri == 0:
                break
        return out, s

    c = forward(L // 2, None)
    _, start = trace(L // 2, c.argmin(1), True)
    forward(0, start)
    idx, _ = trace(0, start, False)
    return V[idx], idx


def step_mask(kmask):
    """kernel ring-position mask -> EXL3 Viterbi-step mask (step i shifts KA + bit(i mod 16))."""
    return sum(1 << j for j in range(16) if (kmask >> ((16 - j) % 16)) & 1)


_EXT, _TMP = None, {}
TMP_TILES = 64                          # scratch: 64 tiles x 256 x 2^(16-KA) shorts = 1 GB at KA 1
EXT_DIR = "/tmp/nestquant/12-reference-encoder/ext"


def ext():
    """exllamav3 quantize_tiles_frac_kernel<KA, MASK> instantiated for the production patterns (csrc/nq_fracvit.cu)."""
    global _EXT
    if _EXT is None:
        import os
        import exllamav3
        import torch.utils.cpp_extension as CE
        from torch.utils.cpp_extension import load
        os.makedirs(EXT_DIR, exist_ok=True)
        os.environ.setdefault("CUDA_HOME", "/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        if CE.CUDA_HOME is None:
            CE.CUDA_HOME = os.environ["CUDA_HOME"]
        nb = "/tmp/nestquant/12-reference-encoder/bin"                      # ninja (static binary symlink)
        if not os.path.exists(os.path.join(nb, "ninja")):                  # /tmp wiped -> the venv's ninja
            nb = "/home/coder/git/glm52/.venv/bin"
        if os.path.isdir(nb) and nb not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nb + ":" + os.environ.get("PATH", "")
        inc = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
        _EXT = load(name="nq_fracvit", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "nq_fracvit.cu")],
                    extra_include_paths=[inc], build_directory=EXT_DIR, extra_cuda_cflags=["-O3", "-lineinfo"],
                    verbose=False)
    return _EXT


def patq_cuda(tiles, K):
    KA, MASK = D.PATTERNS[float(K)]
    tiles = tiles.float().contiguous()
    key = (tiles.device, KA)
    if key not in _TMP:
        e = 65536 >> KA
        _TMP[key] = (torch.zeros((TMP_TILES, 2, e), dtype=torch.half, device=tiles.device),
                     torch.zeros((TMP_TILES, 256, e), dtype=torch.short, device=tiles.device))
    q = torch.zeros_like(tiles); idx = torch.zeros_like(tiles, dtype=torch.short)
    ext().quantize_tiles_frac(tiles, q, idx, *_TMP[key], KA, step_mask(MASK))
    return q, idx.long() & 0xFFFF


def free_tmp():
    _TMP.clear()
    torch.cuda.empty_cache()


USE_CUDA = True


def patq(tiles, K, chunk=128):
    """quantizer(tiles [R,256], K) -> (values, states) in Viterbi order (ExtTileQuantizer signature)."""
    if USE_CUDA:
        return patq_cuda(tiles, K)
    St = vsteps(K)
    q, idx = [], []
    for a in range(0, tiles.shape[0], chunk):
        v, i = _run(tiles[a:a + chunk].float(), St)
        q.append(v); idx.append(i)
    return torch.cat(q), torch.cat(idx)
