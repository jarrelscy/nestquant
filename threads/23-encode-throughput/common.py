"""Thread 23 shared setup: production config, inputs, and the reference (thread 12) single-pass encode.

Production "p4126 single-pass inner0" (threads/12-reference-encoder/nq_dist48.py arm "nq", nq_layer.py default):
    NE.encode_expert(teacher, HG, res_K={gate: 2, up: 2, down: 2.3125}, canonical_base=False, inner=0, lam=0.3)
    base_var "sign", sigma gate/up 0.5 down 1.0, sigma_out 0.03, seed 91426 (NE.PROD defaults)
    HG = nq19_load.Capture(root, stats).glm_H(L, E)  (thread-08 recipe, boundary weight 1 == nq_bnd.glm_H_bnd w=1)
    teacher = orbit_duet.source.weights(SRC, L, E) -> fp32 (cuda)
"""
import os, sys, json, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "16")
T12_LIVE = "/home/coder/git/nestquant/threads/12-reference-encoder"
# NQ23_T12=<dir>: import a pinned copy of thread 12's encoder files (T12's working copy changes under long runs;
# ref and batch arms must import the same code). Default: the live thread-12 directory.
T12 = os.environ.get("NQ23_T12", T12_LIVE)
T19 = "/home/coder/git/nestquant/threads/19-full-capture"
for p in (T12, T19):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch

SCR = "/tmp/nestquant/23-encode-throughput"
SRC = "/tmp/nestquant/src/glm53-fp8"
ROOT = "/tmp/nestquant/19-capture"
STATS = "stats0"
PK = {"gate": 2.0, "up": 2.0, "down": 2.3125}
PROD_KW = dict(res_K=PK, canonical_base=False, inner=0, lam=0.3)
# 9+ bit-identity experts: a spread of layers incl. the 9 canonical (16/49/66 x 36/92/165) + MTP-side layers
CHECK_EXPERTS = [(3, 0), (16, 36), (16, 92), (16, 165), (40, 7), (49, 36), (66, 165), (76, 200), (77, 11), (30, 128)]


def setup(cap_gb=12):
    import nq_patvit as PV                  # build thread 12's CUDA ext in our scratch, never in thread 12's dir
    PV.EXT_DIR = f"{T12}/ext" if T12 != T12_LIVE else f"{SCR}/ext_live"
    torch.cuda.set_per_process_memory_fraction(cap_gb / 80)
    torch.backends.cuda.matmul.allow_tf32 = False


T12_FILES = ("nq_encode.py", "nq_decode.py", "nq_patvit.py", "nq_layer.py", "csrc/nq_fracvit.cu")


def ref_shas(d=None):
    """sha16 of thread 12's encoder files in dir d (default: the imported dir, T12)."""
    d = d or T12
    return {f: hashlib.sha256(open(f"{d}/{f}", "rb").read()).hexdigest()[:16] for f in T12_FILES}


IMPORT_SHAS = ref_shas()     # what this process imports (files read before any T12 module import below)


_CAP = None


def capture():
    global _CAP
    if _CAP is None:
        import nq19_load
        _CAP = nq19_load.Capture(root=ROOT, stats=STATS)
    return _CAP


def teacher(L, E, device="cuda"):
    from orbit_duet.source import weights
    return [w.float() for w in weights(SRC, L, E, device=device)]


def load_HG(L, E, device="cuda"):
    """nq_layer.py: glm_H (bw 1) + the unrouted-expert fallbacks (H=I, G=None) -> (HG, flags)."""
    HG = capture().glm_H(L, E, device=device)
    flags = []
    for i, Hm in enumerate(HG["H"]):
        if not torch.isfinite(Hm).all() or float(Hm.diagonal().mean()) <= 0:
            HG["H"][i] = torch.eye(Hm.shape[0], device=Hm.device); flags.append(f"H{i}=I")
    for i, G in enumerate(HG["G"][:2]):
        if G is not None and not torch.isfinite(G).all():
            HG["G"][i] = None; flags.append(f"G{i}=none")
    return HG, flags


def ref_encode(L, E):
    """Thread 12 production single-pass encode of one expert -> artifact (as nq_layer writes it, minus meta extras)."""
    import nq_encode as NE
    HG, flags = load_HG(L, E)
    art, _ = NE.encode_expert(teacher(L, E), HG, **PROD_KW)
    art["meta"].update(layer=L, expert=E, flags=flags)
    return art


# ------------------------------------------------------------------------------------------------ artifact compare
def flatten(x, pre=""):
    """artifact -> {path: tensor | python value}"""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f"{pre}/{k}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(flatten(v, f"{pre}[{i}]"))
    else:
        out[pre] = x
    return out


VOLATILE = ("/time",)        # wall-clock fields in meta.info are not artifact bytes


def compare(a, b):
    """-> (n_tensors, n_tensor_bytes, list of mismatching paths). Tensors: dtype, shape and raw bytes equal."""
    fa, fb = flatten(a), flatten(b)
    bad = []
    nt = nbytes = 0
    for k in sorted(set(fa) | set(fb)):
        if k.endswith(VOLATILE):
            continue
        if k not in fa or k not in fb:
            bad.append(f"missing {k}"); continue
        x, y = fa[k], fb[k]
        if torch.is_tensor(x) or torch.is_tensor(y):
            if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.dtype != y.dtype or x.shape != y.shape:
                bad.append(f"type/shape {k}"); continue
            nt += 1; nbytes += x.numel() * x.element_size()
            xb = x.detach().cpu().contiguous().view(torch.uint8) if x.numel() else x
            yb = y.detach().cpu().contiguous().view(torch.uint8) if y.numel() else y
            if not torch.equal(xb, yb):
                bad.append(k)
        else:
            if isinstance(x, float) and isinstance(y, float) and (x != x) and (y != y):
                continue
            if x != y:
                bad.append(f"value {k}: {x!r} vs {y!r}")
    return nt, nbytes, bad
