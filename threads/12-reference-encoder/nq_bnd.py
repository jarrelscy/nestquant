"""Boundary-weighted calibration (user requirement 2026-09-28) for the NestQuant encoder + the eval boundary split.

Window: the 32 tokens before (a) a non-empty </think>  ("think")  or  (b) an end of answer <|im_end|>  ("end").
Buckets by distance d to the boundary token (d = 1: the next token is the boundary): d1, d2_4, d5_16, d17_32.
T19 stores, per bucket, additive sums of the same form as the plain capture components (the rows of the bucket only):
    A2 (routed p^2 x x^T), A0 (routed x x^T), D2, D0 (down inputs), C_ctx / Dc (context rows, optional), g (gdiag rows, optional)
Weighting = replace every row weight 1 by w_b inside the window, i.e. component += (w_b - 1) * bucket_component, applied to
BOTH the routed (p-weighted) and uniform grams BEFORE trace normalisation, and to the salience (G) sums:
    H = 0.25 nt(W + sum_b (w_b-1) W_b) + 0.75 nt(U + sum_b (w_b-1) U_b)   (+ the usual sigma damping in the encoder)
All weights 1 reproduces Capture.glm_H exactly (the boundary sums are not even loaded).

Interface expected from T19 (either works):
    cap.components_bnd(L, E, device) -> {(kind, bucket): {"A2","A0","D2","D0"[, "C_ctx","Dc","g"]}}
Eval capture: per-row int tensors "bnd_think" and "bnd_end" = distance d in 1..32 to the next such boundary (0 = none).
"""
import torch

KINDS = ("think", "end")
BUCKETS = ("d1", "d2_4", "d5_16", "d17_32")
BUCKET_RANGE = {"d1": (1, 1), "d2_4": (2, 4), "d5_16": (5, 16), "d17_32": (17, 32)}
DEFAULT_BND = 50.0
ADD_KEYS = ("A2", "A0", "D2", "D0", "C_ctx", "Dc", "g")


def parse_bnd(spec=None):
    """'50' -> flat 50 on every (kind, bucket); '1' -> old behaviour;
    'think:d1=50,end:d17_32=4,*=10' -> per bucket ('*' = default for the rest, else DEFAULT_BND)."""
    if spec is None:
        spec = str(DEFAULT_BND)
    if isinstance(spec, dict):
        return {(k, b): float(spec.get((k, b), DEFAULT_BND)) for k in KINDS for b in BUCKETS}
    spec = str(spec)
    if "=" not in spec:
        return {(k, b): float(spec) for k in KINDS for b in BUCKETS}
    kv = dict(x.split("=") for x in spec.split(","))
    dflt = float(kv.pop("*", DEFAULT_BND))
    w = {(k, b): dflt for k in KINDS for b in BUCKETS}
    for key, v in kv.items():
        k, b = key.split(":")
        for kk in (KINDS if k == "*" else (k,)):
            assert (kk, b) in w, f"unknown bucket {key}"
            w[(kk, b)] = float(v)
    return w


def is_flat_one(w):
    return all(v == 1.0 for v in w.values())


def bnd_tag(w):
    vals = sorted(set(w.values()))
    return f"bnd{vals[0]:g}" if len(vals) == 1 else "bnd" + "_".join(f"{w[(k, b)]:g}" for k in KINDS for b in BUCKETS)


def glm_H_bnd(cap, L, E, w=None, device="cuda", **kw):
    """Capture.glm_H with boundary-bucket weights w {(kind, bucket): weight}. w all-1 -> identical to cap.glm_H."""
    w = parse_bnd(w) if not isinstance(w, dict) or len(w) != len(KINDS) * len(BUCKETS) else w
    if is_flat_one(w):
        HG = cap.glm_H(L, E, device=device, **kw)
        HG["meta"]["bnd"] = "none(w=1)"
        return HG
    if not hasattr(cap, "components_bnd"):
        raise RuntimeError("boundary weights requested but this T19 capture has no components_bnd(); "
                           "pass --bnd 1 for the old behaviour")
    c = cap.components(L, E, device, keys=("A2", "A0", "D2", "D0", "Dc", "C_ctx"))
    bnd = cap.components_bnd(L, E, device)
    used = {}
    for (k, b), wb in w.items():
        if wb == 1.0 or (k, b) not in bnd:
            continue
        for key in ADD_KEYS:
            if key in bnd[(k, b)] and key in c:
                c[key] = c[key].double() + (wb - 1.0) * bnd[(k, b)][key].double()
                used.setdefault(f"{k}:{b}", []).append(key)
    for key in ADD_KEYS:
        if key in c and torch.is_tensor(c[key]) and key != "g":
            c[key] = c[key].float()
    HG = cap.glm_H(L, E, device=device, c=c, **kw)
    HG["meta"]["bnd"] = {f"{k}:{b}": v for (k, b), v in w.items()}
    HG["meta"]["bnd_used"] = used
    return HG


# ---------------------------------------------------------------------------------------------- eval boundary split
SPLITS = [(k, lim) for k in KINDS for lim in (1, 32)]          # (a)/(b) at d = 1 and d <= 32


def boundary_rows(cap):
    """{f"{kind}/d{1|<=32}": row indices} from the eval capture's per-row distance labels (empty dict if absent)."""
    out = {}
    for k in KINDS:
        key = f"bnd_{k}"
        if key not in cap:
            continue
        d = torch.as_tensor(cap[key]).long()
        for lim in (1, 32):
            m = (d >= 1) & (d <= lim)
            out[f"{k}/d{'=1' if lim == 1 else '<=32'}"] = m.nonzero().flatten()
    return out


@torch.no_grad()
def evaluate_boundary(data, methods):
    """harness-style rel-L2 on held-out boundary rows: {split: {"forced": ..., "routed": ...}} (same _errors as evaluate)."""
    import harness as h
    cap = data.capture
    rows_by = boundary_rows(cap)
    if not rows_by:
        return {}
    routed, slots = torch.where(cap["ids"] == data.expert)
    out = {}
    for name, rows in rows_by.items():
        take = torch.isin(routed, rows); actual = routed[take]; prob = cap["p"][actual, slots[take]]
        out[f"bnd:{name}"] = dict(forced=h._errors(cap, data.teacher, methods, rows, torch.ones(len(rows))),
                                  routed=h._errors(cap, data.teacher, methods, actual, prob))
    return out
