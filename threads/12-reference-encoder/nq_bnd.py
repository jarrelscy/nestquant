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
DEFAULT_BND = 1.0      # user 2026-09-29: no boundary upweighting in calibration (weights only pick fixed_set.json)
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
    for key, v in kv.items():                   # 'think:*=50' / '*:d1=20' wildcards
        k, b = key.split(":")
        for kk in (KINDS if k == "*" else (k,)):
            for bb in (BUCKETS if b == "*" else (b,)):
                assert (kk, bb) in w, f"unknown bucket {key}"
                w[(kk, bb)] = float(v)
    return w


def is_flat_one(w):
    return all(v == 1.0 for v in w.values())


def bnd_tag(w):
    vals = sorted(set(w.values()))
    return f"bnd{vals[0]:g}" if len(vals) == 1 else "bnd" + "_".join(f"{w[(k, b)]:g}" for k in KINDS for b in BUCKETS)


ROUTED_KEYS = ("A2", "A0", "D2", "D0")
CTX_KEYS = ("C_ctx", "Dc")
DEFAULT_K = None          # ESS shrink constant (None = no shrink); set from the held-out A/B (nq_bnd_ab.py)
DEFAULT_CAP = None        # max trace share of the whole boundary increment (None = no cap)


def shrink(n, k):
    """ESS/(ESS+k) (1 if k is None/0)."""
    return 1.0 if not k else n / (n + k)


def glm_H_bnd(cap, L, E, w=None, device="cuda", k=DEFAULT_K, cap_frac=DEFAULT_CAP, cache=None, **kw):
    """Capture.glm_H with boundary-bucket weights w {(kind, bucket): weight}. w all-1 -> identical to cap.glm_H.

    Damping (lead 2026-09-28, thin per-expert think stats): per group the effective weight is
        w_eff = 1 + (w - 1) * n / (n + k)
    with n = the group's routed p-ESS for the routed sums (A2, A0, D2, D0, g[0:4]) and the group's context-row count
    for the layer-wide context sums (C_ctx, Dc, g[4:6]).  Then, if cap_frac, the whole increment Delta is scaled by
    s <= 1 so that tr(Delta_key) / tr(base_key + Delta_key) <= cap_frac for every routed key (one s per expert)."""
    w = parse_bnd(w) if not isinstance(w, dict) or len(w) != len(KINDS) * len(BUCKETS) else w
    if is_flat_one(w):
        HG = cap.glm_H(L, E, device=device, **kw)
        HG["meta"]["bnd"] = "none(w=1)"
        return HG
    if not hasattr(cap, "components_bnd"):
        raise RuntimeError("boundary weights requested but this T19 capture has no components_bnd(); "
                           "pass --bnd 1 for the old behaviour")
    c = cap.components(L, E, device, keys=ROUTED_KEYS + CTX_KEYS)
    delta = {key: None for key in ROUTED_KEYS + CTX_KEYS}
    dg = torch.zeros_like(c["g"], dtype=torch.float64)
    groups = {}
    for (kd, b), wb in w.items():
        if wb == 1.0:
            continue
        if cache is not None and (L, E, kd, b) in cache:          # host-side cache (A/B arms share the sums)
            cb = {x: (y.to(device) if torch.is_tensor(y) else y) for x, y in cache[(L, E, kd, b)].items()}
        else:
            cb = cap.components_bnd(L, E, device, groups=[(kd, b)])[(kd, b)]
            if cache is not None:
                cache[(L, E, kd, b)] = {x: (y.cpu() if torch.is_tensor(y) else y) for x, y in cb.items()}
        nr, nc = float(cb.get("ess", 0.)), float(cb.get("n_ctx_rows", 0))
        wr, wc = 1 + (wb - 1) * shrink(nr, k), 1 + (wb - 1) * shrink(nc, k)
        for key in ROUTED_KEYS + CTX_KEYS:
            if key in cb:
                inc = (wr if key in ROUTED_KEYS else wc) - 1.0
                t = inc * cb[key].double()
                delta[key] = t if delta[key] is None else delta[key] + t
        g = cb["g"].double()
        dg[:4] += (wr - 1) * g[:4]; dg[4:] += (wc - 1) * g[4:]
        groups[f"{kd}:{b}"] = dict(w=wb, ess=nr, n_routed=int(cb.get("n_routed", 0)), n_ctx_rows=int(nc),
                                   w_eff_routed=wr, w_eff_ctx=wc)
        del cb
    s, share = 1.0, {}
    for key in ROUTED_KEYS:
        if delta[key] is not None:
            tb, td = float(c[key].double().trace()), float(delta[key].trace())
            share[key] = td / (tb + td) if tb + td > 0 else 0.
            if cap_frac and share[key] > cap_frac:
                s = min(s, cap_frac * tb / ((1 - cap_frac) * td))
    for key, d in delta.items():
        if d is not None:
            c[key] = (c[key].double() + s * d).float()
    c["g"] = (c["g"].double() + s * dg).to(c["g"].dtype)
    del delta, dg
    HG = cap.glm_H(L, E, device=device, c=c, **kw)
    HG["meta"]["bnd"] = {f"{kd}:{b}": v for (kd, b), v in w.items()}
    HG["meta"]["bnd_damp"] = dict(k=k, cap_frac=cap_frac, scale=s, trace_share_undamped_by_cap=share, groups=groups)
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
