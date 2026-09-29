"""T29 H-only diagnostic (CPU): how much of the LDLQ feedback gain the 128-row ring layout loses, per projection.

  CUDA_VISIBLE_DEVICES= python diag29.py L:E [L:E ...]   -> /tmp/nestquant/29-outlier-gap/diag/L{L}_E{E}.json

For H_d = rot(H + sigma mean(diag) I) (random signs + Had128, as prep), LDLQ with white quantization noise costs
sigma_q^2 tr(D_b) where D_b = block-diagonal of the block-LDL at granularity b.  EXL3 = b 16, nq = b 128 (no feedback
inside a ring's 128 rows), no feedback at all = b k (= tr H).  rho = tr(D128)/tr(D16) predicts nq/EXL3 proxy.
Metrics: the encoder's own metric (lr_H with the PROD lr V, tau .05 r4) and plain H; orderings: identity, act-order
(input channels sorted by diag H ascending, so the largest channels land in the first-processed = highest blocks).
Also the eigen spectrum shares (top 1/4/8/16/64) of H.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "32")
import torch

R = "/tmp/nestquant/nq-encode-v1"
OUT = "/tmp/nestquant/29-outlier-gap/diag"
PROJ = ("gate", "up", "down")


def had128():
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < 128:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / Hm.shape[0] ** 0.5


def rot(H, signs):
    k = H.shape[0]
    Hd = H * signs[:, None] * signs[None, :]
    B = torch.block_diag(*[had128()] * (k // 128))
    return B @ Hd @ B.T


def trD(Hr, b, Ms=()):
    """tr(D_b) and, for each eval metric M (same basis), the white-noise LDLQ cost tr(L^-1 M L^-T) (E = L^-T Z)."""
    C = torch.linalg.cholesky(Hr)
    k = Hr.shape[0]; m = k // b
    blocks = torch.diagonal(C.view(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    out = [float(blocks.square().sum())]
    if Ms:
        Lb = C @ torch.block_diag(*torch.linalg.inv(blocks))          # unit block-lower
        Li = torch.linalg.solve_triangular(Lb, torch.eye(k, dtype=Hr.dtype), upper=False)
        for M in Ms:
            out.append(float(((Li @ M) * Li).sum()))
    return out


def analyse(H, sigma, V=None, perm=None, signs=None, evals=()):
    """evals: [(name, M)] metrics in the original basis (e.g. undamped calibration H, val Gram); the lr plane removes
    the V-span of the error exactly, so M -> (I-P) M (I-P) when V is given."""
    import nq_encode as NE
    H = H.double()
    Ms = [M.double() / M.double().diagonal().mean() for _, M in evals]      # fixed scale across arms
    if V is not None and V.shape[0]:
        Vf = V.double(); P = Vf.T @ torch.linalg.solve(Vf @ Vf.T, Vf); I = torch.eye(H.shape[0], dtype=H.dtype)
        H = NE.lr_H(H, V.to(H.device)).double()
        Ms = [(I - P) @ M @ (I - P) for M in Ms]
    if perm is not None:
        H = H[perm][:, perm]; Ms = [M[perm][:, perm] for M in Ms]
    s = H.diagonal().mean()
    H = H / s
    H = H + sigma * torch.eye(H.shape[0], dtype=H.dtype)
    Hr = rot(H, signs); Mr = [rot(M, signs) for M in Ms]
    r16, r128 = trD(Hr, 16, Mr), trD(Hr, 128, Mr)
    tall = float(Hr.diagonal().sum())
    out = dict(tr16=r16[0], tr128=r128[0], tr_nofb=tall, rho=r128[0] / r16[0], gain16=tall / r16[0], gain128=tall / r128[0])
    for i, (nm, _) in enumerate(evals):
        out[f"rho_{nm}"] = r128[i + 1] / r16[i + 1]
        out[f"c16_{nm}"] = r16[i + 1]; out[f"c128_{nm}"] = r128[i + 1]
    return out


def pivot_order(H, sigma):
    """Greedy pivoted Cholesky on the damped H: channel with the largest conditional variance first (Cholesky order)."""
    Hd = H / H.diagonal().mean() + sigma * torch.eye(H.shape[0], dtype=H.dtype)
    k = Hd.shape[0]
    d = Hd.diagonal().clone(); Lc = torch.zeros(k, k, dtype=H.dtype); order = []
    used = torch.zeros(k, dtype=torch.bool)
    for j in range(k):
        dd = d.clone(); dd[used] = -1
        i = int(dd.argmax()); order.append(i); used[i] = True
        col = (Hd[:, i] - Lc[:, :j] @ Lc[i, :j]) / d[i].sqrt()
        col[used] = 0
        Lc[:, j] = col; Lc[i, j] = d[i].sqrt()
        d = d - col.square()
    return torch.tensor(order)


def orders(H, sigma):
    dg = H.diagonal()
    asc = torch.argsort(dg); desc = asc.flip(0)
    k = H.shape[0]; nb = k // 128
    rr = torch.empty(k, dtype=torch.long)        # rank r -> block r mod nb (big channels spread over blocks)
    rr[(torch.arange(k) % nb) * 128 + torch.arange(k) // nb] = desc
    out = dict(ao=asc, desc=desc, rr=rr)
    if k <= 2048:
        pv = pivot_order(H, sigma)
        out.update(piv=pv, pivr=pv.flip(0))
    return out


def val_grams(L, E):
    """val routed Grams (p^2-weighted): hidden-state x for gate/up, teacher bf16 SwiGLU act for down (as t29_run)."""
    from orbit_duet.source import weights
    tb = [w.bfloat16() for w in weights("/tmp/nestquant/src/glm53-fp8", L, E, device="cpu")]
    cap = torch.load(f"{R}/_stats/eval/val/layer_{L}.pt", weights_only=True, mmap=True)
    r, s = torch.where(cap["ids"] == E)
    p = cap["p"][r, s].double()
    Gx = 0; Ga = 0
    for i in range(0, len(r), 1024):
        x = cap["x"][r[i:i + 1024]].bfloat16()
        a = (torch.nn.functional.silu(x @ tb[0].T) * (x @ tb[1].T)).double() * p[i:i + 1024, None]
        xd = x.double() * p[i:i + 1024, None]
        Gx = Gx + xd.T @ xd; Ga = Ga + a.T @ a
    return [Gx, Gx, Ga]


def main():
    import nq_layer as NL, nq_encode as NE
    torch.manual_seed(0)
    hcap = NL.open_stats(f"{R}/_stats", f"{R}/_stats_mm", 0.25)
    os.makedirs(OUT, exist_ok=True)
    sig = NE.PROD["sigma"]
    for pr in sys.argv[1:]:
        L, E = map(int, pr.split(":"))
        o = f"{OUT}/L{L}_E{E}.json"
        if os.path.exists(o):
            continue
        t0 = time.time()
        HG = hcap.glm_H(L, E, device="cpu")
        Hval = val_grams(L, E)
        res = dict(layer=L, expert=E, proj={})
        for pi, pn in enumerate(PROJ):
            if pi == 1:
                res["proj"]["up"] = res["proj"]["gate"]; continue       # same input Gram
            H = HG["H"][pi].double()
            k = H.shape[0]
            signs = torch.randn(k, generator=torch.Generator().manual_seed(91426)).sign().double()
            w = torch.linalg.eigvalsh(H).flip(0).clamp_min(0)
            sh = (w / w.sum())
            V = NE.lr_detect(H.float(), **NE.PROD["lr"]).float()
            d = dict(eig_share={str(n): float(sh[:n].sum()) for n in (1, 2, 4, 8, 16, 64, 256)},
                     eff_rank=float(w.sum() ** 2 / w.square().sum()), lr_rank=int(V.shape[0]),
                     diag_top_share={str(n): float(H.diagonal().sort(descending=True).values[:n].sum() / H.diagonal().sum()) for n in (1, 4, 16)})
            ev = [("H", H), ("val", Hval[pi])]
            for nm, kw in (("plain", dict()), ("lr", dict(V=V))):
                d[nm] = analyse(H, sig[pn], signs=signs, evals=ev, **kw)
            for on, perm in orders(H, sig[pn]).items():
                d[f"lr_{on}"] = analyse(H, sig[pn], signs=signs, evals=ev, V=V, perm=perm)
            res["proj"][pn] = d
            print(L, E, pn, d["lr_rank"], {k2: [round(v[x], 3) for x in ("rho", "rho_H", "rho_val")] for k2, v in d.items()
                              if isinstance(v, dict) and "rho" in v}, flush=True)
        json.dump(res, open(o, "w"), indent=1)
        print(f"L{L} E{E} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
