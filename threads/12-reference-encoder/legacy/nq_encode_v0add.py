"""NestQuant v0 reference encoder (thread 12): one dynamic 2/3/4-bit artifact per expert, kernel layout of nq_decode.

Per projection (EXL3 layout W (k=in, n=out)):
  1. EXL3 preprocessing, mirrored from harness.quantize_exl3_like: H/count + sigma*mean(diag), su signs +
     blockwise Had128 on H, sv signs, block_rms regularisation (skew rule), input/output Had128, g_scale at K=2.
  2. Block LDL of the rotated H at the kernel's k-chunk size (128; a ring spans a whole chunk, so LDLQ feedback runs
     between chunks, not inside one). Optional output metric G (gate/up): prepare_H_out(G, sv), block 16 = strip.
  3. Dual-state LDLQ over units (128 k x 16 n): two-sided -> anti-diagonals from the far corner (ldlq_2hess order),
     one-sided -> one chunk row at a time. Two feedback states F2 (from E2 = W - Q2) and F4 (E4 = W - Q4):
        T2 = W + F2, T4 = W + F4, base target (1-lam) T2 + lam T4 -> K_b-bit mul1 Viterbi -> Q2
        residual r = T4 - Q2 -> r / s0 -> K_r-bit Viterbi (K_r = 2, or 3 on K3 units) per candidate codebook
        delta = LS fit <g, Din r Dout> / <g, Din g Dout> per unit (or per ring), rounded to fp16; Q4 = Q2 + delta*g
        the candidate with the lowest local cost tr(eta^T Din eta Dout) wins (eta = r - delta g)
  4. (Level 3 dropped by user decision 2026-09-28: the artifact serves levels 2 and 4 only.)
  5. Per-level su/sv refit (exllamav3 refit_scales, un-rotated H), fp16 decode per level; planes packed per shard.
The internal dense reconstructions are returned so nq_decode can be checked bit-exactly against them.
"""
import os, sys, math, time
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq_decode as D

H05 = D.H05
import harness as h

CB_RMS = 1.24371088
INNER_BETA = float(os.environ.get("NQ_INNER_BETA", "1.0"))    # damping of the within-chunk feedback iteration


def _Qm():
    return h._ex()


def viterbi(rings, K, cb):
    """rings [R, 256] fp32 (EXL3 step order) -> states [R, 256] int64 (EXL3 order)."""
    qa = {"K": K}
    if cb == "mul1": qa["mul1"] = True
    elif cb == "mcg": qa["mcg"] = True
    _, st = _Qm().quantize_tiles(rings.contiguous(), qa)
    return st.long() & 0xFFFF


# per-ring base variants (thread 17 sg4 etc.): table in nq_decode.BASE_VARIANTS


def base_quant(R, tgt, K, var=None):
    """Base code of units tgt [T,128,16]: mul1 Viterbi per ring, optionally best-of per-ring scale variants
    (plain ring MSE, as thread 17). -> (values [T,128,16], kernel symbols [T,8,256] uint8, variant ids [T,8] uint8)."""
    rings = R.to_rings(tgt)
    T = tgt.shape[0]
    if var is None:
        v, sy = R.from_states(viterbi(rings, K, "mul1"), K, "mul1")
        return v, sy.to(torch.uint8), torch.zeros(T, 8, dtype=torch.uint8, device=tgt.device)
    tab = D.variant_table(var, tgt.device)
    best = None
    for vi in range(len(tab)):
        a = tab[vi]
        st = viterbi(rings / a, K, "mul1")
        q = D.lut("mul1", tgt.device)[st] * a
        m = (q - rings).square().sum(1)
        if best is None:
            best, bst, bsel = m, st, torch.zeros_like(m, dtype=torch.long)
        else:
            w = m < best
            best = torch.where(w, m, best); bst = torch.where(w[:, None], st, bst); bsel = torch.where(w, vi, bsel)
    v, sy = R.from_states(bst, K, "mul1", scale=tab[bsel])
    return v, sy.to(torch.uint8), bsel.view(T, 8).to(torch.uint8)


class Ring:
    """Unit <-> ring gathers. Kernel ring position p is EXL3 Viterbi step i = 255 - p."""
    def __init__(self, dev):
        self.RI = D.ring_index(dev)                         # [8, 256] kernel order
        self.VO = self.RI.flip(1).contiguous()              # [8, 256] Viterbi order
        self.vo = self.VO.flatten()

    def to_rings(self, X):                                   # X [T, 128, 16] -> [T*8, 256] Viterbi order
        return X.reshape(X.shape[0], 2048)[:, self.vo].reshape(-1, 256)

    def from_states(self, st, K, cb, scale=None):
        """states [T*8, 256] (Viterbi order) -> (values [T,128,16], kernel-order symbols [T, 8, 256])."""
        T = st.shape[0] // 8
        vals = D.lut(cb, st.device)[st]
        if scale is not None:
            vals = vals * scale.view(-1, 1)
        out = torch.empty(T, 2048, device=st.device)
        out[:, self.vo] = vals.view(T, 2048)
        sym = (st & ((1 << K) - 1)).flip(1).view(T, 8, 256)
        return out.view(T, 128, 16), sym


def ldl_blocks(Hr, b, sigma):
    """H = L' D L'^T with unit block-lower L' (identity diagonal blocks) and block-diagonal D (block b)."""
    n = Hr.shape[0]; m = n // b
    Hc = Hr.clone()
    for attempt in range(11):
        try:
            C = torch.linalg.cholesky(Hc); break
        except torch._C._LinAlgError:
            Hc.diagonal().add_(2.0 * sigma * Hc.diagonal().mean())
    DL = torch.diagonal(C.view(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1).contiguous()     # [m, b, b]
    Dm = DL @ DL.transpose(1, 2)
    DLi = torch.linalg.inv(DL)
    L = torch.empty_like(C)
    for i in range(m):
        L[:, i * b:(i + 1) * b] = C[:, i * b:(i + 1) * b] @ DLi[i]
    del C, Hc
    return L, Dm


@torch.no_grad()
def prep(W, H, count, sigma, seed=91426, G=None, sigma_out=0.03, dev="cuda"):
    """Mirror of harness.quantize_exl3_like preprocessing (same RNG order, same numerics)."""
    Qm = _Qm()
    weight = W.to(dev, torch.float32).T.contiguous()
    k, n = weight.shape
    torch.manual_seed(seed)
    Hm = H.to(dev, torch.float32).clone() / count
    dm = torch.diag(Hm).mean().item()
    Hm.diagonal().add_(sigma * dm)
    H_diag = Hm.diagonal().clone()
    su = (torch.randn(k, device=dev).sign() + 1e-5).sign().float().unsqueeze(1)
    su_signs = su.clone()
    Hm *= su.T; Qm.blockwise_preapply_had_r_(Hm, 128); Hm *= su; Qm.blockwise_preapply_had_l_(Hm, 128)
    sv = (torch.randn(n, device=dev).sign() + 1e-5).sign().float().unsqueeze(0)
    weight_orig = weight.clone()
    d = torch.sort(H_diag.sqrt(), descending=True).values
    skew = (d[:k // 50].sum() / d.sum()).item()
    aos = skew < 0.15
    ocs = Qm.block_rms(weight, dim=0, keepdim=True); ocs /= ocs.mean().item()
    zero = ocs.abs() < 1e-30
    if aos:
        ocs[zero] = 0.1
        sv = (sv * ocs + 1e-10).float()
    weight /= sv
    sv[zero] = 0.0
    Qm.blockwise_preapply_had_r_(weight, 128)
    ics = Qm.block_rms(weight, dim=1, keepdim=True)
    ics[ics.abs() < 1e-30] = 0.1
    su = (su * ics / (-Qm.codebook_scale) + 1e-10).float()
    weight /= su
    Qm.blockwise_preapply_had_l_(weight, 128)
    q = h.ExtTileQuantizer("mul1")
    samp = Qm.sample_scale_tiles(weight, 3)
    gs, _ = h._g_scale_search(samp * Qm.ldlq_drift(2), 2, q)
    gsr = {K: h._g_scale_search(samp * Qm.ldlq_drift(K), K, q)[0] for K in (2, 3)}   # residual input scale per K
    weight *= gs
    su /= gs
    Lk, Din = ldl_blocks(Hm, 128, sigma)
    # inner block-16 LDL of every 128x128 D_a, for the within-chunk feedback iteration: U_a = L16'^T - I
    Uin = torch.stack([ldl_blocks(Din[i], 16, sigma)[0].T - torch.eye(128, device=dev) for i in range(Din.shape[0])])
    P = dict(weight=weight, weight_orig=weight_orig, su=su, sv=sv, su_signs=su_signs, Hr=Hm, Lk=Lk, Din=Din,
             Uin=Uin, k=k, n=n, gs=gs, gsr=gsr, aos=aos, skew=skew, sigma=sigma)
    if G is not None:
        Lo, Ho = Qm.prepare_H_out(G.to(dev).float(), sv, {"sigma_reg_out": sigma_out, "sigma_reg": sigma_out}, False, dev)
        Ho = Ho.to(dev)
        Ln, Dout = ldl_blocks(Ho, 16, sigma_out)
        P.update(Ln=Ln, Dout=Dout, Ho=Ho)
        del Lo
    return P


@torch.no_grad()
def encode_rotated(P, lam=0.3, base_K=None, k3=None, cands=(("mul1",),), delta_per="unit", gain=1.0,
                   shard_axis="n", inner=0, base_var=None):
    """Dual-state LDLQ. base_K: per-chunk list (default 2). k3: bool [tk, tn] units with a 3-bit residual.
    cands: residual codebook candidates, each (cb,) or (cb, -1) for a sign-flipped Viterbi input.
    Returns dict with Q2, Q4 [k, n], symbols, deltas, cb ids, per-unit costs."""
    dev = P["weight"].device
    Wt = P["weight"]; k, n = Wt.shape; tk, tn = k // 128, n // 16
    Lk, Din = P["Lk"], P["Din"]
    two = "Ln" in P
    Ln = P.get("Ln"); Dout = P.get("Dout")
    R = Ring(dev)
    base_K = base_K or [2] * tk
    k3 = k3 if k3 is not None else torch.zeros(tk, tn, dtype=torch.bool, device=dev)
    k3 = k3.to(dev)
    W4 = Wt.view(tk, 128, tn, 16)
    M = torch.zeros(2, k, n, device=dev)                      # E_done @ Ln' for the two states
    Q2 = torch.zeros(tk, tn, 128, 16, device=dev); Q4 = torch.zeros_like(Q2)
    sym2 = torch.zeros(tk, tn, 8, 256, dtype=torch.uint8, device=dev)
    sym4 = torch.zeros(tk, tn, 8, 256, dtype=torch.uint8, device=dev)
    nd = 8 if delta_per == "ring" else 1
    delta = torch.zeros(tk, tn, nd, dtype=torch.float16, device=dev)
    cbid = torch.zeros(tk, tn, dtype=torch.uint8, device=dev)
    bvar = torch.zeros(tk, tn, 8, dtype=torch.uint8, device=dev)
    cost2 = torch.zeros(tk, tn, device=dev); cost4 = torch.zeros(tk, tn, device=dev)
    ar128 = torch.arange(128, device=dev); ar16 = torch.arange(16, device=dev)
    Lkt = Lk.T.contiguous()                                    # Lkt[r, :] = Lk[:, r]
    if two:
        steps = [(torch.arange(max(0, s - (tn - 1)), min(tk - 1, s) + 1, device=dev), None, s)
                 for s in range(tk + tn - 2, -1, -1)]
    else:
        steps = [(torch.full((tn,), a, device=dev, dtype=torch.long), torch.arange(tn, device=dev), None)
                 for a in range(tk - 1, -1, -1)]
    stream = _Qm().get_quant_stream(dev)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for a_idx, c_idx, s in steps:
            if c_idx is None:
                c_idx = s - a_idx
            T = len(a_idx)
            rows = (a_idx.unsqueeze(1) * 128 + ar128)                 # [T, 128]
            cols = (c_idx.unsqueeze(1) * 16 + ar16)                   # [T, 16]
            Wu = W4[a_idx, :, c_idx, :]                               # [T, 128, 16]
            # feedback F[unit] = Lk'[:, rows]^T @ M[:, cols]  (M holds done errors times Ln')
            if two:
                Lsel = Lkt[rows.flatten()].view(T, 128, k)
                Ms = M[:, :, cols.flatten()].view(2, k, T, 16).permute(0, 2, 1, 3)       # [2, T, k, 16]
                F2 = torch.bmm(Lsel, Ms[0]); F4 = torch.bmm(Lsel, Ms[1])
            else:
                Lsel = Lkt[rows[0]]                                    # [128, k]
                F2 = (Lsel @ M[0]).view(128, tn, 16).permute(1, 0, 2)
                F4 = (Lsel @ M[1]).view(128, tn, 16).permute(1, 0, 2)
            T2 = Wu + F2; T4 = Wu + F4
            tb = (1 - lam) * T2 + lam * T4 if lam else T2
            Dk = Din[a_idx]                                            # [T, 128, 128]
            Do = Dout[c_idx] if two else None

            def lcost(E):
                X = torch.bmm(Dk, E)
                if Do is not None:
                    X = torch.bmm(X, Do)
                return (E * X).sum((1, 2))
            # ---- base
            Ku = torch.tensor([base_K[a] for a in a_idx.tolist()], device=dev)
            Ua = P["Uin"][a_idx] if inner else None

            def bq(tgt):
                q = torch.empty_like(Wu); sy = torch.empty(T, 8, 256, dtype=torch.uint8, device=dev)
                vv = torch.empty(T, 8, dtype=torch.uint8, device=dev)
                for K in sorted(set(Ku.tolist())):
                    sel = (Ku == K).nonzero().flatten()
                    q[sel], sy[sel], vv[sel] = base_quant(R, tgt[sel], K, base_var)
                return q, sy, vv
            q2, s2, v2 = bq(tb)
            if inner:
                cb2 = lcost(tb - q2); cur = q2
                for it in range(inner):
                    nq, ns, nv = bq(tb + INNER_BETA * torch.bmm(Ua, tb - cur))
                    c = lcost(tb - nq); w = c < cb2
                    q2[w] = nq[w]; s2[w] = ns[w]; v2[w] = nv[w]; cb2 = torch.where(w, c, cb2); cur = nq
            r = T4 - q2
            # ---- residual (per candidate), K per unit
            K3u = k3[a_idx, c_idx]
            rms = r.square().mean((1, 2)).sqrt().clamp_min(1e-12)
            best = None
            jobs = [(ci, it) for ci in range(len(cands)) for it in range(inner + 1)]
            prev = {}
            for ci, it in jobs:
                cand = cands[ci]
                cb = cand[0]; sg = cand[1] if len(cand) > 1 else 1.0
                rt = r if it == 0 else r + INNER_BETA * torch.bmm(Ua, r - prev[ci])
                g = torch.empty_like(r); sy4 = torch.empty(T, 8, 256, dtype=torch.uint8, device=dev)
                for K in (2, 3):
                    sel = (K3u == (K == 3)).nonzero().flatten()
                    if not len(sel):
                        continue
                    s0 = (rms[sel] / (CB_RMS * P["gsr"][K] * gain)).view(-1, 1, 1)
                    st = viterbi(R.to_rings(sg * rt[sel] / s0), K, cb)
                    v, sy = R.from_states(st, K, cb)
                    g[sel] = v; sy4[sel] = sy.to(torch.uint8)
                # LS delta under the local metric
                Xg = torch.bmm(Dk, g)
                if Do is not None:
                    Xg = torch.bmm(Xg, Do)
                if nd == 1:
                    dl = ((Xg * r).sum((1, 2)) / (Xg * g).sum((1, 2)).clamp_min(1e-30)).half()
                    dfull = dl.float().view(T, 1, 1).expand(T, 128, 16)
                    dstore = dl.view(T, 1)
                else:
                    # per-ring delta: 8x8 normal equations  A_ij = <g_i, Din g_j Dout>, b_i = <g_i, Din r Dout>
                    gi = torch.zeros(T, 8, 2048, device=dev)
                    gi.scatter_(2, R.RI.unsqueeze(0).expand(T, 8, 256), g.reshape(T, 2048)[:, R.RI.flatten()].view(T, 8, 256))
                    gi = gi.view(T * 8, 128, 16)
                    Xi = torch.bmm(Dk.repeat_interleave(8, 0), gi)
                    if Do is not None:
                        Xi = torch.bmm(Xi, Do.repeat_interleave(8, 0))
                    Xi = Xi.view(T, 8, 2048); gi = gi.view(T, 8, 2048)
                    A = torch.bmm(Xi, gi.transpose(1, 2)); b = (Xi * r.reshape(T, 1, 2048)).sum(-1)
                    A = A + 1e-9 * A.diagonal(dim1=1, dim2=2).mean(-1).view(T, 1, 1) * torch.eye(8, device=dev)
                    dl = torch.linalg.solve(A, b).half()                              # [T, 8]
                    dmap = torch.empty(T, 2048, device=dev)
                    dmap[:, R.RI.flatten()] = dl.float().repeat_interleave(256, 1)
                    dfull = dmap.view(T, 128, 16)
                    dstore = dl
                q4 = q2 + dfull * g
                prev[ci] = dfull * g
                c = lcost(T4 - q4)
                if best is None:
                    best = dict(c=c, q4=q4, sy=sy4, d=dstore, ci=torch.zeros(T, dtype=torch.uint8, device=dev))
                else:
                    w = c < best["c"]
                    best["c"] = torch.where(w, c, best["c"])
                    best["q4"][w] = q4[w]; best["sy"][w] = sy4[w]; best["d"][w] = dstore[w]; best["ci"][w] = ci
            q4 = best["q4"]
            Q2[a_idx, c_idx] = q2; Q4[a_idx, c_idx] = q4
            sym2[a_idx, c_idx] = s2; bvar[a_idx, c_idx] = v2; sym4[a_idx, c_idx] = best["sy"]
            delta[a_idx, c_idx] = best["d"]
            cbid[a_idx, c_idx] = best["ci"]
            cost2[a_idx, c_idx] = lcost(T2 - q2); cost4[a_idx, c_idx] = best["c"]
            # ---- feedback update: M[rows, :] += dE @ Ln'[cols, :]
            dE2 = Wu - q2; dE4 = Wu - q4
            if two:
                n_hi = int(c_idx.max().item()) * 16 + 16
                Lsub = Ln[cols.flatten(), :n_hi].view(T, 16, n_hi)
                M[0][rows.flatten(), :n_hi] += torch.bmm(dE2, Lsub).reshape(T * 128, n_hi)
                M[1][rows.flatten(), :n_hi] += torch.bmm(dE4, Lsub).reshape(T * 128, n_hi)
            else:
                M[0][rows[0]] = dE2.permute(1, 0, 2).reshape(128, n)
                M[1][rows[0]] = dE4.permute(1, 0, 2).reshape(128, n)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    cand_names = [c[0] for c in cands]
    cb_global = torch.tensor([D.CODEBOOKS.index(c) for c in cand_names], dtype=torch.uint8, device=dev)[cbid.long()]
    return dict(Q2=Q2.permute(0, 2, 1, 3).reshape(k, n), Q4=Q4.permute(0, 2, 1, 3).reshape(k, n),
                sym2=sym2, sym4=sym4, delta=delta, cb=cb_global, cost2=cost2, cost4=cost4, k3=k3,
                base_K=list(base_K), tk=tk, tn=tn, bvar=bvar, base_var=base_var)


def select_k3(cost4, shard_axis, frac):
    """Top `frac` units per shard by level-4 local innovation cost (pilot) -> bool [tk, tn]."""
    tk, tn = cost4.shape
    _, shard = D.unit_order(tk, tn, shard_axis, cost4.device)
    c = cost4.flatten()
    mask = torch.zeros_like(c, dtype=torch.bool)
    for s in shard.unique().tolist():
        ids = (shard == s).nonzero().flatten()
        nsel = int(round(frac * len(ids)))
        if nsel:
            mask[ids[torch.topk(c[ids], nsel).indices]] = True
    return mask.view(tk, tn)


@torch.no_grad()
def refit_dense(P, Qrot):
    """Back-transform, exllamav3 refit_scales (un-rotated H), fp16 decode. -> (dense [out,in], suh, svh, proxy)."""
    Qm = _Qm()
    Wr = Qrot.clone()
    Wr = Qm.preapply_had_l(Wr, 128); Wr *= P["su"]; Wr = Qm.preapply_had_r(Wr, 128); Wr *= P["sv"]
    H_orig = Qm.unrotate_H(P["Hr"].cpu(), P["su_signs"].cpu())
    _, su, sv, e0, e1 = Qm.refit_scales(P["weight_orig"], Wr, H_orig, P["su"], P["sv"])
    del H_orig
    suh = su.flatten().half(); svh = sv.flatten().half()
    return D.dense_from_rotated(Qrot, suh, svh), suh, svh, (e0, e1)


@torch.no_grad()
def pack(P, enc, shard_axis, scales):
    """Planes per shard. scales: {level: (suh, svh)} for levels 2 and 4."""
    dev = enc["Q2"].device
    tk, tn = enc["tk"], enc["tn"]
    order, shard = D.unit_order(tk, tn, shard_axis, dev)
    Kb = torch.tensor(enc["base_K"], device=dev).view(tk, 1).expand(tk, tn).flatten()
    s2 = enc["sym2"].view(-1, 8, 256).long(); s4 = enc["sym4"].view(-1, 8, 256).long()
    k3 = enc["k3"].flatten()
    dl = enc["delta"].view(tk * tn, -1); cb = enc["cb"].flatten()
    nsh = int(shard.max().item()) + 1
    multi_cb = bool((cb != 0).any())
    planes = {x: dict(shards=[]) for x in ("base", "p4", "p4x")}
    planes["p4"].update(delta=[], cb=[])
    planes["p4x"]["mask"] = []
    shard_sorted = shard[order]
    for s in range(nsh):
        u = order[shard_sorted == s]                                        # storage order units of shard s
        out = [None] * len(u)
        for K in sorted(set(Kb[u].tolist())):
            pos = (Kb[u] == K).nonzero().flatten()
            pk = D.pack_bits(s2[u[pos]].view(-1, 256), K).view(len(pos), -1)
            for i, p in enumerate(pos.tolist()):
                out[p] = pk[i]
        planes["base"]["shards"].append(torch.cat(out).cpu())
        if enc.get("base_var"):
            planes["base"].setdefault("var", []).append(enc["bvar"].view(-1, 8)[u].flatten().cpu())
        planes["p4"]["shards"].append(D.pack_bits((s4[u] & 3).view(-1, 256), 2).flatten().cpu())
        planes["p4"]["delta"].append(dl[u].flatten().cpu())
        if multi_cb:
            planes["p4"]["cb"].append(cb[u].cpu())
        ux = u[k3[u]]
        planes["p4x"]["shards"].append(D.pack_bits((s4[ux] >> 2).view(-1, 256), 1).flatten().cpu())
        planes["p4x"]["mask"].append(k3[u].cpu())
    for L, x in D.SCALE_PLANE.items():
        planes[x]["suh"], planes[x]["svh"] = scales[L][0].cpu(), scales[L][1].cpu()
    planes["meta"] = dict(k=P["k"], n=P["n"], tk=tk, tn=tn, shard_axis=shard_axis, base_K=enc["base_K"],
                          base_var=enc.get("base_var"))
    return planes


def bits_per_level(planes):
    """Bits read at each level / weights (symbols + per-unit metadata + one copy of the level's fp16 scales)."""
    m = planes["meta"]; nw = m["k"] * m["n"]
    def nb(pl, keys=("shards", "delta")):
        b = 0
        for key in keys:
            if key in pl:
                b += sum(t.numel() * t.element_size() * 8 for t in pl[key])
        if "cb" in pl:                                   # 3 codebooks -> 2 bits per unit (stored as a byte here)
            b += sum(2 * t.numel() for t in pl["cb"])
        if "mask" in pl:
            b += sum(t.numel() for t in pl["mask"])
        if "var" in pl:                                  # base variant id per ring: log2(#variants) bits
            b += sum(D.variant_bits(m["base_var"]) * t.numel() for t in pl["var"])
        return b
    sc = 16 * (m["k"] + m["n"])
    base = nb(planes["base"]); a = nb(planes["p4"]); x = nb(planes["p4x"])
    return {2: (base + sc) / nw, 4: (base + a + x + sc) / nw, "artifact": (base + a + x + 2 * sc) / nw}


@torch.no_grad()
def encode_projection(W, H, count, sigma, *, G=None, lam=0.3, shard_axis="n", base_K=None, k3=None,
                      cands=(("mul1",),), delta_per="unit", l3frac=0.5, gain=1.0, seed=91426, P=None, inner=0,
                      base_var=None):
    """Full pipeline for one projection. Returns (planes, internal {2,4: dense [out,in]}, info, P)."""
    t0 = time.time()
    P = P or prep(W, H, count, sigma, seed=seed, G=G)
    enc = encode_rotated(P, lam=lam, base_K=base_K, k3=k3, cands=cands, delta_per=delta_per, gain=gain,
                         shard_axis=shard_axis, inner=inner, base_var=base_var)
    rot = {2: enc["Q2"], 4: enc["Q4"]}
    dense, scales, prox = {}, {}, {}
    Hr = P["Hr"]
    def tr(E):
        X = Hr @ E
        if "Ho" in P:
            X = X @ P["Ho"]
        return float((E * X).sum())
    den = tr(P["weight"])
    for L, Qr in rot.items():
        prox[L] = tr(P["weight"] - Qr) / den
        dense[L], suh, svh, _ = refit_dense(P, Qr)
        scales[L] = (suh, svh)
    planes = pack(P, enc, shard_axis, scales)
    info = dict(time=time.time() - t0, gs=P["gs"], gsr=P["gsr"], aos=P["aos"], proxy_rot=prox,
                bits=bits_per_level(planes), cost4_mean=float(enc["cost4"].mean()), cost2_mean=float(enc["cost2"].mean()))
    return planes, dense, info, P, enc


def free(P):
    for key in list(P.keys()):
        P[key] = None
    torch.cuda.empty_cache()
