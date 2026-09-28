"""NestQuant v1 reference encoder (thread 12): one fit -> 2-bit base + level-4 int-fold plane (kernel-exact, ref15).

Per projection (EXL3 layout W (k=in, n=out)):
  1. EXL3 preprocessing mirrored from harness.quantize_exl3_like (su signs + Had128 on H, sv, block_rms, g_scale K2).
  2. Block LDL of the rotated H at the kernel chunk (128; a ring spans a whole chunk). Output metric G (gate/up,
     two-sided): prepare_H_out(G, sv), block 16 = strip.
  3. Dual-state LDLQ over 128x16 units (two-sided: anti-diagonals from the far corner; one-sided: chunk rows,
     k high -> low). F2 from E2 = W - Q2, F4 from E4 = W - Q4:
        base   target (1-lam) T2 + lam T4 -> K2 mul1 Viterbi (sg4: best of 8 per-ring variants a) -> Q2 (fp16 exact)
        resid  r = T4 - Q2 -> Viterbi of r / (a s0) at the unit's residual K (2, or 1.5/2.5/3 by the rule)
               delta = LS <a g, M r>/<a g, M a g> (>= 0), (Mb, N) candidates around delta_to_MbN(delta) scored by the
               EXACT ref15 fold under the local metric; Q4 = fold(S(sb), S(sr), Mb, N, a).
        inner  optional accept-if-better within-chunk feedback iterations (GLM default 2), base and residual.
     Kernel ring position p <-> EXL3 Viterbi step i = (-p) mod 256 (so 0xAAAA's extra bit at odd p = EXL3 odd i).
  4. FROZEN BASE: `base=` passes a previous fit's base (states, variants, Q2) and its L2 scales; only the level-4
     state/residual is re-run. Rate variants reuse the reference (uniform K2) fit's base => base plane bytes identical.
  5. Per-level su/sv refit (exllamav3 refit_scales), fp16 decode; planes packed per TP shard (nq_decode docstring).
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
INNER_BETA = float(os.environ.get("NQ_INNER_BETA", "1.0"))
RES_KS = (1.5, 2, 2.5, 3)


def _Qm():
    return h._ex()


def _k(K):
    return int(K) if float(K).is_integer() else float(K)


def viterbi(rings, K):
    """rings [R, 256] fp32 (Viterbi step order) -> 16-bit states [R, 256] (Viterbi order)."""
    _, st = _Qm().quantize_tiles(rings.float().contiguous(), {"K": _k(K), "mul1": True})
    return st.long() & 0xFFFF


class Ring:
    """Unit [T,128,16] <-> rings. Kernel ring position p = (-i) mod 256 of Viterbi step i."""
    def __init__(self, dev):
        self.RI = D.ring_index(dev)                                     # [8, 256] kernel order
        self.ip = (-torch.arange(256, device=dev)) % 256                  # i(p) == p(i)
        self.vo = self.RI[:, self.ip].flatten()                          # Viterbi order gather
        self.ri = self.RI.flatten()

    def to_rings(self, X):                                               # -> [T*8, 256] Viterbi order
        return X.reshape(X.shape[0], 2048)[:, self.vo].reshape(-1, 256)

    def to_kring(self, X):                                               # -> [T, 8, 256] kernel order
        return X.reshape(X.shape[0], 2048)[:, self.ri].view(-1, 8, 256)

    def kstates(self, st):                                               # Viterbi-order states -> kernel [T,8,256]
        return st[:, self.ip].view(-1, 8, 256)

    def to_unit(self, V):                                                # kernel-order [T,8,256] -> [T,128,16]
        out = torch.empty(V.shape[0], 2048, device=V.device, dtype=torch.float32)
        out[:, self.ri] = V.reshape(V.shape[0], 2048).float()
        return out.view(-1, 128, 16)

    def ring_map(self, a):                                               # per-ring [T,8] -> per-weight [T,128,16]
        return self.to_unit(a.float().unsqueeze(-1).expand(-1, 8, 256))


def roundtrip(sk, K):
    """kernel states [T,8,256] -> states re-derived from the packed stream (what the decoder sees), #mismatch."""
    T = sk.shape[0]
    st = D.stream_states(D.pack_stream(D.symbols(sk.reshape(-1, 256), K), K), K).view(T, 8, 256)
    return st, int((st != sk).sum())


def base_quant(R, tgt, var=None):
    """Base code of units tgt [T,128,16] at K2 -> (Q2 [T,128,16], states [T,8,256], variant ids [T,8], a [T,8,1]|None).
    Variant choice per ring by plain ring MSE (thread 17)."""
    T = tgt.shape[0]
    rings = R.to_rings(tgt)
    tk = R.to_kring(tgt).double()
    tab = D.variant_table(var, tgt.device) if var else torch.ones(1, device=tgt.device, dtype=torch.float64)
    best = None
    for vi in range(len(tab)):
        a = tab[vi]
        sk, _ = roundtrip(R.kstates(viterbi(rings / float(a), 2)), 2)
        q = D.q2_values(D.hsum(sk), a if var else None)                   # [T,8,256] exact fp16 values
        m = (q - tk).square().sum(-1)                                     # [T, 8]
        if best is None:
            best, bsk, bq = m, sk, q
            bsel = torch.zeros_like(m, dtype=torch.long)
        else:
            w = m < best
            best = torch.where(w, m, best); bsel = torch.where(w, vi, bsel)
            bsk = torch.where(w.unsqueeze(-1), sk, bsk); bq = torch.where(w.unsqueeze(-1), q, bq)
    av = tab[bsel].unsqueeze(-1) if var else None
    return R.to_unit(bq), bsk, bsel.to(torch.uint8), av


def ldl_blocks(Hr, b, sigma):
    """H = L' D L'^T with unit block-lower L' (identity diagonal blocks) and block-diagonal D (block b)."""
    n = Hr.shape[0]; m = n // b
    Hc = Hr.clone()
    for attempt in range(11):
        try:
            C = torch.linalg.cholesky(Hc); break
        except torch._C._LinAlgError:
            Hc.diagonal().add_(2.0 * sigma * Hc.diagonal().mean())
    DL = torch.diagonal(C.view(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1).contiguous()
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
    gsr = {K: h._g_scale_search(samp * Qm.ldlq_drift(_k(K)), _k(K), q)[0] for K in RES_KS}
    weight *= gs
    su /= gs
    Lk, Din = ldl_blocks(Hm, 128, sigma)
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


def mbn_candidates(delta):
    """(Mb, N) candidates [T, C] around the ref15 largest-Mb rational of delta (invalid ones -> the ref15 pick)."""
    Mb0, N0 = D.delta_to_MbN(delta)
    Mbs, Ns = [], []
    for dm in (0, -1, -2, -3):
        Mb = (Mb0 + dm).clamp(1, 255)
        Nr = torch.round(delta.double().clamp_min(0) * Mb).long()
        for dn in (-1, 0, 1):
            Mbs.append(Mb); Ns.append(Nr + dn)
    Mb = torch.stack(Mbs, 1); N = torch.stack(Ns, 1)
    ok = (N >= 0) & (Mb + N <= 257) & (Mb >= 1) & (Mb <= 255)
    N = torch.where(ok, N, N0.view(-1, 1)); Mb = torch.where(ok, Mb, Mb0.view(-1, 1))
    return Mb, N


def A_K0(S):
    return D.A * S.double() + D.K0


@torch.no_grad()
def encode_rotated(P, meta, lam=0.3, base=None, base_var=None, inner=0):
    """Dual-state LDLQ with the ref15 fold. meta: dict(tk, tn, shard_axis, res_rule, mask_flat) -> residual K per
    unit via nq_decode.res_K_units. base: frozen base dict from a previous call (or None = fit the base)."""
    dev = P["weight"].device
    Wt = P["weight"]; k, n = Wt.shape; tk, tn = k // 128, n // 16
    Lk, Din = P["Lk"], P["Din"]
    two = "Ln" in P
    Ln = P.get("Ln"); Dout = P.get("Dout")
    R = Ring(dev)
    Kr = D.res_K_units(meta, meta.get("mask_flat"), dev).view(tk, tn)
    W4 = Wt.view(tk, 128, tn, 16)
    M = torch.zeros(2, k, n, device=dev)
    Q4 = torch.zeros(tk, tn, 128, 16, device=dev)
    if base is None:
        Q2 = torch.zeros_like(Q4)
        sb_all = torch.zeros(tk, tn, 8, 256, dtype=torch.int32, device=dev)
        var_all = torch.zeros(tk, tn, 8, dtype=torch.uint8, device=dev)
    else:
        sb_all, var_all, Q2 = base["sb"].to(dev), base["var"].to(dev), base["Q2"].to(dev)
    sr_all = torch.zeros(tk, tn, 8, 256, dtype=torch.int32, device=dev)
    Mb_all = torch.zeros(tk, tn, dtype=torch.long, device=dev); N_all = torch.zeros_like(Mb_all)
    cost2 = torch.zeros(tk, tn, device=dev); cost4 = torch.zeros(tk, tn, device=dev)
    ar128 = torch.arange(128, device=dev); ar16 = torch.arange(16, device=dev)
    Lkt = Lk.T.contiguous()
    tab = D.variant_table(base_var, dev) if base_var else None
    mism = 0
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
            rows = (a_idx.unsqueeze(1) * 128 + ar128)
            cols = (c_idx.unsqueeze(1) * 16 + ar16)
            Wu = W4[a_idx, :, c_idx, :]
            if two:
                Lsel = Lkt[rows.flatten()].view(T, 128, k)
                Ms = M[:, :, cols.flatten()].view(2, k, T, 16).permute(0, 2, 1, 3)
                F2 = torch.bmm(Lsel, Ms[0]); F4 = torch.bmm(Lsel, Ms[1])
            else:
                Lsel = Lkt[rows[0]]
                F2 = (Lsel @ M[0]).view(128, tn, 16).permute(1, 0, 2)
                F4 = (Lsel @ M[1]).view(128, tn, 16).permute(1, 0, 2)
            T2 = Wu + F2; T4 = Wu + F4
            Dk = Din[a_idx]
            Do = Dout[c_idx] if two else None
            Ua = P["Uin"][a_idx] if inner else None

            def mt(E):
                X = torch.bmm(Dk, E)
                return torch.bmm(X, Do) if Do is not None else X

            def lcost(E):
                return (E * mt(E)).sum((1, 2))
            # ---- base (fit, or frozen)
            if base is None:
                tb = (1 - lam) * T2 + lam * T4 if lam else T2
                q2, sb, vv, av = base_quant(R, tb, base_var)
                if inner:
                    cb2 = lcost(tb - q2); cur = q2
                    for it in range(inner):
                        nq, ns, nv, na = base_quant(R, tb + INNER_BETA * torch.bmm(Ua, tb - cur), base_var)
                        c = lcost(tb - nq); w = c < cb2
                        q2[w] = nq[w]; sb[w] = ns[w]; vv[w] = nv[w]
                        if av is not None:
                            av[w] = na[w]
                        cb2 = torch.where(w, c, cb2); cur = nq
                sb_all[a_idx, c_idx] = sb.int(); var_all[a_idx, c_idx] = vv; Q2[a_idx, c_idx] = q2
            else:
                q2 = Q2[a_idx, c_idx]; sb = sb_all[a_idx, c_idx].long(); vv = var_all[a_idx, c_idx]
                av = tab[vv.long()].unsqueeze(-1) if base_var else None
            Sb = D.hsum(sb)
            amap = R.ring_map(av.squeeze(-1)) if av is not None else torch.ones_like(Wu)
            r = T4 - q2
            rms = r.square().mean((1, 2)).sqrt().clamp_min(1e-12)
            Ku = Kr[a_idx, c_idx]
            best = None; prev = None
            for it in range(inner + 1):
                rt = r if it == 0 else r + INNER_BETA * torch.bmm(Ua, r - prev)
                sr = torch.empty(T, 8, 256, dtype=torch.long, device=dev)
                for K in sorted(set(Ku.tolist())):
                    sel = (Ku == K).nonzero().flatten()
                    s0 = (rms[sel] / (CB_RMS * P["gsr"][_k(K)])).view(-1, 1, 1)
                    st = viterbi(R.to_rings(rt[sel] / (amap[sel] * s0)), K)
                    sk, mm = roundtrip(R.kstates(st), K)
                    mism += mm
                    sr[sel] = sk
                Sr = D.hsum(sr)
                ag = amap * R.to_unit(A_K0(Sr))                                    # unrounded a * mul1(sr)
                Xg = mt(ag)
                dl = ((Xg * r).sum((1, 2)) / (Xg * ag).sum((1, 2)).clamp_min(1e-30)).clamp_min(0)
                Mb, N = mbn_candidates(dl)
                cbest = None
                for ci in range(Mb.shape[1]):
                    q4c = R.to_unit(D.fold(Sb, Sr, Mb[:, ci], N[:, ci], av))
                    c = lcost(T4 - q4c)
                    if cbest is None:
                        cbest, q4, bMb, bN = c, q4c, Mb[:, ci].clone(), N[:, ci].clone()
                    else:
                        w = c < cbest
                        cbest = torch.where(w, c, cbest); q4[w] = q4c[w]; bMb[w] = Mb[w, ci]; bN[w] = N[w, ci]
                prev = q4 - q2
                if best is None:
                    best = dict(c=cbest, q4=q4, sr=sr, Mb=bMb, N=bN)
                else:
                    w = cbest < best["c"]
                    best["c"] = torch.where(w, cbest, best["c"]); best["q4"][w] = q4[w]; best["sr"][w] = sr[w]
                    best["Mb"][w] = bMb[w]; best["N"][w] = bN[w]
            q4 = best["q4"]
            Q4[a_idx, c_idx] = q4; sr_all[a_idx, c_idx] = best["sr"].int()
            Mb_all[a_idx, c_idx] = best["Mb"]; N_all[a_idx, c_idx] = best["N"]
            cost2[a_idx, c_idx] = lcost(T2 - q2); cost4[a_idx, c_idx] = best["c"]
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
    return dict(Q2=Q2, Q4=Q4, Q2r=Q2.permute(0, 2, 1, 3).reshape(k, n), Q4r=Q4.permute(0, 2, 1, 3).reshape(k, n),
                sb=sb_all, var=var_all, sr=sr_all, Mb=Mb_all, N=N_all, Kr=Kr, cost2=cost2, cost4=cost4,
                tk=tk, tn=tn, mismatch=mism, base_var=base_var)


def frozen_base(enc):
    return dict(sb=enc["sb"], var=enc["var"], Q2=enc["Q2"])


def select_mask(cost4, shard_axis, frac):
    """Top `frac` units per shard by level-4 local cost (reference fit) -> bool flat [tk*tn]."""
    tk, tn = cost4.shape
    _, shard = D.unit_order(tk, tn, shard_axis, cost4.device)
    c = cost4.flatten()
    mask = torch.zeros_like(c, dtype=torch.bool)
    for s in shard.unique().tolist():
        ids = (shard == s).nonzero().flatten()
        nsel = int(round(frac * len(ids)))
        if nsel:
            mask[ids[torch.topk(c[ids], nsel).indices]] = True
    return mask


@torch.no_grad()
def refit_dense(P, Qrot):
    """Back-transform, exllamav3 refit_scales (un-rotated H), fp16 decode. -> (dense [out,in], suh, svh)."""
    Qm = _Qm()
    Wr = Qrot.clone()
    Wr = Qm.preapply_had_l(Wr, 128); Wr *= P["su"]; Wr = Qm.preapply_had_r(Wr, 128); Wr *= P["sv"]
    H_orig = Qm.unrotate_H(P["Hr"].cpu(), P["su_signs"].cpu())
    _, su, sv, e0, e1 = Qm.refit_scales(P["weight_orig"], Wr, H_orig, P["su"], P["sv"])
    del H_orig
    suh = su.flatten().half(); svh = sv.flatten().half()
    return D.dense_from_rotated(Qrot, suh, svh), suh, svh


@torch.no_grad()
def pack(P, enc, meta, scales):
    dev = enc["Q2"].device
    tk, tn = enc["tk"], enc["tn"]
    order, shard = D.unit_order(tk, tn, meta["shard_axis"], dev)
    sb = enc["sb"].view(-1, 8, 256).long(); sr = enc["sr"].view(-1, 8, 256).long()
    Kr = enc["Kr"].flatten()
    word = (enc["Mb"] | (enc["N"] << 8)).flatten().int()
    var = enc["var"].view(-1, 8)
    nsh = int(shard.max().item()) + 1
    planes = {"base": dict(shards=[]), "p4": dict(shards=[], word=[])}
    if meta["res_rule"]["kind"] == "mask":
        planes["p4"]["mask"] = []
    shard_sorted = shard[order]
    for s in range(nsh):
        u = order[shard_sorted == s]
        planes["base"]["shards"].append(D.pack_stream(D.symbols(sb[u].view(-1, 256), 2), 2).flatten().cpu())
        if enc["base_var"]:
            planes["base"].setdefault("var", []).append(var[u].flatten().cpu())
        out = [None] * len(u)
        for K in sorted(set(Kr[u].tolist())):
            pos = (Kr[u] == K).nonzero().flatten()
            pk = D.pack_stream(D.symbols(sr[u[pos]].view(-1, 256), K), K).view(len(pos), -1)
            for i, p in enumerate(pos.tolist()):
                out[p] = pk[i]
        planes["p4"]["shards"].append(torch.cat(out).cpu())
        planes["p4"]["word"].append(word[u].cpu())
        if "mask" in planes["p4"]:
            planes["p4"]["mask"].append((Kr[u] != float(meta["res_rule"]["K"])).cpu())
    for L, x in D.SCALE_PLANE.items():
        planes[x]["suh"], planes[x]["svh"] = scales[L][0].cpu(), scales[L][1].cpu()
    planes["meta"] = dict(k=P["k"], n=P["n"], tk=tk, tn=tn, shard_axis=meta["shard_axis"], base_K=2,
                          base_var=enc["base_var"], res_rule=meta["res_rule"])
    return planes


@torch.no_grad()
def encode_projection(P, *, shard_axis, res_rule=None, mask_flat=None, lam=0.3, base=None, base_var=None, inner=0,
                      base_scales=None):
    """One projection -> (planes, internal {2,4: dense [out,in]}, info, enc, scales). res_rule default uniform K2.
    base + base_scales: frozen base and its L2 scales (so the whole base plane is byte-identical)."""
    t0 = time.time()
    k, n = P["k"], P["n"]
    meta = dict(tk=k // 128, tn=n // 16, shard_axis=shard_axis, res_rule=res_rule or dict(kind="uniform", K=2),
                mask_flat=mask_flat)
    enc = encode_rotated(P, meta, lam=lam, base=base, base_var=base_var, inner=inner)
    rot = {2: enc["Q2r"], 4: enc["Q4r"]}
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
        if L == 2 and base_scales is not None:
            scales[L] = base_scales
            dense[L] = D.dense_from_rotated(Qr, *base_scales)
        else:
            dense[L], suh, svh = refit_dense(P, Qr)
            scales[L] = (suh, svh)
    planes = pack(P, enc, meta, scales)
    delta = enc["N"].double() / enc["Mb"].double()
    info = dict(time=time.time() - t0, gs=P["gs"], gsr=P["gsr"], aos=P["aos"], proxy_rot=prox,
                bits=D.bits_per_level(planes), cost4_mean=float(enc["cost4"].mean()),
                cost2_mean=float(enc["cost2"].mean()), stream_mismatch=enc["mismatch"],
                delta_mean=float(delta.mean()), Mb_min=int(enc["Mb"].min()), Mb_mean=float(enc["Mb"].float().mean()),
                N0_frac=float((enc["N"] == 0).float().mean()),
                K_frac={str(K): float((enc["Kr"] == K).float().mean()) for K in enc["Kr"].unique().tolist()})
    return planes, dense, info, enc, scales


def free(P):
    for key in list(P.keys()):
        P[key] = None
    torch.cuda.empty_cache()


# ================================================================================================ production API
PROJ = ("gate", "up", "down")
PROD = dict(base_var="sign", lam=0.3, inner=2, K_hi=2.5, sigma={"gate": 0.5, "up": 0.5, "down": 1.0},
            axis={"gate": "n", "up": "n", "down": "k"}, units_per_shard=768)
DEFAULT_RATE = 4.09375          # L4 bpw incl. all metadata; see REPORT/final message for the selection rule


def rate_rule(ref_bits, rate, K_hi=PROD["K_hi"]):
    """Positional residual rule for an L4 target rate given the uniform-K2 L4 bits (per projection)."""
    if abs(rate - ref_bits) < 1e-9:
        return dict(kind="uniform", K=2)
    Kh = K_hi if rate > ref_bits else 1.5
    f = max(0, round((rate - ref_bits) / (Kh - 2) * PROD["units_per_shard"])) / PROD["units_per_shard"]
    return dict(kind="pos", K=2, K_hi=Kh, frac=f)


@torch.no_grad()
def encode_expert(Ws, HG, rate=DEFAULT_RATE, count=1, sigma=None, sigma_out=0.03, lam=PROD["lam"],
                  base_var=PROD["base_var"], inner=PROD["inner"], check=True, canonical_base=True):
    """One expert -> (artifact {gate, up, down: planes, meta}, dense {2, 4: [g, u, d] fp32 [out, in]}).
    Ws: [Wg, Wu, Wd] teacher [out, in]; HG: thread-12 glm_H format {"H": [Hx, Hx, Ha], "G": [Gg, Gu, None]}
    (e.g. threads/19-full-capture/nq19_load.Capture().glm_H(L, E)).
    canonical_base: fit the base once against the uniform-K2 residual (rate-independent base bytes), then re-fit
    only the P4 plane at `rate` on that frozen base (2 passes). False = single joint pass at `rate`."""
    sigma = sigma or PROD["sigma"]
    art, dense, info = {}, {2: [], 4: []}, {}
    for pi, pn in enumerate(PROJ):
        P = prep(Ws[pi], HG["H"][pi], count, sigma[pn], G=HG["G"][pi], sigma_out=sigma_out)
        ax = PROD["axis"][pn]
        nw = P["k"] * P["n"]
        ref_bits = 4 + 16 / 2048 + 16 * (P["k"] + P["n"]) / nw + (D.variant_bits(base_var) / 256 if base_var else 0)
        rule = rate_rule(ref_bits, rate)
        if canonical_base and rule["kind"] != "uniform":
            _, dn0, _, enc0, sc0 = encode_projection(P, shard_axis=ax, lam=lam, base_var=base_var, inner=inner)
            planes, dn, inf, enc, _ = encode_projection(P, shard_axis=ax, lam=lam, base_var=base_var, inner=inner,
                                                        base=frozen_base(enc0), base_scales=sc0[2], res_rule=rule)
            inf["L2_equal_canonical"] = bool(torch.equal(dn[2], dn0[2]))
            del enc0, dn0
        else:
            planes, dn, inf, enc, _ = encode_projection(P, shard_axis=ax, lam=lam, base_var=base_var, inner=inner,
                                                        res_rule=rule)
        if check:
            rot = D.rotated_levels(planes)
            inf["bitexact"] = {L: bool(torch.equal(D.decode_matrix(planes, L, rot=rot), dn[L])) for L in (2, 4)}
            assert all(inf["bitexact"].values()), (pn, inf["bitexact"])
            del rot
        art[pn] = planes
        for L in (2, 4):
            dense[L].append(dn[L].cpu())
        info[pn] = {k: v for k, v in inf.items() if k in ("bits", "proxy_rot", "time", "bitexact", "L2_equal_canonical",
                                                         "stream_mismatch", "K_frac")}
        del enc; free(P)
    art["meta"] = dict(format="nestquant-v1", rate=rate, base_var=base_var, lam=lam, inner=inner, sigma=sigma,
                       canonical_base=canonical_base, info=info)
    return art, dense


def main():
    import argparse
    ap = argparse.ArgumentParser(description="NestQuant v1: encode one GLM expert into a base + P4 artifact")
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE, help="level-4 bpw incl. metadata")
    ap.add_argument("--stats", choices=["t19", "t12"], default="t19",
                    help="t19 = threads/19 full-model capture (nq19_load.Capture().glm_H); t12 = thread-08 H from the "
                         "orbit training sample (nq_run.glm_H)")
    ap.add_argument("--out", required=True); ap.add_argument("--dense-out", help="also save internal dense {2,4}")
    ap.add_argument("--single-pass", action="store_true", help="joint fit at --rate (base not rate-canonical)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = h.load_expert(a.layer, a.expert)
    if a.stats == "t19":
        sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
        import nq19_load
        HG = nq19_load.Capture().glm_H(a.layer, a.expert)
    else:
        import nq_run
        HG = nq_run.glm_H(data, a.layer, a.expert)
    art, dense = encode_expert(data.teacher, HG, rate=a.rate, canonical_base=not a.single_pass)
    torch.save(art, a.out)
    if a.dense_out:
        torch.save(dense, a.dense_out)
    for pn in PROJ:
        print(pn, art["meta"]["info"][pn])


if __name__ == "__main__":
    main()
