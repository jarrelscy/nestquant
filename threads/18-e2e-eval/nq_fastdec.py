"""Batched GPU NestQuant decoder (thread 18): bit-exact with threads/12 nq_decode.decode_matrix, many experts per call.

Reads one layer's TP8 shard files (L{L}/tp{s}.safetensors, thread-25 container) lazily: only the requested experts'
tensors are read (each rank reads ~1/WORLD of every shard file); no per-expert E{E}.pt unpickling.

Exactness (vs nq_decode.decode_matrix + decode_expert's inter_perm un-permute):
  - stream states: 16-bit tail-biting windows gathered from 3 bytes of the ring (+2 wrapped bytes) instead of
    nq_decode's bit-expansion; integer-identical.
  - hash, Q2 (q2_values), Q4 (fold): nq_decode's own elementwise functions on stacked [B*U, 8, 256] tensors.
  - to_matrix: the same scatter, batched.
  - Hadamard + scales: exllamav3 preapply_had_l/r (fp32 GEMM with the same 128x128 matrix), per matrix by default
    (batch_had=True stacks the matrices along the GEMM batch dim; only used if the gate shows it bit-exact).
  - low-rank / ocol planes and the level-4 add order: nq_decode.apply_ocol per expert (the same function).
Planes come from the shard files via T12's nq_layer.assemble rule (reimplemented on the lazily-read parts; the gate
compares against E.pt decodes, so any divergence of the assembly would show).
"""
import os, sys, json
import torch
from safetensors import safe_open

T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T25 = "/home/coder/git/nestquant/threads/25-campaign"
for p in (T12, T25):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq_decode as D                      # noqa: E402

PROJ = ("gate", "up", "down")
NSH = 8
FORMAT = "nestquant-v1-tp-safetensors"


def _untree(n, get):
    if "D" in n:
        return {k: _untree(v, get) for k, v in n["D"]}
    if "T" in n:
        return get(n["T"])
    return n["V"]


class LayerShards:
    """Lazy view of L{L}/tp{s}.safetensors: art(E) == nq_layer.assemble(root, L, E) without reading other experts."""

    def __init__(self, root, L):
        d = f"{root}/L{L}"
        self.man = json.load(open(f"{d}/manifest.json"))
        self.files, self.trees = [], []
        for s in range(NSH):
            f = safe_open(f"{d}/tp{s}.safetensors", framework="pt", device="cpu")
            m = f.metadata()
            assert m.get("format") == FORMAT, (d, s, m.get("format"))
            self.files.append(f)
            self.trees.append({E: sub for E, sub in json.loads(m["tree"])["D"]})
        self.experts = [int(E) for E in self.trees[0]]

    def parts(self, E):
        key = E if E in self.trees[0] else str(E)
        return [_untree(self.trees[s][key], self.files[s].get_tensor) for s in range(NSH)]

    def art(self, E):
        """mirror of threads/12 nq_layer.assemble (shard -> nq_decode artifact)."""
        parts = self.parts(E)
        art = {}
        for pn in PROJ:
            meta = dict(self.man["proj_meta"][pn])
            cat = lambda key: [p[pn][key] for p in parts]           # noqa: E731
            if pn == "down":
                suh2, svh2 = torch.cat(cat("suh2")), parts[0][pn]["svh2"]
                suh4, svh4 = torch.cat(cat("suh4")), parts[0][pn]["svh4"]
            else:
                suh2, svh2 = parts[0][pn]["suh2"], torch.cat(cat("svh2"))
                suh4, svh4 = parts[0][pn]["suh4"], torch.cat(cat("svh4"))
            base = dict(shards=cat("base"), suh=suh2, svh=svh2)
            if "var" in parts[0][pn]:
                base["var"] = cat("var")
            p4 = dict(shards=cat("p4"), word=cat("word"), suh=suh4, svh=svh4)
            if "lrU2" in parts[0][pn]:
                if pn == "down":
                    V, U2, U4 = torch.cat(cat("lrV"), 1), parts[0][pn]["lrU2"], parts[0][pn]["lrU4"]
                else:
                    V = parts[0][parts[0][pn].get("lrV_from", pn)]["lrV"]
                    U2, U4 = torch.cat(cat("lrU2"), 1), torch.cat(cat("lrU4"), 1)
                base["lr"] = dict(V=V, U2=U2); p4["lr"] = dict(U4=U4)
            art[pn] = dict(base=base, p4=p4, meta=meta)
        return art


# ------------------------------------------------------------------------------------------------ streams
_W = {}


def _widths(K, dev):
    key = (float(K), str(dev))
    if key not in _W:
        w, off, nb = D.widths(float(K), dev)
        _W[key] = (off.to(torch.int32), nb)
    return _W[key]


def states_rings(buf, K):
    """uint8 [R, nb/8] ring streams -> int64 [R, 256] 16-bit states (== nq_decode.stream_states)."""
    off, nb = _widths(K, buf.device)
    ext = torch.cat([buf, buf[:, :2]], 1).to(torch.int32)             # tail-biting wrap: bits nb..nb+15 = 0..15
    o8 = (off >> 3).long()
    sh = (off & 7)
    w = ext[:, o8] | (ext[:, o8 + 1] << 8) | (ext[:, o8 + 2] << 16)
    return ((w >> sh) & 0xFFFF).long()


def gather_states(raw, Ks):
    """raw uint8 [sum bytes] (units back to back, 8 rings each), Ks [U] float -> [U, 8, 256] (== _gather_streams)."""
    dev = raw.device
    kv = sorted(set(Ks.unique().tolist()))
    per = torch.zeros(len(Ks), dtype=torch.long, device=dev)
    for K in kv:
        per[Ks == K] = 8 * D.ring_bytes(K)
    offs = torch.cumsum(per, 0) - per
    assert int(per.sum()) == raw.numel(), (int(per.sum()), raw.numel())
    st = torch.empty(len(Ks), 8, 256, dtype=torch.long, device=dev)
    for K in kv:
        sel = (Ks == K).nonzero().flatten()
        nbytes = 8 * D.ring_bytes(K)
        if len(sel) == len(Ks):                                       # uniform K: plain view
            rings = raw.view(-1, nbytes // 8)
        else:
            idx = offs[sel].unsqueeze(1) + torch.arange(nbytes, device=dev)
            rings = raw[idx].view(-1, nbytes // 8)
        st[sel] = states_rings(rings, K).view(len(sel), 8, 256)
    return st


# ------------------------------------------------------------------------------------------------ decode
_ORD = {}


def _order(m, dev):
    key = (m["tk"], m["tn"], m["shard_axis"], str(dev))
    if key not in _ORD:
        _ORD[key] = D.unit_order(m["tk"], m["tn"], m["shard_axis"], dev)[0]
    return _ORD[key]


@torch.no_grad()
def rotated_batch(Ps, levels, dev):
    """Ps: list of projection planes with identical meta geometry -> {level: fp32 [B, k, n] rotated matrices}."""
    m = Ps[0]["meta"]
    tk, tn, k, n = m["tk"], m["tn"], m["k"], m["n"]
    U, B = tk * tn, len(Ps)
    order = _order(m, dev)
    for P in Ps:
        assert all(P["meta"][x] == m[x] for x in ("tk", "tn", "k", "n", "shard_axis", "base_var"))
    raw_b = torch.cat([t for P in Ps for t in P["base"]["shards"]]).to(dev)
    Sb = D.hsum(gather_states(raw_b, torch.full((B * U,), 2.0, device=dev)))
    del raw_b
    a = None
    if m.get("base_var"):
        var = torch.cat([t for P in Ps for t in P["base"]["var"]]).to(dev).long().view(B * U, 8)
        a = D.variant_table(m["base_var"], dev)[var].unsqueeze(-1)
    out = {}
    if 2 in levels:
        out[2] = D.q2_values(Sb, a)
    if 4 in levels:
        Ku = []
        for P in Ps:
            mflat = None
            if P["p4"].get("mask"):
                mflat = torch.zeros(U, dtype=torch.bool, device=dev)
                mflat[order] = D._cat(P["p4"]["mask"], dev).bool()
            Ku.append(D.res_K_units(P["meta"], mflat, dev)[order])
        raw_r = torch.cat([t for P in Ps for t in P["p4"]["shards"]]).to(dev)
        Sr = D.hsum(gather_states(raw_r, torch.cat(Ku)))
        del raw_r
        bw = torch.cat([t for P in Ps for t in P["p4"]["word"]]).to(dev).long()
        out[4] = D.fold(Sb, Sr, bw & 255, (bw >> 8) & 255, a)
        del Sr
    del Sb, a
    RI = D.ring_index(dev).flatten()
    res = {}
    for lv, Q in out.items():
        X = torch.empty(B, U, 2048, device=dev)
        X[:, order.unsqueeze(1), RI.unsqueeze(0)] = Q.view(B, U, 2048).float()
        res[lv] = X.view(B, tk, tn, 128, 16).permute(0, 1, 3, 2, 4).reshape(B, k, n)
        del X
    return res


def _had():
    from exllamav3.modules.quant.exl3_lib import quantize as Qm
    return Qm


@torch.no_grad()
def dense_batch(R, suh, svh, batch_had=False):
    """R fp32 [B, k, n] rotated; suh [B, k], svh [B, n] fp16 -> list of fp32 [n, k] (== nq_decode.dense_from_rotated)."""
    Qm = _had()
    if not batch_had:
        outs = []
        for b in range(R.shape[0]):
            outs.append(D.dense_from_rotated(R[b], suh[b], svh[b]))
        return outs
    B, k, n = R.shape
    w = R.half()
    had = Qm.get_hadamard_dt(128, w.device, torch.float, 1 / 128 ** 0.5)
    w = (had @ w.float().view(-1, 128, n)).view(B, k, n).half()
    w *= suh.to(w.device).unsqueeze(2)
    w = (w.float().view(B, k, -1, 128) @ had).view(B, k, n).half()
    w *= svh.to(w.device).unsqueeze(1)
    return [w[b].float().T.contiguous() for b in range(B)]


@torch.no_grad()
def decode_experts(arts, levels, dev, batch_had=False, perms=None):
    """arts: list of nq_decode artifacts -> {level: [[gate, up, down] fp32 [out, in] per expert]}.
    perms: optional list of inter_perm (or None) per expert, un-permuted as nq_decode.decode_expert."""
    out = {lv: [[None] * 3 for _ in arts] for lv in levels}
    for j, pn in enumerate(PROJ):
        Ps = [a[pn] for a in arts]
        rot = rotated_batch(Ps, levels, dev)
        for lv in levels:
            pl = D.SCALE_PLANE[lv]
            suh = torch.stack([P[pl]["suh"] for P in Ps]).to(dev)
            svh = torch.stack([P[pl]["svh"] for P in Ps]).to(dev)
            Ws = dense_batch(rot[lv], suh, svh, batch_had)
            for i, (P, W) in enumerate(zip(Ps, Ws)):
                out[lv][i][j] = D.apply_ocol(P, W, lv)
            del Ws
        del rot
    if perms is not None:
        for lv in levels:
            for i, perm in enumerate(perms):
                if perm is not None:
                    inv = torch.argsort(torch.as_tensor(perm, device=dev))
                    W = out[lv][i]
                    out[lv][i] = [W[0][inv], W[1][inv], W[2][:, inv]]
    return out
