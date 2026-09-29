"""T29 format extension: down-projection input rotation width `in_had_down` in {128, 512} (absent = 128).

in_had_down = 512: the down projection's k side (SwiGLU-output channels, 2048 = 4 TP4 shards x 512) uses one sign +
Sylvester Hadamard-512 block per TP4 shard instead of sign + Hadamard-128 blocks.  The signs are the encoder's usual su
(same seed / RNG order); the n side (hidden, 6144) keeps Had128; the low-rank plane V stays in the un-rotated basis.
Nothing else in the planes changes (ring layout, K, scales, words), so bpw is identical.

Implemented WITHOUT editing the pinned thread-12 encoder / decoder: `k_had(k, width)` scopes a patch of exllamav3's
Hadamard helpers so that a 128-wide transform along a dimension of size k runs at `width` instead; the encoder and
decoder bodies are the pinned ones.  Only the down projection is ever run inside the scope (gate/up have n = 2048 = k).

  decode_expert29(art, level)      = nq_decode.decode_expert, honouring art["meta"]["in_had_down"]
  assemble29(root, L, E)           = nq_layer.assemble + the manifest's config.in_had_down
  encode_down(Wd, HG, width)       = the down iteration of nq_encode.encode_expert (PROD config) under k_had
  refit_down(art, Wd, HG, width)   = shipped artifact with the down projection re-encoded at `width`
"""
import os, sys, json, contextlib
import torch

T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T05 = "/home/coder/git/nestquant/threads/05-exl3-harness"
for p in (T12, T05):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq_decode as D
import nq_encode as NE

FIELD = "in_had_down"
WIDTHS = (128, 512)
_FN = ("preapply_had_l", "preapply_had_r", "blockwise_preapply_had_l_", "blockwise_preapply_had_r_")


def _Q():
    from exllamav3.modules.quant.exl3_lib import quantize as Q
    return Q


@contextlib.contextmanager
def k_had(k, width):
    """Within the scope, every 128-wide Hadamard along a dimension of size k (rows for *_l, columns for *_r) uses
    `width`.  width 128 = no patch."""
    if width == 128:
        yield
        return
    assert width in WIDTHS and k % width == 0, (k, width)
    Q = _Q()
    orig = {f: getattr(Q, f) for f in _FN}

    def wrap(f):
        o = orig[f]; left = "_l" in f

        def g(x, had_dim):
            dim = x.shape[0] if left else x.shape[1]
            return o(x, width if (had_dim == 128 and dim == k) else had_dim)
        return g
    try:
        for f in _FN:
            setattr(Q, f, wrap(f))
        yield
    finally:
        for f, o in orig.items():
            setattr(Q, f, o)


@contextlib.contextmanager
def flat_ics(k, on=True):
    """Encoder-only option (no format change): replace EXL3's per-input-channel scale ics (prep: block_rms(weight, dim=1)
    on the [k, n] weight) by its RMS constant for a projection with input dim k.  The trellis H is rotated with the
    signs only, so a strongly varying ics makes the LDLQ metric wrong; T29 measured this is what the composite
    drot4 arm neutralised (L3 E138 down L2: had512+ics 12.28, had512+flat 8.03, drot4 composite 8.10)."""
    if not on:
        yield
        return
    Q = _Q(); o = Q.block_rms

    def g(x, dim, keepdim=False, blocksize=32):
        r = o(x, dim, keepdim, blocksize)
        if dim == 1 and x.shape[0] == k:
            r = torch.full_like(r, float(r.square().mean().sqrt()))
        return r
    try:
        Q.block_rms = g
        yield
    finally:
        Q.block_rms = o


def width_of(art):
    return int(art.get("meta", {}).get(FIELD, 128))


@torch.no_grad()
def decode_expert29(art, level, device="cuda"):
    w = width_of(art)
    if w == 128:
        return D.decode_expert(art, level, device)            # pinned path, bit-identical by construction
    W = [D.decode_matrix(art[p], level, device) for p in ("gate", "up")]
    with k_had(art["down"]["meta"]["k"], w):
        W.append(D.decode_matrix(art["down"], level, device))
    perm = art.get("meta", {}).get("inter_perm")
    if perm is not None:
        inv = torch.argsort(torch.as_tensor(perm, device=device))
        W = [W[0][inv], W[1][inv], W[2][:, inv]]
    return W


def assemble29(root, L, E):
    import nq_layer as NL
    art = NL.assemble(root, L, E)
    man = json.load(open(f"{root}/L{L}/manifest.json"))
    w = int(man.get("config", {}).get(FIELD, 128))
    art["meta"] = {FIELD: w}
    return art


@torch.no_grad()
def encode_down(Wd, HG, width, count=1, sigma=None, sigma_out=0.03, lam=NE.PROD["lam"], base_var=NE.PROD["base_var"],
                inner=NE.PROD["inner"], seed=91426, lr=NE.PROD["lr"], res_K=None, check=True, flat=False):
    """The pi = 2 (down) iteration of nq_encode.encode_expert (production branch: res_K pattern, single joint pass,
    no ocol) with the k-side Hadamard at `width`.  -> (planes, dense {2, 4}, info)."""
    import harness as h
    import nq_patvit as PV
    sigma = sigma or NE.PROD["sigma"]
    res_K = res_K or NE.PROD["res_K"]
    pn, pi = "down", 2
    H = HG["H"][pi]
    oidx = torch.zeros(0, dtype=torch.long)
    Hq = NE.ocol_H(H, oidx)
    Vlr = None
    if lr:
        Vlr = NE.lr_detect(H.cuda(), **lr)
        Hq = NE.lr_H(H.cuda(), Vlr)
    k = Wd.shape[1]
    with k_had(k, width), flat_ics(k, flat):
        P = NE.prep(Wd, Hq, count, sigma[pn], seed=seed, G=HG["G"][pi], sigma_out=sigma_out,
                    ks=(2, float(res_K[pn])))
        rule = dict(kind="uniform", K=float(res_K[pn]))
        planes, dn, inf, enc, _ = NE.encode_projection(P, shard_axis=NE.PROD["axis"][pn], lam=lam, base_var=base_var,
                                                       inner=inner, res_rule=rule)
        PV.free_tmp(); h.free_scratch()
        if lr and Vlr.shape[0]:
            NE.lr_apply(planes, dn, Wd, Vlr)
            planes["meta"]["lr"] = dict(lr, r=int(Vlr.shape[0]), shared_V=False, nnz=NE.lr_nnz(Vlr))
            inf["bits"] = D.bits_per_level(planes)
        if check:
            rot = D.rotated_levels(planes)
            inf["bitexact"] = {L: bool(torch.equal(D.decode_matrix(planes, L, rot=rot), dn[L])) for L in (2, 4)}
            assert all(inf["bitexact"].values()), inf["bitexact"]
            del rot
    info = {kk: v for kk, v in inf.items() if kk in ("bits", "proxy_rot", "time", "bitexact", "L2_equal_canonical",
                                                    "stream_mismatch", "K_frac")}
    del enc; NE.free(P)
    return planes, {L: dn[L].cpu() for L in (2, 4)}, info


def refit_down(art, Wd, HG, width, **kw):
    """Shipped artifact (gate/up kept as is) with the down projection re-encoded at `width`; meta updated."""
    planes, dense, info = encode_down(Wd, HG, width, **kw)
    out = dict(art)
    out["down"] = planes
    m = dict(art["meta"])
    m["info"] = dict(m["info"], down=info)
    m["lr_rank"] = dict(m.get("lr_rank") or {}, down=int(planes["base"]["lr"]["V"].shape[0]) if "lr" in planes["base"] else 0)
    m["rate"] = sum(m["info"][p]["bits"][4] for p in NE.PROJ) / len(NE.PROJ)
    m[FIELD] = int(width)
    m["t29"] = dict(refit="down input rotation width", in_had_down=int(width), flat_ics=bool(kw.get("flat")),
                    source_gate_up="shipped nq-encode-v1")
    out["meta"] = m
    return out, dense
