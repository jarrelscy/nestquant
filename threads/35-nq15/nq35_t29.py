"""T35 port of the T29 down-input rotation (threads/29-outlier-gap/nq29_had.py) into the b175 campaign, for the layers the
shipped v1 refit (L3-L6: down = sign + Had512 input rotation + flat EXL3 ics; gate/up and everything else unchanged).

  install_encoder(L)   wrap nq_encode.encode_expert: for a T29 layer the DOWN iteration (prep -> encode_projection ->
                       lr_apply -> bit-exact check) runs under nq29_had.k_had(2048, 512) + flat_ics(2048), exactly the
                       scope of nq29_had.encode_down; art meta gets in_had_down = 512.  No-op for other layers.
  decode_expert(art, level, device)  nq_decode.decode_expert honouring meta in_had_down (= nq29_had.decode_expert29,
                       but bound to the ORIGINAL nq_decode.decode_expert so it can be installed over it).
  install_decoder()    D.decode_expert := decode_expert (identical for in_had_down 128 / absent).
  patch_manifest(root, L)  fin29's manifest fields (config.in_had_down/ics_down/had_sign_seed, rotation, campaign.t29).
  down_scope(art)      context for any other decode of art["down"] (D.decode_matrix / rotated_levels are unaffected by
                       the k-side width, only dense_from_rotated is).
"""
import os, sys, json, time, contextlib
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T29 = "/home/coder/git/nestquant/threads/29-outlier-gap"
for p in (T12, T29):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch                 # noqa: E402
import nq_decode as D        # noqa: E402
import nq29_had as NH        # noqa: E402

LAYERS = (3, 4, 5, 6)
WIDTH = 512
FIELD = NH.FIELD
_dec_orig = D.decode_expert


def is_t29(L):
    return int(L) in LAYERS


def down_scope(art):
    w = int(art.get("meta", {}).get(FIELD, 128))
    return NH.k_had(art["down"]["meta"]["k"], w)


@torch.no_grad()
def decode_expert(art, level, device="cuda"):
    w = int(art.get("meta", {}).get(FIELD, 128))
    if w == 128:
        return _dec_orig(art, level, device)
    W = [D.decode_matrix(art[p], level, device) for p in ("gate", "up")]
    with NH.k_had(art["down"]["meta"]["k"], w):
        W.append(D.decode_matrix(art["down"], level, device))
    perm = art.get("meta", {}).get("inter_perm")
    if perm is not None:
        inv = torch.argsort(torch.as_tensor(perm, device=device))
        W = [W[0][inv], W[1][inv], W[2][:, inv]]
    return W


def install_decoder():
    D.decode_expert = decode_expert


def install_encoder(L):
    if not is_t29(L):
        return False
    import nq_encode as NE
    enc0 = NE.encode_expert

    def encode_expert(Ws, HG, **kw):
        prep0 = NE.prep
        stack = contextlib.ExitStack()
        k_down = Ws[2].shape[1]

        def prep(W, *a, **k):
            if W is Ws[2]:                                   # the down iteration (pi = 2, last): open the T29 scope
                stack.enter_context(NH.k_had(k_down, WIDTH))
                stack.enter_context(NH.flat_ics(k_down, True))
            return prep0(W, *a, **k)
        NE.prep = prep
        try:
            art, dense = enc0(Ws, HG, **kw)
        finally:
            stack.close()
            NE.prep = prep0
        art["meta"][FIELD] = WIDTH
        art["meta"]["t29"] = dict(refit="down input rotation width", in_had_down=WIDTH, flat_ics=True,
                                  port="threads/35-nq15/nq35_t29.py")
        return art, dense
    NE.encode_expert = encode_expert
    return True


def patch_manifest(root, L):
    p = f"{root}/L{L}/manifest.json"
    man = json.load(open(p))
    man["config"].update(in_had_down=WIDTH, ics_down="flat", had_sign_seed=91426)
    man["rotation"] = dict(
        down_in=dict(had=WIDTH, blocks="4 x 512 over k = 2048 = one block per TP4 rank (TP8 shard pair 2s, 2s+1)",
                     signs="random signs folded into suh2/suh4 exactly as for Had128 (encoder seed 91426, same RNG order)",
                     decode="W = diag(svh) Had128_n^T ... : nq_decode.dense_from_rotated with the k-side Hadamard at 512 "
                            "(threads/29-outlier-gap/nq29_had.decode_expert29)"),
        down_out=128, gate_up_in=128, gate_up_out=128,
        encoder="down: sign + Had512 input rotation, EXL3 input-channel scale (ics) replaced by its RMS (flat); "
                "gate/up: Had128 (T35 b175 campaign, nq15 base)")
    camp = dict(man.get("campaign") or {})
    camp["t29"] = dict(refit="L3-L6 down in_had_down 512 + flat ics (T29 recipe, encoded in-campaign)",
                       code="threads/35-nq15/nq35_t29.py + threads/29-outlier-gap/nq29_had.py",
                       time=time.strftime("%Y-%m-%d %H:%M:%S"))
    man["campaign"] = camp
    json.dump(man, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
