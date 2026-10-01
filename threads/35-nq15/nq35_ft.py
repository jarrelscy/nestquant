"""T35 Stage 1 §5 fine-tune harness (smoke): end-to-end tuning of NestQuant continuous expert parameters.

Model truncated to layers [0, --end) + final norm + fp32 lm_head.  Layers < --layer are FP8 (exact), --layer is the
trainable sparse layer, layers in (--layer, --end) are FP8 downstream (dense or sparse; their routers give the router
KL).  In --layer, the --n-train most-routed experts (on the train windows) get trainable nq27 ExpertParams (su/sv
scale multipliers + low-rank U/V deltas, format unchanged) initialised from their artifact; every other routed expert
is frozen (--frozen ref: FP8, or real: its artifact's decode at --level).  Because upstream is exact and attention is
frozen, the MoE input / routing of --layer are constant, so the frozen part (shared + frozen experts) is precomputed once.
Loss (decode rows only, i.e. tokens the model generated; dec mask of ft_corpus.py):
  w_kl * KL(p_FP8 || p_cand) + w_loc * routed-weighted rel L2 of the trainable experts' outputs vs FP8
  + w_rt * KL of normalised router sigmoid scores at downstream sparse layers.
Then: write the tuned artifacts (nq27 ExpertParams.write, layout_check), RE-DECODE them through nq_decode (fp16
rounding, the real serving decode) and report KL before / after on train and held-out windows.
  python nq35_ft.py --art enc:/tmp/nestquant/35-nq15/enc_b175 | v1:/tmp/nestquant/nq-encode-v1 --layer 3 --end 4 ...
CPU by default (smoke); --dev cuda:0 must run under gpu.lock.  Windows come from the PRIVATE ft corpus; nothing here
leaves the box.  Outputs: --out/{summary.json, L{L}/E{E}.pt (tuned), log}.
"""
import os, sys, json, time, argparse, math
T = "/home/coder/git/nestquant/threads"
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--art", required=True, help="enc:ROOT (E{E}.pt) or v1:ROOT (tp safetensors, nq25_st.assemble)")
ap.add_argument("--layer", type=int, default=3); ap.add_argument("--end", type=int, default=4)
ap.add_argument("--seq", type=int, default=256); ap.add_argument("--n-win", type=int, default=2)
ap.add_argument("--n-held", type=int, default=2); ap.add_argument("--ft", default="/tmp/nestquant/35-nq15/private/ft")
ap.add_argument("--n-train", type=int, default=4); ap.add_argument("--experts", default="")
ap.add_argument("--frozen", default="ref", choices=["ref", "real"]); ap.add_argument("--level", type=int, default=2)
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tune", default="su,sv,U,V")
ap.add_argument("--w-kl", type=float, default=1.0); ap.add_argument("--w-loc", type=float, default=0.1)
ap.add_argument("--w-rt", type=float, default=0.1)
ap.add_argument("--dev", default="cpu"); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--out", default="/tmp/nestquant/35-nq15/ft_smoke")
a = ap.parse_args()
os.environ["NQ_SEQ"] = str(a.seq)
os.environ.setdefault("NQ27_DECODE_DEV", a.dev)
for p in (f"{T}/05-exl3-harness", "/home/coder/git/orbit-duet", f"{T}/25-campaign", f"{T}/27-pv-tune",
          f"{T}/18-e2e-eval", f"{T}/12-reference-encoder", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np                  # noqa: E402
import torch                        # noqa: E402
import torch.nn.functional as F     # noqa: E402
import nq15                         # noqa: E402,F401  base-K decoder
import nq_decode as D               # noqa: E402
import nq27_tune as T27             # noqa: E402
import nq_e2e as E2E                # noqa: E402
import nq_io                        # noqa: E402

torch.set_num_threads(a.threads)
dev = a.dev
os.makedirs(a.out, exist_ok=True)
LOG = open(f"{a.out}/log", "a")


def log(m):
    s = f"{time.strftime('%H:%M:%S')} {m}"
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


kind, _, ROOT = a.art.partition(":")


def had_width(L, art):
    if kind == "enc":
        return int(art.get("meta", {}).get("in_had_down", 128))
    return int(json.load(open(f"{ROOT}/L{L}/manifest.json")).get("config", {}).get("in_had_down", 128))


def load_art(L, e):
    if kind == "enc":
        f = f"{ROOT}/L{L}/experts/E{e}.pt"
        return torch.load(f, weights_only=False, map_location="cpu") if os.path.exists(f) else None
    import nq25_st as ST
    return ST.assemble(ROOT, L, e)


def unperm(art, W):
    perm = art.get("meta", {}).get("inter_perm")
    if perm is None:
        return W
    inv = torch.argsort(torch.as_tensor(perm, device=W[0].device))
    return [W[0][inv], W[1][inv], W[2][:, inv]]


def real_W(art, level):
    rot = {p: D.rotated_levels(art[p], dev) for p in T27.PROJ}
    return unperm(art, [D.decode_matrix(art[p], level, dev, rot=rot[p]) for p in T27.PROJ])


def ffn3(x, W):
    g, u, d = W
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


# ------------------------------------------------------------------ data
def windows(split, n):
    tok = np.load(f"{a.ft}/{split}.tok.npy", mmap_mode="r"); dec = np.load(f"{a.ft}/{split}.dec.npy", mmap_mode="r")
    pick = np.linspace(0, tok.shape[0] - 1, n).round().astype(int)      # spread over sources/tasks
    t = torch.as_tensor(np.asarray(tok[pick, -a.seq:]), dtype=torch.long)
    m = torch.as_tensor(np.asarray(dec[pick, -a.seq:]))
    m[:, -1] = False                                                     # last row predicts outside the window
    return t, m


cfg = E2E.load_config()
assert cfg.mlp_layer_types[a.layer] == "sparse"
fp8 = nq_io.FP8Model(E2E.FP8_DIR)
bb = E2E.Backbone(cfg, fp8, dev)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding, GlmMoeDsaRMSNorm  # noqa
rotemb = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
pos = torch.arange(a.seq, device=dev).view(1, -1)
cos_sin = rotemb(torch.empty(1, a.seq, 1, device=dev, dtype=torch.bfloat16), pos)
emb = fp8.tensor("model.embed_tokens.weight", dev)
fnorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
fnorm.load_state_dict({"weight": fp8.tensor("model.norm.weight", dev)}, assign=True)
lm_head = fp8.tensor("lm_head.weight", dev).float()
layers = {}
for li in range(a.end):
    t0 = time.time()
    layers[li] = bb.build(li)
    for p_ in layers[li][0].parameters():
        p_.requires_grad_(False)
log(f"built layers 0..{a.end - 1} on {dev}")


def attn(layer, h):
    N = h.shape[0]
    att, _, _ = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=cos_sin,
                                attention_mask=None, position_ids=pos.expand(N, -1),
                                prev_topk_indices=pos.to(torch.int32).view(1, 1, -1).expand(N, a.seq, a.seq))
    return h + att


def route(layer, x):
    logits, w, i = layer.mlp.gate(x)
    return logits, w, i


DS_CACHE = {}


def ref_expert(li, e, cache=True):
    if (li, e) not in DS_CACHE:
        W = fp8.expert(li, e, dev)
        W = [W["gate_proj"], W["up_proj"], W["down_proj"]]
        if not cache:
            return W
        DS_CACHE[li, e] = W
    return DS_CACHE[li, e]


def moe_full(layer, li, h, rt_out=None, cache=True):
    """FP8 MoE block (frozen weights, differentiable in h). rt_out: list to append router logits."""
    N = h.shape[0]
    f = h.reshape(N * a.seq, -1)
    x = layer.post_attention_layernorm(f)
    logits, w, i = route(layer, x)
    if rt_out is not None:
        rt_out.append(logits.float())
    out = layer.mlp.shared_experts(x).float()
    for e in torch.unique(i).tolist():
        tk, sl = (i == e).nonzero(as_tuple=True)
        y = ffn3(x[tk], [t.to(x.dtype) for t in ref_expert(li, e, cache)]).float()
        out = out.index_add(0, tk, y * w[tk, sl, None].float())
    return (f.float() + out).to(h.dtype).view_as(h)


def down(h, li0, rt_out=None):
    for li in range(li0, a.end):
        layer, sparse = layers[li]
        h = attn(layer, h)
        if sparse:
            h = moe_full(layer, li, h, rt_out)
        else:
            h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h


def logp(h, m):
    z = fnorm(h)[m].float() @ lm_head.T
    return F.log_softmax(z, -1)


# ------------------------------------------------------------------ per-set precompute at the trainable layer
L = a.layer


@torch.no_grad()
def prep(split, n):
    tok, m = windows(split, n)
    h = F.embedding(tok.to(dev), emb).to(torch.bfloat16)
    for li in range(L):
        layer, sparse = layers[li]
        h = attn(layer, h)
        if sparse:                                                       # exact FP8 upstream, experts streamed
            h = moe_full(layer, li, h, cache=False)
        else:
            h = h + layer.mlp(layer.post_attention_layernorm(h))
    layer = layers[L][0]
    h = attn(layer, h)
    f = h.reshape(-1, h.shape[-1])
    x = layer.post_attention_layernorm(f)
    logits, w, i = route(layer, x)
    shared = layer.mlp.shared_experts(x).float()
    return dict(split=split, tok=tok, m=m.to(dev), hmid=f, x=x, w=w.float(), i=i, shared=shared)


log("prep train / held-out sets")
S = {"train": prep("train", a.n_win), "heldout": prep("heldout", a.n_held)}
cnt = torch.bincount(S["train"]["i"].reshape(-1).cpu(), minlength=256)
if a.experts:
    TR = [int(e) for e in a.experts.split(",")]
else:
    TR = [int(e) for e in torch.argsort(cnt, descending=True)[:a.n_train]]
log(f"trainable experts L{L} {TR} (train slots {[int(cnt[e]) for e in TR]}); frozen={a.frozen} level={a.level}")

ARTS, PARAMS = {}, {}
for e in TR:
    art = load_art(L, e)
    assert art is not None, f"no artifact L{L} E{e}"
    assert had_width(L, art) == 128, f"L{L}: in_had_down={had_width(L, art)} (T29) - ExpertParams/nq_decode model Had128 only"
    ARTS[e] = art
    PARAMS[e] = T27.ExpertParams(art, dev, tune=tuple(a.tune.split(",")))


def params_W(e, lv):
    W = [PARAMS[e].W(p, lv) for p in T27.PROJ]
    return unperm(ARTS[e], W)


# consistency: ExpertParams model at init vs the real decoder
cons = {}
with torch.no_grad():
    for e in TR:
        Wr, Wp = real_W(ARTS[e], a.level), params_W(e, a.level)
        cons[e] = max(float((x - y).norm() / x.norm()) for x, y in zip(Wr, Wp))
log(f"ExpertParams vs nq_decode rel diff (init, level {a.level}): {cons}")


@torch.no_grad()
def frozen_part(s):
    """shared + all experts not in TR (ref or real decode) -> [T,H] fp32; the ref MoE output; the TR slot lists."""
    x, w, i = s["x"], s["w"], s["i"]
    base = s["shared"].clone()
    refout = s["shared"].clone()
    slots = {}
    for e in torch.unique(i).tolist():
        tk, sl = (i == e).nonzero(as_tuple=True)
        yr = ffn3(x[tk], [t.to(x.dtype) for t in ref_expert(L, e)]).float() * w[tk, sl, None]
        refout.index_add_(0, tk, yr)
        if e in TR:
            slots[e] = (tk, w[tk, sl], yr)
            continue
        if a.frozen == "real":
            art = load_art(L, e)
            Wq = [t.to(x.dtype) for t in real_W(art, a.level)] if art is not None else None
            y = ffn3(x[tk], Wq).float() * w[tk, sl, None] if Wq is not None else yr
        else:
            y = yr
        base.index_add_(0, tk, y)
    return base, refout, slots


for s in S.values():
    s["base"], s["refout"], s["slots"] = frozen_part(s)
    with torch.no_grad():
        N = s["tok"].shape[0]
        h = (s["hmid"].float() + s["refout"]).to(torch.bfloat16).view(N, a.seq, -1)
        rt = []
        h = down(h, L + 1, rt)
        s["ref_lp"] = logp(h, s["m"])
        s["ref_rt"] = rt
    log(f"{s['split']}: {int(s['m'].sum())} decode rows, {len(s['slots'])} trainable experts routed")
DS_FROZEN = dict(DS_CACHE)


def cand_forward(s, Wfn, grad=True):
    x = s["x"]
    out = s["base"]
    loc_n = loc_d = 0.
    for e, (tk, pw, yr) in s["slots"].items():
        xe = x[tk].float()
        y = ffn3(xe, Wfn(e)) * pw[:, None]
        out = out.index_add(0, tk, y)
        loc_n = loc_n + (y - yr).pow(2).sum(); loc_d = loc_d + yr.pow(2).sum()
    N = s["tok"].shape[0]
    h = (s["hmid"].float() + out).to(torch.bfloat16).view(N, a.seq, -1)
    rt = []
    h = down(h, L + 1, rt)
    lp = logp(h, s["m"])
    kl = (s["ref_lp"].exp() * (s["ref_lp"] - lp)).sum(-1).mean()
    rkl = torch.zeros((), device=dev)
    for r0, r1 in zip(s["ref_rt"], rt):
        p0 = r0.sigmoid(); p0 = p0 / p0.sum(-1, keepdim=True)
        p1 = r1.sigmoid(); p1 = p1 / p1.sum(-1, keepdim=True)
        rkl = rkl + (p0 * (p0.clamp_min(1e-30).log() - p1.clamp_min(1e-30).log())).sum(-1).mean()
    loc = loc_n / max(float(loc_d), 1e-30) if s["slots"] else torch.zeros((), device=dev)
    loss = a.w_kl * kl + a.w_loc * loc + a.w_rt * rkl
    return loss, dict(kl=float(kl), loc=float(loc), rkl=float(rkl))


def score(tag, Wfn):
    with torch.no_grad():
        return {k: cand_forward(s, Wfn)[1] for k, s in S.items()}


res = dict(args=vars(a), trainable=TR, consistency=cons, steps=[])
real0 = {e: [t.float() for t in real_W(ARTS[e], a.level)] for e in TR}
res["before_real"] = score("before", lambda e: real0[e])
res["fp8_ceiling"] = score("fp8", lambda e: [t.float() for t in ref_expert(L, e)])
log(f"before (real decode): {res['before_real']}  | trainable experts at FP8: {res['fp8_ceiling']}")
opt = torch.optim.Adam([p_ for e in TR for p_ in PARAMS[e].parameters() if p_.requires_grad], lr=a.lr)
t0 = time.time()
for step in range(a.steps):
    opt.zero_grad()
    loss, info = cand_forward(S["train"], lambda e: params_W(e, a.level))
    loss.backward()
    opt.step()
    info.update(step=step, loss=float(loss), s=round(time.time() - t0, 1))
    res["steps"].append(info)
    log(f"step {step} {info}")
    if len(DS_CACHE) != len(DS_FROZEN):
        log(f"note: downstream routing reached {len(DS_CACHE) - len(DS_FROZEN)} new FP8 experts (cached)")
        DS_FROZEN = dict(DS_CACHE)

od = f"{a.out}/L{L}"
os.makedirs(od, exist_ok=True)
tuned = {}
for e in TR:
    new = PARAMS[e].write(ARTS[e]); T27.layout_check(ARTS[e], new)
    torch.save(new, f"{od}/E{e}.pt.part"); os.replace(f"{od}/E{e}.pt.part", f"{od}/E{e}.pt")
    re = torch.load(f"{od}/E{e}.pt", weights_only=False, map_location="cpu"); T27.layout_check(ARTS[e], re)
    tuned[e] = [t.float() for t in real_W(re, a.level)]
res["after_real"] = score("after", lambda e: tuned[e])
log(f"after (tuned, re-decoded through nq_decode): {res['after_real']}")
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)
log("done")
