"""T35 fine-tune GPU pilot (1 GPU): full-depth, on-policy, decode-rows-only tuning of one NestQuant layer.

Same parameterisation / loss as nq35_ft.py (nq27 ExpertParams su/sv/U/V on the --n-train most-routed experts of
--layer; codes fixed, layout_check), but at FULL depth: layers < L are exact FP8 (precomputed once), L is the trainable
sparse layer (other experts frozen at their real decode, --frozen real, --level 2 = everything cold), layers L+1..77
are FP8 and the logit KL is taken at the real lm_head.  Backprop through the 37 frozen FP8 layers is done layer by layer
by hand (forward no-grad saving each layer input, then recompute + backward per layer in reverse), with the routed FP8
experts streamed (threaded pread into pinned host buffers, prefetch of the next layer in pass order).
Data: fp8dec_run1b retired chains ONLY (ft corpus src=fp8dec, PRIVATE): train = fp8dec-train windows, held-out =
fp8dec-heldout windows (task_id and chain id disjoint, asserted).  Every loss term is on decode rows only:
  w_kl * KL(p_FP8 || p_cand) at decode rows + w_loc * routed-weighted rel L2 of trainable-expert outputs vs FP8 on
  decode-row slots + w_rt * sum over downstream sparse layers of KL(normalised sigmoid router scores) at decode rows.
Sweep: --n-train list x --w-rt list (one process; upstream / frozen part / references computed once).
Per arm, held-out: KL (row mean + per-window), router KL, downstream top-8 overlap with FP8 and FP8 routing mass kept,
and the downstream routing (ids, w, |x|^2, all rows) saved for the jF hit-rate replay (nq35_ftp_jf.py).
Tuned artifacts are re-decoded through nq_decode for the after-numbers; bit-exactness vs the packed serving format is
checked separately (nq35_ftp_check.py).  Nothing leaves the box.
  python nq35_ftp.py --art enc:/tmp/nestquant/35-nq15/enc_b175 --layer 40 --dev cuda:0 --out /tmp/nestquant/35-nq15/ftp
"""
import os, sys, json, time, argparse, threading, queue
from concurrent.futures import ThreadPoolExecutor
T = "/home/coder/git/nestquant/threads"
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--art", required=True)
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--n-win", type=int, default=64, help="train windows (fp8dec-train, spread)")
ap.add_argument("--n-held", type=int, default=0, help="held-out windows (0 = all fp8dec-heldout)")
ap.add_argument("--ft", default="/tmp/nestquant/35-nq15/private/ft")
ap.add_argument("--cache", default="/tmp/nestquant/35-nq15/private/ftp_cache")
ap.add_argument("--n-train", default="4"); ap.add_argument("--w-rt", default="0,0.1,0.3,1.0")
ap.add_argument("--frozen", default="real", choices=["ref", "real"]); ap.add_argument("--level", type=int, default=2)
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tune", default="su,sv,U,V")
ap.add_argument("--w-kl", type=float, default=1.0); ap.add_argument("--w-loc", type=float, default=0.1)
ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--io-threads", type=int, default=8); ap.add_argument("--attn-chunk", type=int, default=16)
ap.add_argument("--head-chunk", type=int, default=1024)
ap.add_argument("--out", default="/tmp/nestquant/35-nq15/ftp")
ap.add_argument("--hot", default="", help="serving mix: nq35_ftp_hot.py hot-set file; hot slots of layer L at real "
                "level 4 (frozen), cold at --level; candidates = most-routed by COLD train decode slots")
ap.add_argument("--bw-chunk", type=int, default=32, help="windows per backward chunk")
ap.add_argument("--eval-every", type=int, default=4, help="held-out KL (params path) every k steps (0 = off)")
ap.add_argument("--patience", type=int, default=0, help="early stop: checkpoints without val gain (0 = off); val = "
                "even held-out windows, best-val params are kept and written")
ap.add_argument("--es-metric", default="held", choices=("held", "val"),
                help="early-stop criterion: held = decode-row-mean KL over all held-out windows; val = even-window mean")
ap.add_argument("--offload", type=int, default=1, help="keep the saved per-layer inputs in host memory")
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
dev = torch.device(a.dev)
torch.cuda.set_device(dev)
os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
LOG = open(f"{a.out}/log", "a")
L = a.layer
kind, _, ROOT = a.art.partition(":")
assert kind == "enc"


def log(m):
    s = f"{time.strftime('%H:%M:%S')} {m}"
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def jdump(o, p):
    json.dump(o, open(p + ".part", "w"), indent=1); os.replace(p + ".part", p)


# ------------------------------------------------------------------ data (fp8dec only, task + chain disjoint)
def windows(split, n):
    meta = [json.loads(x) for x in open(f"{a.ft}/{split}.meta.jsonl")]
    rows = [k for k, m in enumerate(meta) if m["src"] == "fp8dec"]
    if n and n < len(rows):
        rows = [rows[k] for k in np.linspace(0, len(rows) - 1, n).round().astype(int)]
    tok = np.load(f"{a.ft}/{split}.tok.npy", mmap_mode="r"); dec = np.load(f"{a.ft}/{split}.dec.npy", mmap_mode="r")
    t = torch.as_tensor(np.asarray(tok[rows, -a.seq:]), dtype=torch.long)
    m = torch.as_tensor(np.asarray(dec[rows, -a.seq:])).clone()
    m[:, -1] = False
    return t, m, [meta[k] for k in rows]


DATA = {"train": windows("train", a.n_win), "heldout": windows("heldout", a.n_held)}
tr_t = {m["task_id"] for m in DATA["train"][2]}; ho_t = {m["task_id"] for m in DATA["heldout"][2]}
tr_c = {m["id"] for m in DATA["train"][2]}; ho_c = {m["id"] for m in DATA["heldout"][2]}
assert not tr_t & ho_t and not tr_c & ho_c
log(f"data: train {len(tr_c)} windows / {len(tr_t)} tasks, {int(DATA['train'][1].sum())} decode rows; held-out "
    f"{len(ho_c)} windows / {len(ho_t)} tasks, {int(DATA['heldout'][1].sum())} decode rows (task + chain disjoint)")

cfg = E2E.load_config()
assert cfg.mlp_layer_types[L] == "sparse"
NL = cfg.num_hidden_layers
fp8 = nq_io.FP8Model(E2E.FP8_DIR)
bb = E2E.Backbone(cfg, fp8, dev)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding, GlmMoeDsaRMSNorm  # noqa
rotemb = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
pos = torch.arange(a.seq, device=dev).view(1, -1)
cos_sin = rotemb(torch.empty(1, a.seq, 1, device=dev, dtype=torch.bfloat16), pos)
fnorm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
fnorm.load_state_dict({"weight": fp8.tensor("model.norm.weight", dev)}, assign=True)
for p_ in fnorm.parameters():
    p_.requires_grad_(False)
lm_head = fp8.tensor("lm_head.weight", dev).float()
HID = cfg.hidden_size


# ------------------------------------------------------------------ routed-expert streamer
class Streamer:
    """Layer-granular FP8 expert streaming: threaded pread of all 256 experts' (weight, scale_inv) of a layer into a
    pinned host buffer (2 buffers, prefetch of the next layer in pass order), one H2D copy, per-expert dequant."""
    PROJ = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, NE=256):
        self.NE = NE
        self.items = {}
        for li in range(NL):
            if cfg.mlp_layer_types[li] != "sparse":
                continue
            its, off = [], 0
            for e in range(NE):
                for k in self.PROJ:
                    for suf in ("weight", "weight_scale_inv"):
                        nm = f"model.layers.{li}.mlp.experts.{e}.{k}.{suf}"
                        f, dt, shape, o, nb = fp8.idx.map[nm]
                        its.append((e, k, suf, f, dt, shape, o, nb, off)); off += (nb + 255) // 256 * 256
            self.items[li] = (its, off)
        nbmax = max(v[1] for v in self.items.values())
        self.host = [torch.empty(nbmax, dtype=torch.uint8).pin_memory() for _ in range(2)]
        self.gpu = torch.empty(nbmax, dtype=torch.uint8, device=dev)
        self.free = queue.Queue(); [self.free.put(i) for i in range(2)]
        self.pool = ThreadPoolExecutor(a.io_threads)
        self.fut = {}
        self.cur = None
        self.lock = threading.Lock()

    def _read(self, li, hb):
        its, _ = self.items[li]
        buf = memoryview(self.host[hb].numpy())

        def rd(it):
            _, _, _, f, _, _, o, nb, off = it
            fd = fp8.idx._file(f); got = 0
            while got < nb:
                got += os.preadv(fd, [buf[off + got:off + nb]], o + got)
        list(self.pool.map(rd, its))
        return hb

    def prefetch(self, li):
        if li is None or li not in self.items or li in self.fut or li == self.cur:
            return
        hb = self.free.get()
        res = {}

        def job():
            res["hb"] = self._read(li, hb)
        th = threading.Thread(target=job); th.start()
        self.fut[li] = (th, res)

    def _cancel(self, keep):
        for k in [k for k in self.fut if k not in keep]:
            th, res = self.fut.pop(k); th.join(); self.free.put(res["hb"])

    def load(self, li, nxt=None):
        self._cancel((li, nxt))
        if li == self.cur:
            self.prefetch(nxt); return self
        if li not in self.fut:
            self.prefetch(li)
        th, res = self.fut.pop(li)
        th.join()
        hb = res["hb"]
        nb = self.items[li][1]
        self.gpu[:nb].copy_(self.host[hb][:nb], non_blocking=False)
        self.free.put(hb)
        self.cur = li
        self.prefetch(nxt)
        return self

    def expert(self, e):
        its, _ = self.items[self.cur]
        out = {}
        for k_i, k in enumerate(self.PROJ):
            j = (e * 3 + k_i) * 2
            _, _, _, _, dt, sh, _, nb, off = its[j]
            _, _, _, _, dt2, sh2, _, nb2, off2 = its[j + 1]
            w = self.gpu[off:off + nb].view(nq_io.DT[dt]).view(sh)
            s = self.gpu[off2:off2 + nb2].view(nq_io.DT[dt2]).view(sh2)
            out[k] = nq_io.fp8_dequant(w, s)
        return [out["gate_proj"], out["up_proj"], out["down_proj"]]


ST = Streamer()
log(f"streamer: {len(ST.items)} sparse layers, {max(v[1] for v in ST.items.values()) / 2**30:.2f} GiB per layer")


def ffn3(x, W):
    g, u, d = W
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def attn(layer, h):
    outs = []
    for c0 in range(0, h.shape[0], a.attn_chunk):
        hc = h[c0:c0 + a.attn_chunk]; N = hc.shape[0]
        att, _, _ = layer.self_attn(hidden_states=layer.input_layernorm(hc), position_embeddings=cos_sin,
                                    attention_mask=None, position_ids=pos.expand(N, -1),
                                    prev_topk_indices=pos.to(torch.int32).view(1, 1, -1).expand(N, a.seq, a.seq))
        outs.append(hc + att)
    return torch.cat(outs) if len(outs) > 1 else outs[0]


class FP8FFN(torch.autograd.Function):
    """SwiGLU on a streamed FP8 expert; saves only x and re-dequantises the weights in backward (ST must still hold
    the layer), so backward through a layer does not keep 256 dequantised experts alive."""
    @staticmethod
    def forward(ctx, x, e):
        Wg, Wu, Wd = ST.expert(e)
        ctx.save_for_backward(x); ctx.e = e; ctx.li = ST.cur
        return F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        assert ST.cur == ctx.li, (ST.cur, ctx.li)
        Wg, Wu, Wd = ST.expert(ctx.e)
        g = F.linear(x, Wg).float(); u = F.linear(x, Wu).float(); sg = torch.sigmoid(g)
        da = (dy.to(x.dtype) @ Wd).float()
        du = da * (g * sg)
        dg = da * u * (sg * (1 + g * (1 - sg)))
        return dg.to(x.dtype) @ Wg + du.to(x.dtype) @ Wu, None


def moe_fp8(layer, h, dmask=None):
    """FP8 MoE block on streamed experts (ST must hold this layer). -> h_out, router logits [T,NE] fp32, w, i, x"""
    f = h.reshape(-1, HID)
    x = layer.post_attention_layernorm(f)
    logits, w, i = layer.mlp.gate(x)
    out = layer.mlp.shared_experts(x).float()
    for e in torch.unique(i).tolist():
        tk, sl = (i == e).nonzero(as_tuple=True)
        if torch.is_grad_enabled():
            y = FP8FFN.apply(x[tk], e).float()
        else:
            y = ffn3(x[tk], ST.expert(e)).float()
        out.index_add_(0, tk, y * w[tk, sl, None].float())
    return (f.float() + out).to(h.dtype).view_as(h), logits.float(), w, i, x


# ------------------------------------------------------------------ upstream (exact FP8), cached
def build_layer(li):
    layer, sparse = bb.build(li)
    for p_ in layer.parameters():
        p_.requires_grad_(False)
    return layer, sparse


ALL_T = torch.cat([DATA["train"][0], DATA["heldout"][0]])
NTR = DATA["train"][0].shape[0]
key = f"L{L}_s{a.seq}_tr{a.n_win}_ho{a.n_held}"
hkey = key + ("_" + os.path.basename(a.hot)[len(key) + 1:-3] if a.hot else "")
cf = f"{a.cache}/{key}.pt"
if os.path.exists(cf):
    C = torch.load(cf, weights_only=False)
    assert torch.equal(C["tok"], ALL_T)
    hmid = C["hmid"].to(dev)
    log(f"upstream from cache {cf}")
else:
    t0 = time.time()
    with torch.no_grad():
        emb = fp8.tensor("model.embed_tokens.weight", dev)
        h = F.embedding(ALL_T.to(dev), emb).to(torch.bfloat16); del emb
        for li in range(L):
            layer, sparse = build_layer(li)
            h = attn(layer, h)
            if sparse:
                nxt = next((j for j in range(li + 1, L) if j in ST.items), None)
                ST.load(li, nxt)
                h = moe_fp8(layer, h)[0]
            else:
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            del layer
            if li % 10 == 0:
                log(f"upstream L{li} {time.time() - t0:.0f}s")
        layer, _ = build_layer(L)
        hmid = attn(layer, h).reshape(-1, HID).contiguous(); del h, layer
    torch.save(dict(tok=ALL_T, hmid=hmid.cpu()), cf + ".part"); os.replace(cf + ".part", cf)
    log(f"upstream L0..{L - 1} + L{L} attention: {time.time() - t0:.0f}s, cached")
torch.cuda.empty_cache()

LAYERS = {li: build_layer(li) for li in range(L, NL)}
DOWN = list(range(L + 1, NL))
log(f"built layers {L}..{NL - 1} on {dev}; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

# ------------------------------------------------------------------ layer L: routing, frozen part, candidates
layer = LAYERS[L][0]
with torch.no_grad():
    xL = layer.post_attention_layernorm(hmid)
    _, wL, iL = layer.mlp.gate(xL)
    wL = wL.float()
    sharedL = layer.mlp.shared_experts(xL).float()
seqT = a.seq
tok_split = torch.zeros(ALL_T.shape[0] * seqT, dtype=torch.long, device=dev); tok_split[NTR * seqT:] = 1
DM = torch.cat([DATA["train"][1], DATA["heldout"][1]]).reshape(-1).to(dev)      # decode rows, all windows
if a.hot:                                                   # [rows, 256]: expert served at level 4 for that row
    HZ = torch.load(a.hot, weights_only=False)
    assert HZ["layer"] == L and HZ["hot"].shape[0] == ALL_T.shape[0] and HZ["hot"].shape[1] * HZ["G"] == seqT
    HOT = HZ["hot"].to(dev).repeat_interleave(HZ["G"], 1).reshape(-1, 256)
    SH = torch.gather(HOT, 1, iL.long())                        # [rows, 8] slot served hot
    del HOT
else:
    SH = torch.zeros_like(iL, dtype=torch.bool)
_cm = ((tok_split == 0) & DM)[:, None] & ~SH
cnt = torch.bincount(iL[_cm].reshape(-1), minlength=256).cpu()
NTS = sorted({int(x) for x in a.n_train.split(",")})
CAND = [int(e) for e in torch.argsort(cnt, descending=True)[:max(NTS)]]
log(f"L{L} top experts by train decode-row slots: {CAND} ({[int(cnt[e]) for e in CAND]})")


def load_art(e):
    return torch.load(f"{ROOT}/L{L}/experts/E{e}.pt", weights_only=False, map_location="cpu")


def real_W(art, level):
    assert art.get("meta", {}).get("inter_perm") is None
    rot = {p: D.rotated_levels(art[p], dev) for p in T27.PROJ}
    return [D.decode_matrix(art[p], level, dev, rot=rot[p]) for p in T27.PROJ]


ffile = f"{a.cache}/{hkey}_frozen_{a.frozen}{a.level}.pt"
SLOTS = {}
with torch.no_grad():
    if os.path.exists(ffile):
        Z = torch.load(ffile, weights_only=False)
        assert Z["cand"] == CAND
        base_ex, refout = Z["base_ex"].to(dev), Z["refout"].to(dev)
        log("frozen part from cache")
    else:
        t0 = time.time()
        base_ex, refout = sharedL.clone(), sharedL.clone()
        ST.load(L, DOWN[0])
        for e in torch.unique(iL).tolist():
            tk, sl = (iL == e).nonzero(as_tuple=True)
            yr = ffn3(xL[tk], [t.to(xL.dtype) for t in ST.expert(e)]).float() * wL[tk, sl, None]
            refout.index_add_(0, tk, yr)
            hs = SH[tk, sl]
            if a.frozen != "real":
                if e not in CAND:
                    base_ex.index_add_(0, tk, yr)
                continue
            art = load_art(e)
            if hs.any():                                        # hot slots: real level 4 (also for candidates)
                W4 = [t.to(xL.dtype) for t in real_W(art, 4)]
                base_ex.index_add_(0, tk[hs], ffn3(xL[tk[hs]], W4).float() * wL[tk[hs], sl[hs], None])
            if e in CAND or not (~hs).any():
                continue
            Wq = [t.to(xL.dtype) for t in real_W(art, a.level)]
            c_ = ~hs
            base_ex.index_add_(0, tk[c_], ffn3(xL[tk[c_]], Wq).float() * wL[tk[c_], sl[c_], None])
        torch.save(dict(cand=CAND, base_ex=base_ex.cpu(), refout=refout.cpu()), ffile + ".part"); os.replace(ffile + ".part", ffile)
        log(f"frozen part ({a.frozen}, level {a.level}) {time.time() - t0:.0f}s")
    ST.load(L, DOWN[0])
    for e in CAND:
        tk, sl = (iL == e).nonzero(as_tuple=True)
        c_ = ~SH[tk, sl]; tk, sl = tk[c_], sl[c_]                # trainable = cold slots only
        yr = ffn3(xL[tk], [t.to(xL.dtype) for t in ST.expert(e)]).float() * wL[tk, sl, None]
        Wq = [t.to(xL.dtype) for t in real_W(load_art(e), a.level)]
        yq = ffn3(xL[tk], Wq).float() * wL[tk, sl, None]
        SLOTS[e] = dict(tk=tk, pw=wL[tk, sl], yr=yr, yq=yq, ref_W=None)
del sharedL


def split_rows(s):
    """token-row index range of split s in the concatenated set"""
    return (0, NTR * seqT) if s == "train" else (NTR * seqT, ALL_T.shape[0] * seqT)


SPL = {s: split_rows(s) for s in ("train", "heldout")}


def out_L(s, TR, Wfn):
    """MoE output of layer L for split s: frozen + (CAND \\ TR at real) + TR via Wfn(e) (None -> real decode)."""
    r0, r1 = SPL[s]
    out = base_ex[r0:r1].clone()
    loc_n, loc_d = 0., 0.
    for e in CAND:
        S_ = SLOTS[e]; sel = (S_["tk"] >= r0) & (S_["tk"] < r1)
        tk = S_["tk"][sel]
        if e in TR and Wfn is not None:
            W = Wfn(e)
            xe = xL[tk] if W[0].dtype == torch.bfloat16 else xL[tk].float()   # bf16 = the serving/eval path of yq
            y = ffn3(xe, W).float() * S_["pw"][sel][:, None]
            dm = DM[tk]
            loc_n = loc_n + ((y - S_["yr"][sel]).pow(2).sum(-1) * dm).sum()
            loc_d = loc_d + (S_["yr"][sel].pow(2).sum(-1) * dm).sum()
        else:
            y = S_["yq"][sel]
        out.index_add_(0, tk - r0, y)
    loc = loc_n / max(float(loc_d), 1e-30) if isinstance(loc_d, torch.Tensor) else torch.zeros((), device=dev)
    return (hmid[r0:r1].float() + out).to(torch.bfloat16), loc


def dense_down(li, h):
    layer = LAYERS[li][0]
    h = attn(layer, h)
    return h + layer.mlp(layer.post_attention_layernorm(h))


def rnorm(lg):
    p = lg.sigmoid(); return p / p.sum(-1, keepdim=True)


# ------------------------------------------------------------------ references (all-FP8 layer L)
REF = {}


@torch.no_grad()
def forward_down(s, h0, collect=False):
    """no-grad pass L+1..77; -> final h, per-layer inputs (if collect), router info at decode rows + all-row ids"""
    r0, r1 = SPL[s]; dm = DM[r0:r1]
    N = (r1 - r0) // seqT
    h = h0.view(N, seqT, HID)
    saved, rt = [], []
    for j, li in enumerate(DOWN):
        if collect:                                         # layer inputs offloaded to host (GPU memory = backward)
            saved.append(h.cpu() if a.offload else h)
        layer, sparse = LAYERS[li]
        if sparse:
            ST.load(li, next((d for d in DOWN[j + 1:] if d in ST.items), None))
            hn = attn(layer, h)
            h, lg, w, i, x = moe_fp8(layer, hn)
            rt.append(dict(li=li, p=rnorm(lg[dm]), ids=i.to(torch.int16), w=w.half(), xn2=x.float().pow(2).sum(-1)))
        else:
            h = dense_down(li, h)
    return h, saved, rt


def head_lp_chunks(h, dm):
    hd = h.reshape(-1, HID)[dm]
    for c0 in range(0, hd.shape[0], a.head_chunk):
        yield c0, hd[c0:c0 + a.head_chunk]


@torch.no_grad()
def ref_pass(s):
    r0, r1 = SPL[s]; dm = DM[r0:r1]
    h0 = (hmid[r0:r1].float() + refout[r0:r1]).to(torch.bfloat16)
    h, _, rt = forward_down(s, h0)
    lps = []
    for c0, hc in head_lp_chunks(h, dm):
        lps.append(F.log_softmax(fnorm(hc).float() @ lm_head.T, -1).cpu().pin_memory())
    REF[s] = dict(lp=lps, rt=rt)


t0 = time.time()
for s in ("train", "heldout"):
    ref_pass(s)
log(f"FP8 references (train + held-out) {time.time() - t0:.0f}s; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
if not os.path.exists(f"{a.out}/routing/fp8.pt"):
    os.makedirs(f"{a.out}/routing", exist_ok=True)
    _rt = REF["heldout"]["rt"]; _r0, _r1 = SPL["heldout"]
    torch.save(dict(layers=[r["li"] for r in _rt], ids=torch.stack([r["ids"] for r in _rt]).cpu(),
                    w=torch.stack([r["w"] for r in _rt]).cpu(), xn2=torch.stack([r["xn2"] for r in _rt]).cpu(),
                    dec=DM[_r0:_r1].view(-1, seqT).cpu()), f"{a.out}/routing/fp8.pt")


def head_kl(s, h, grad=False):
    """mean KL(p_ref || p_cand) over decode rows; grad: also d(mean KL)/dh (same shape as h). Also per-row KL."""
    r0, r1 = SPL[s]; dm = DM[r0:r1]
    hf = h.reshape(-1, HID)
    rows = dm.nonzero().squeeze(1)
    n = rows.numel()
    g = torch.zeros_like(hf, dtype=torch.float32) if grad else None
    tot, per = 0., []
    for k, c0 in enumerate(range(0, n, a.head_chunk)):
        rr = rows[c0:c0 + a.head_chunk]
        lr = REF[s]["lp"][k].to(dev, non_blocking=True)
        hc = hf[rr].detach().float().requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            lp = F.log_softmax(fnorm(hc.to(h.dtype)).float() @ lm_head.T, -1)
            klr = (lr.exp() * (lr - lp)).sum(-1)
            if grad:
                (klr.sum() / n).backward()
                g[rr] = hc.grad
        tot += float(klr.detach().sum()); per.append(klr.detach())
    return tot / n, torch.cat(per), g


def rt_stats(s, rt):
    ref = REF[s]["rt"]
    r0, r1 = SPL[s]; dm = DM[r0:r1]
    rkl, ov, mass = 0., [], []
    for a_, b_ in zip(ref, rt):
        p0, p1 = a_["p"], b_["p"]
        rkl += float((p0 * (p0.clamp_min(1e-30).log() - p1.clamp_min(1e-30).log())).sum(-1).mean())
        i0, i1 = a_["ids"][dm].long(), b_["ids"][dm].long()
        hit = (i1[:, :, None] == i0[:, None, :])                    # [R, 8(cand), 8(ref)]
        ov.append(float(hit.any(1).float().mean()))
        w0 = a_["w"][dm].float()
        mass.append(float((w0 * hit.any(1)).sum(-1).div(w0.sum(-1)).mean()))
    return rkl, ov, mass


def evaluate(s, TR, Wfn, save=None):
    with torch.no_grad():
        h0, loc = out_L(s, TR, Wfn)
        h, _, rt = forward_down(s, h0)
        kl, per, _ = head_kl(s, h)
    rkl, ov, mass = rt_stats(s, rt)
    r0, r1 = SPL[s]
    N = (r1 - r0) // seqT
    dm = DM[r0:r1].view(N, seqT)
    win = torch.arange(N, device=dev)[:, None].expand(N, seqT)[dm]
    wkl = torch.zeros(N, device=dev, dtype=torch.float64).index_add_(0, win, per.double()) / dm.sum(1).clamp_min(1)
    out = dict(kl=kl, win_kl=wkl.tolist(), rkl=rkl, top8_overlap=float(np.mean(ov)), top8_overlap_layers=ov,
               ref_mass_kept=float(np.mean(mass)), loc=float(loc))
    if save:
        torch.save(dict(layers=[r["li"] for r in rt], ids=torch.stack([r["ids"] for r in rt]).cpu(),
                        w=torch.stack([r["w"] for r in rt]).cpu(), xn2=torch.stack([r["xn2"] for r in rt]).cpu(),
                        dec=dm.cpu()), save)
    return out


# ------------------------------------------------------------------ training step (manual layer-wise backprop)
def train_step(TR, PARAMS, opt, w_rt):
    s = "train"
    r0, r1 = SPL[s]; dm = DM[r0:r1]
    N = (r1 - r0) // seqT
    opt.zero_grad()
    with torch.no_grad():                                   # layer-L graph rebuilt at the end (not alive downstream)
        hd, _ = out_L(s, TR, lambda e: PARAMS[e].W_all(a.level))
    with torch.no_grad():
        h, saved, _ = forward_down(s, hd, collect=True)
    kl, _, g = head_kl(s, h, grad=True)
    g = (a.w_kl * g).to(torch.float32); del h, hd
    rkl_tot = 0.
    ref_rt = {r["li"]: r for r in REF[s]["rt"]}
    ndm = int(dm.sum()); cdm = torch.cat([dm.new_zeros(1, dtype=torch.long), dm.view(N, seqT).sum(1).cumsum(0)]).tolist()
    for j in range(len(DOWN) - 1, -1, -1):
        li = DOWN[j]
        hall = saved.pop()
        layer, sparse = LAYERS[li]
        if sparse:
            prv = next((d for d in reversed(DOWN[:j]) if d in ST.items), None)
            ST.load(li, prv)
        gn = torch.empty_like(g)
        for c0 in range(0, N, a.bw_chunk):                   # window chunks: backward memory ~ bw_chunk windows
            c1 = min(N, c0 + a.bw_chunk); q0, q1 = c0 * seqT, c1 * seqT
            hin = hall[c0:c1].to(dev).detach().requires_grad_(True)
            gc = g[q0:q1]
            if sparse:
                hn = attn(layer, hin)
                ho, lg, _, _, _ = moe_fp8(layer, hn)
                p0 = ref_rt[li]["p"][cdm[c0]:cdm[c1]]; p1 = rnorm(lg[dm[q0:q1]])
                rkl = (p0 * (p0.clamp_min(1e-30).log() - p1.clamp_min(1e-30).log())).sum(-1).sum() / ndm
                rkl_tot += float(rkl)
                outs, grads = [ho], [gc.view_as(ho).to(ho.dtype)]
                if w_rt > 0:
                    outs.append(w_rt * rkl); grads.append(None)
                torch.autograd.backward(outs, grads)
            else:
                ho = dense_down(li, hin)
                torch.autograd.backward([ho], [gc.view_as(ho).to(ho.dtype)])
            gn[q0:q1] = hin.grad.float().reshape(-1, HID)
            del hin, ho
        g = gn; del hall
    h0, loc = out_L(s, TR, lambda e: PARAMS[e].W_all(a.level))
    outs, grads = [h0], [g.view_as(h0).to(h0.dtype)]
    if a.w_loc > 0 and loc.requires_grad:
        outs.append(a.w_loc * loc); grads.append(None)
    torch.autograd.backward(outs, grads)
    opt.step()
    return dict(kl=kl, rkl=rkl_tot, loc=float(loc), loss=a.w_kl * kl + a.w_loc * float(loc) + w_rt * rkl_tot)


T27.ExpertParams.W_all = lambda self, lv: [self.W(p, lv) for p in T27.PROJ]

# ------------------------------------------------------------------ sweep
RES_F = f"{a.out}/summary.json"
RES = json.load(open(RES_F)) if os.path.exists(RES_F) else dict(args=vars(a), arms={})
RES["args"] = vars(a); RES["cand"] = CAND; RES["cand_slots"] = [int(cnt[e]) for e in CAND]
_dm = DM[:, None].expand_as(SH)
RES["hot"] = dict(file=a.hot, hot_slot_frac_dec=float(SH[_dm].float().mean()),
                  cand_cold_frac_dec={int(e): float((~SH[(iL == e) & _dm]).float().mean()) for e in CAND})
RES["data"] = dict(train_windows=len(tr_c), train_tasks=len(tr_t), train_dec=int(DATA["train"][1].sum()),
                   held_windows=len(ho_c), held_tasks=len(ho_t), held_dec=int(DATA["heldout"][1].sum()))
RD = f"{a.out}/routing"; os.makedirs(RD, exist_ok=True)


def arm(name, fn):
    if name in RES["arms"]:
        log(f"{name}: done (cached)"); return RES["arms"][name]
    t0 = time.time()
    r = fn(); r["s"] = round(time.time() - t0)
    RES["arms"][name] = r; jdump(RES, RES_F)
    h = r.get("heldout", r)
    log(f"{name}: held-out KL {h['kl']:.6f} rkl {h['rkl']:.4f} top8 {h['top8_overlap']:.4f} mass {h['ref_mass_kept']:.4f}"
        + (f" | train KL {r['train']['kl']:.6f}" if "train" in r else "") + f" ({r['s']}s)")
    return r


arm("allreal", lambda: dict(train=evaluate("train", [], None),
                            heldout=evaluate("heldout", [], None, save=f"{RD}/allreal.pt")))
for nt in NTS:
    TR = CAND[:nt]
    FPW = {}

    def fp8W(e):
        if e not in FPW:
            ST.load(L)
            FPW[e] = ST.expert(e)                       # bf16 = exactly the yr path
        return FPW[e]
    arm(f"fp8ceil_n{nt}", lambda: dict(train=evaluate("train", TR, fp8W),
                                       heldout=evaluate("heldout", TR, fp8W, save=f"{RD}/fp8ceil_n{nt}.pt")))
    FPW.clear(); torch.cuda.empty_cache()
    for wrt in [float(x) for x in a.w_rt.split(",")]:
        name = f"n{nt}_wrt{wrt:g}"
        if name in RES["arms"]:
            log(f"{name}: done (cached)"); continue
        ARTS = {e: load_art(e) for e in TR}
        PARAMS = {e: T27.ExpertParams(ARTS[e], dev, tune=tuple(a.tune.split(","))) for e in TR}
        with torch.no_grad():
            cons = {e: max(float((x - y).norm() / x.norm())
                           for x, y in zip(real_W(ARTS[e], a.level), PARAMS[e].W_all(a.level))) for e in TR}
        opt = torch.optim.Adam([p_ for e in TR for p_ in PARAMS[e].parameters() if p_.requires_grad], lr=a.lr)
        steps = []
        t0 = time.time()
        b0 = np.array(RES["arms"]["allreal"]["heldout"]["win_kl"])
        crit0 = RES["arms"]["allreal"]["heldout"]["kl"] if a.es_metric == "held" else float(b0[0::2].mean())
        best = dict(val=crit0, step=-1,
                    sd={e: {k: v.detach().clone() for k, v in PARAMS[e].state_dict().items()} for e in TR})
        bad = 0
        for st in range(a.steps):
            info = train_step(TR, PARAMS, opt, wrt)
            if a.eval_every and ((st + 1) % a.eval_every == 0 or st + 1 == a.steps):
                with torch.no_grad():
                    Wp = {e: [t.to(torch.bfloat16) for t in PARAMS[e].W_all(a.level)] for e in TR}
                ev = evaluate("heldout", TR, lambda e: Wp[e]); del Wp
                wk = np.array(ev["win_kl"]); val = float(wk[0::2].mean())
                info.update(held_kl=ev["kl"], held_rkl=ev["rkl"], held_win_kl=ev["win_kl"], val=val,
                            val_d=val - float(b0[0::2].mean()), test_d=float(wk[1::2].mean() - b0[1::2].mean()))
                crit = ev["kl"] if a.es_metric == "held" else val
                if crit < best["val"]:
                    best = dict(val=crit, step=st, sd={e: {k: v.detach().clone() for k, v in PARAMS[e].state_dict().items()}
                                                       for e in TR}); bad = 0
                else:
                    bad += 1
            info.update(step=st, s=round(time.time() - t0, 1)); steps.append(info)
            log(f"{name} step {st} " + str({k: v for k, v in info.items() if k != "held_win_kl"}))
            if a.patience and bad >= a.patience:
                log(f"{name}: early stop at step {st} (best val step {best['step']})"); break
        if a.patience:
            for e in TR:
                PARAMS[e].load_state_dict(best["sd"][e])
            log(f"{name}: restored best-val params of step {best['step']} (val {best['val']:.6f})")
        best_step = best["step"]; del best
        od = f"{a.out}/{name}/L{L}"; os.makedirs(od, exist_ok=True)
        tuned = {}
        for e in TR:
            new = PARAMS[e].write(ARTS[e]); T27.layout_check(ARTS[e], new)
            torch.save(new, f"{od}/E{e}.pt.part"); os.replace(f"{od}/E{e}.pt.part", f"{od}/E{e}.pt")
            re = torch.load(f"{od}/E{e}.pt", weights_only=False, map_location="cpu"); T27.layout_check(ARTS[e], re)
            tuned[e] = [t.to(torch.bfloat16) for t in real_W(re, a.level)]   # same cast as yq (before)
        del PARAMS, opt; torch.cuda.empty_cache()
        arm(name, lambda: dict(train=evaluate("train", TR, lambda e: tuned[e]),
                               heldout=evaluate("heldout", TR, lambda e: tuned[e], save=f"{RD}/{name}.pt"),
                               steps=steps, consistency=cons, train_s=round(time.time() - t0), best_step=best_step))
        del tuned; torch.cuda.empty_cache()
log("done")
