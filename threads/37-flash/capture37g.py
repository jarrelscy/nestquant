"""T37: 8-GPU layer-sequential capture of GLM-5.3-Flash (FP8 checkpoint) in T19's stats schema (nestquant-19-stats-v2,
D=4096, F=2048, 288 experts), so nq19_load.Capture / nq26_blend.BlendCapture / nq_layer read it with only the
dimension constants patched.

    torchrun --nproc-per-node 8 capture37g.py --corpus txt --out /tmp/nestquant/37-flash/cap-txt
    torchrun --nproc-per-node 8 capture37g.py --corpus mm  --out /tmp/nestquant/37-flash/cap-mm

Token-parallel: rank r owns whole windows (round-robin) and runs every layer over them.  The 4-stream mHC hidden
state lives in pinned host RAM.  Per layer: attention per segment (segment-local, positions restart; glmfmt packing),
then the MoE half batched over CHUNK tokens with the T19 sums accumulated on the GPU.  At layer end the per-expert
sums are reduced to the owner rank of each 36-expert block, which writes its packed rows; rank 0 writes the per-layer
files.  Differences from T19 (documented in meta.json["protocol"]): grams are fp32 GEMMs with TF32 on bf16-exact
inputs (p^2-weighted side as a bf16 hi/lo split), gdiag uses bf16 gx/ux on all rows, and the Flash SwiGLU clamps
(gate <= 10, |up| <= 10) are part of the teacher (clamped units get zero output-weight gradient).
Outputs (ROOT): stats/L{L} -> L{L}.v1, eval/val/layer_{L}.pt, final/ (val top-64 teacher logprobs, CE),
PRIVATE trace/ (per-token ids/w/xn per rank, never uploaded)."""
import argparse, json, os, re, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)
from transformers import AutoConfig
from transformers.models.glm5_next import modeling_glm5_next as M

CK = "/tmp/nestquant/37-flash/fp8"
CORP = "/tmp/nestquant/corpus/glm53_calib_glmfmt_v1"
MM = "/tmp/nestquant/19-capture-mm/corpus/c2048_mm"
IMG = "/tmp/nestquant/calib-mm"
PFX = "model.language_model."
D, FF, NE, TOPK = 4096, 2048, 288, 8
CTX_SEED = 20261005
SAL_CATS = ["all"] + [f"{k}/{b}" for k in ("think", "end") for b in ("d1", "d2_4", "d5_16", "d17_32")]
BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 32))
IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_F = {}


def npk(n):
    return n * (n + 1) // 2


def stride_bytes(n):
    return (npk(n) * 4 + 4095) // 4096 * 4096


def raw(k, dev):
    fn = IDX[k]
    if fn not in _F:
        _F[fn] = safe_open(f"{CK}/{fn}", "pt", device="cpu")
    return _F[fn].get_tensor(k).to(dev, non_blocking=True)


def get(k, dev, dtype=torch.bfloat16):
    w = raw(k, dev)
    ks = k[: -len("weight")] + "weight_scale_inv" if k.endswith("weight") else None
    if ks in IDX:
        s = raw(ks, dev).float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        return (w.float() * s).to(dtype)
    return w.to(dtype) if w.is_floating_point() and w.dtype != torch.float32 else w


RENAME = [(r"self_attn\.(f_a_proj|f_b_proj|dt_bias|A_log)", r"self_attn.forget_gate.\1"),
          (r"hc_attn_(fn|base|scale)", r"attn_hc.\1"), (r"hc_ffn_(fn|base|scale)", r"ffn_hc.\1")]


class Experts:
    """FP8 routed experts of one layer on the GPU, dequantised (bf16, = HF Experts weights) one expert at a time."""

    def __init__(self, L, dev):
        p = f"{PFX}layers.{L}.mlp.experts."
        self.gu = torch.empty(NE, 2 * FF, D, dtype=torch.float8_e4m3fn, device=dev)
        self.gus = torch.empty(NE, 2 * FF // 128, D // 128, device=dev)
        self.dn = torch.empty(NE, D, FF, dtype=torch.float8_e4m3fn, device=dev)
        self.dns = torch.empty(NE, D // 128, FF // 128, device=dev)
        for e in range(NE):
            self.gu[e, :FF] = raw(f"{p}{e}.gate_proj.weight", dev); self.gu[e, FF:] = raw(f"{p}{e}.up_proj.weight", dev)
            self.gus[e, : FF // 128] = raw(f"{p}{e}.gate_proj.weight_scale_inv", dev)
            self.gus[e, FF // 128:] = raw(f"{p}{e}.up_proj.weight_scale_inv", dev)
            self.dn[e] = raw(f"{p}{e}.down_proj.weight", dev); self.dns[e] = raw(f"{p}{e}.down_proj.weight_scale_inv", dev)

    @staticmethod
    def _dq(w, s):
        return (w.float().view(s.shape[0], 128, s.shape[1], 128) * s[:, None, :, None]).view(w.shape).bfloat16()

    def w(self, e):
        return self._dq(self.gu[e], self.gus[e]), self._dq(self.dn[e], self.dns[e])


def build_layer(cfg, L, dev):
    with torch.device("meta"):
        lay = M.Glm5NextTextDecoderLayer(cfg, L)
    p = f"{PFX}layers.{L}."
    sd, conv = {}, {}
    for k in IDX:
        if not k.startswith(p) or k.endswith("weight_scale_inv") or ".mlp.experts." in k:
            continue
        r = k[len(p):]
        m = re.match(r"self_attn\.([qkv])_conv1d\.weight", r)
        if m:
            conv[m.group(1)] = get(k, dev); continue
        for a, b in RENAME:
            r = re.sub(a, b, r)
        sd[r] = get(k, dev, torch.float32 if r.endswith(("A_log", "dt_bias", "e_score_correction_bias")) or "_hc." in r
                    else torch.bfloat16)
    if conv:
        sd["self_attn.conv1d.weight"] = torch.cat([conv["q"], conv["k"], conv["v"]], 0)
    if cfg.mlp_layer_types[L] == "sparse":                 # experts are handled by Experts (FP8 on GPU)
        lay.mlp.experts = torch.nn.Identity()
    missing, unexpected = lay.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and not missing, (L, missing, unexpected)
    for n, b in lay.named_buffers():
        assert b.device.type != "meta", (L, n)
    return lay.eval()


def cat_of(th, en):
    """exclusive boundary flags -> SAL_CATS index (0 = no boundary)."""
    c = np.zeros(th.shape, np.int64)
    for bi, (lo, hi) in enumerate(BUCKETS):
        c[(th >= lo) & (th <= hi)] = 1 + bi
        c[(en >= lo) & (en <= hi)] = 5 + bi
    return c


def flags(root):
    th = np.load(f"{root}/bnd_think_d.npy").astype(np.int8); en = np.load(f"{root}/bnd_end_d.npy").astype(np.int8)
    both = (th > 0) & (en > 0)                              # bnd19.corpus_flags: nearer wins, ties -> end
    th[both & (en <= th)] = 0; en[both & (th > 0) & (th < en)] = 0
    return th, en


def windows(corpus, smoke):
    """[(group, root, window, split)] for the run; fit windows interleaved across groups."""
    if corpus == "txt":
        spec = [("traces", f"{CORP}/c2048_traces", 761), ("c2048", f"{CORP}/c2048", 1200)]
    else:
        spec = [("mm", MM, 198)]
    out = []
    for g, root, nfit in spec:
        sp = json.load(open(f"{root}/split.json"))
        nf = min(nfit, sp["fit"][1]) if not smoke else smoke
        out += [(g, root, w, "fit") for w in range(sp["fit"][0], sp["fit"][0] + nf)]
        a, b = sp["val"]
        out += [(g, root, w, "val") for w in range(a, b if not smoke else a + 1)]
    return out


class Data:
    """This rank's windows: tokens, segment index lists, flags, rows that count as fit rows."""

    def __init__(self, wins, rank, world):
        self.wins = [w for i, w in enumerate(wins) if i % world == rank]
        self.cache = {}
        toks, segs, cats, fitm, valm, docid, pos, dom, imgs, wn = [], [], [], [], [], [], [], [], [], []
        o = 0; ndoc = 0
        for g, root, w, split in self.wins:
            if root not in self.cache:
                c = dict(tok=np.load(f"{root}/tokens.npy", mmap_mode="r"), seg=np.load(f"{root}/segments.npy", mmap_mode="r"),
                         fl=flags(root), win=[json.loads(l) for l in open(f"{root}/windows.jsonl")])
                c["rk"] = np.load(f"{root}/rowkind.npy", mmap_mode="r") if os.path.exists(f"{root}/rowkind.npy") else None
                c["img"] = {}
                if os.path.exists(f"{root}/images.json"):
                    for r in json.load(open(f"{root}/images.json")):
                        c["img"].setdefault(r["window"], []).append((r["pos"], f"{IMG}/{r['image_file']}"))
                self.cache[root] = c
            c = self.cache[root]
            t = np.asarray(c["tok"][w], np.int64); s = np.asarray(c["seg"][w])
            th, en = c["fl"][0][w], c["fl"][1][w]
            valid = s >= 0
            if c["rk"] is not None:
                valid &= np.asarray(c["rk"][w]) > 0
            catd = {d.get("segment_id", d.get("doc_index")): d.get("category", g) for d in c["win"][w].get("segments", [])}
            for sid in np.unique(s[s >= 0]):
                ix = np.flatnonzero(s == sid)
                segs.append(dict(ix=ix + o, n=len(ix), split=split, img=[(o + p, j) for p, j in c["img"].get(w, []) if s[p] == sid]))
                docid.append(np.full(len(ix), ndoc)); pos.append(np.arange(len(ix))); ndoc += 1
                dom += [f"{split}:{catd.get(int(sid), g)}"] * len(ix)
            toks.append(t); cats.append(cat_of(th, en)); fitm.append(valid & (split == "fit")); valm.append(valid & (split == "val"))
            o += len(t)
        self.tok = torch.from_numpy(np.concatenate(toks)); self.cat = torch.from_numpy(np.concatenate(cats))
        self.fit = torch.from_numpy(np.concatenate(fitm)); self.val = torch.from_numpy(np.concatenate(valm))
        self.segs = segs; self.N = o
        # per-row doc id / position for the val eval capture (in token order of self.tok)
        self.docid = np.zeros(o, np.int64); self.pos = np.zeros(o, np.int64); self.dom = [""] * o
        k = 0
        for sg, di, pp in zip(segs, docid, pos):
            self.docid[sg["ix"]] = di; self.pos[sg["ix"]] = pp
            for i in sg["ix"]:
                self.dom[i] = dom[k]; k += 1
        self.th = np.concatenate([self.cache[r]["fl"][0][w] for _, r, w, _ in self.wins]) if self.wins else np.zeros(0)
        self.en = np.concatenate([self.cache[r]["fl"][1][w] for _, r, w, _ in self.wins]) if self.wins else np.zeros(0)


class Stats:
    def __init__(self, dev):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.A0, self.A2 = z(NE, D, D), z(NE, D, D)
        self.D0, self.D2, self.Dc = z(NE, FF, FF), z(NE, FF, FF), z(NE, FF, FF)
        self.Cc, self.Ca = z(D, D), z(D, D)
        self.g = z(NE, 6, FF, dt=torch.float64)
        self.sal = z(NE, 9, 6, dt=torch.float64)


def gram_(A, X):
    A.addmm_(X.T, X)


def wgram_(A, X, w):
    """A += X^T diag(w) X with X bf16-exact fp32 and w fp32: (w X) split into bf16 hi + lo, both TF32-exact."""
    wx = X * w[:, None]
    hi = wx.bfloat16().float(); lo = (wx - hi).bfloat16().float()
    A.addmm_(X.T, hi); A.addmm_(X.T, lo)


def teacher(x, Wgu, Wd):
    """HF Experts math (bf16 linears, Flash clamps) -> h (down input, bf16), y (bf16), cg^2, cu^2 (fp32)."""
    gu = F.linear(x, Wgu)
    g, u = gu.chunk(2, -1)
    gc = g.clamp(max=10.); uc = u.clamp(-10., 10.)
    h = F.silu(gc) * uc
    y = F.linear(h, Wd)
    gf, uf = g.float(), u.float()
    sg = torch.sigmoid(gf.clamp(max=10.))
    cg = uf.clamp(-10., 10.) * sg * (1 + gf.clamp(max=10.) * (1 - sg)) * (gf <= 10.)
    cu = F.silu(gf.clamp(max=10.)) * (uf.abs() <= 10.)
    return h, y, cg.square(), cu.square()


def moe(lay, Ex, x, st, fitm, cat, ctx_dc, dc_scale, trace):
    """x [n, D] bf16 (post_attention_layernorm output). Returns MoE output (routed + shared) and accumulates stats
    on fit rows (fitm bool [n]); ctx_dc = row indices (into x) pushed through every expert for Dc / ctx gdiag."""
    logits, tw, ti = lay.mlp.gate(x)
    out = torch.zeros_like(x)
    if trace is not None:
        trace.append((ti.short().cpu(), tw.half().cpu(), x.float().square().sum(1).cpu()))
    if st is not None:
        xf = x.float()
        xs = xf[fitm]
        gram_(st.Ca, xs)
    flat = ti.reshape(-1)
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=NE).tolist()
    rows_all = order // TOPK
    starts = np.r_[0, np.cumsum(cnt)]
    if st is not None and len(ctx_dc):
        xc = x[ctx_dc]
    for e in range(NE):
        if cnt[e] == 0 and (st is None or not len(ctx_dc)):
            continue
        Wgu, Wd = Ex.w(e)
        if cnt[e]:
            sl = order[starts[e]:starts[e + 1]]
            rows = rows_all[starts[e]:starts[e + 1]]
            p = tw.reshape(-1)[sl]
            xe = x[rows]
            h, y, cg2, cu2 = teacher(xe, Wgu, Wd)
            out.index_add_(0, rows, (y * p[:, None].to(y.dtype)).to(out.dtype))
            if st is not None:
                m = fitm[rows]
                if m.any():
                    xr, hr, pr = xe[m].float(), h[m].float(), p[m].float()
                    p2 = pr.square()
                    gram_(st.A0[e], xr); wgram_(st.A2[e], xr, p2)
                    gram_(st.D0[e], hr); wgram_(st.D2[e], hr, p2)
                    g = st.g[e]
                    g[0] += (p2 @ cg2[m]).double(); g[1] += cg2[m].sum(0).double()
                    g[2] += (p2 @ cu2[m]).double(); g[3] += cu2[m].sum(0).double()
                    yn = y[m].float().norm(dim=1)
                    v = torch.stack([torch.ones_like(pr), pr, p2, p2.square(), pr * yn, yn], 1).double()
                    st.sal[e, 0] += v.sum(0)
                    cb = cat[rows][m]; bm = cb > 0             # boundary categories (row 0 = all, added above)
                    if bm.any():
                        st.sal[e].index_add_(0, cb[bm], v[bm])
        if st is not None and len(ctx_dc):
            h, _, cg2, cu2 = teacher(xc, Wgu, Wd)
            gram_(st.Dc[e], h.float() * dc_scale ** 0.5)
            st.g[e, 4] += cg2.sum(0).double() * dc_scale; st.g[e, 5] += cu2.sum(0).double() * dc_scale
    return out + lay.mlp.shared_experts(x)


def vision_embeds(cfg_full, data, dev):
    jobs = [(si, p, j) for si, s in enumerate(data.segs) for p, j in s["img"]]
    if not jobs:
        return {}
    from PIL import Image
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
    proc = Glm5NextImageProcessor(**{k: v for k, v in json.load(open(f"{CK}/processor_config.json"))["image_processor"].items()
                                     if k != "image_processor_type"})
    vis = M.Glm5NextVisionModel(cfg_full.vision_config)   # built on CPU: non-persistent buffers (rotary) need real init
    sd = {k[len("model.visual."):]: get(k, dev) for k in IDX if k.startswith("model.visual.") and not k.endswith("weight_scale_inv")}
    vis.load_state_dict(sd, strict=True, assign=True); vis.to(dev).eval()
    bad = [n for n, t in list(vis.named_parameters()) + list(vis.named_buffers()) if t.device.type != "cuda"]
    assert not bad, f"vision tensors off GPU: {bad[:5]}"
    out = {}
    for si, p, j in jobs:
        bf = proc(images=[Image.open(j).convert("RGB")], return_tensors="pt")
        o = vis(bf["pixel_values"].to(dev, torch.bfloat16), grid_thw=bf["image_grid_thw"].to(dev)).pooler_output
        out[(si, p)] = (o[0] if isinstance(o, (tuple, list)) else o).bfloat16()
    del vis
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["txt", "mm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-44")
    ap.add_argument("--smoke", type=int, default=0, help="fit windows per group (smoke run)")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--dc-cap", type=int, default=65536, help="ctx rows per rank pushed through every expert")
    ap.add_argument("--trace", default="", help="PRIVATE per-token routing dir")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    R, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", R % torch.cuda.device_count()); torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(2)
    cfg_full = AutoConfig.from_pretrained(CK); cfg = cfg_full.text_config
    cfg.num_local_experts = cfg.n_routed_experts
    assert cfg.n_routed_experts == NE and cfg.hidden_size == D and cfg.moe_intermediate_size == FF
    data = Data(windows(a.corpus, a.smoke), R, W)
    N = data.N
    fit_idx = torch.nonzero(data.fit)[:, 0]
    T_r = len(fit_idx)
    gen = torch.Generator().manual_seed(CTX_SEED + R)
    ctx = fit_idx[torch.randperm(T_r, generator=gen)[: T_r // 4]]
    n_ctx_r = len(ctx); dc = torch.sort(ctx[: min(n_ctx_r, a.dc_cap)])[0]
    dc_scale = n_ctx_r / max(len(dc), 1)
    ctxm = torch.zeros(N, dtype=torch.bool); ctxm[ctx] = True
    dcm = torch.zeros(N, dtype=torch.bool); dcm[dc] = True
    tot = torch.tensor([T_r, n_ctx_r, len(dc), int(data.val.sum()), N], dtype=torch.float64, device=dev)
    allt = [torch.zeros_like(tot) for _ in range(W)]; dist.all_gather(allt, tot)
    allt = [t.tolist() for t in allt]
    T_fit = int(sum(t[0] for t in allt)); n_ctx = int(sum(t[1] for t in allt))
    if R == 0:
        os.makedirs(f"{a.out}/stats", exist_ok=True); os.makedirs(f"{a.out}/eval/val", exist_ok=True)
        print(f"[cap37] {a.corpus}: {len(data.wins)} windows/rank0, T_fit {T_fit}, n_ctx {n_ctx}, per-rank {allt}", flush=True)
    os.makedirs(f"{a.out}/tmp", exist_ok=True)
    if a.trace:
        os.makedirs(a.trace, exist_ok=True)
        json.dump(dict(wins=[(g, w, s) for g, _, w, s in data.wins], N=N), open(f"{a.trace}/windows.r{R}of{W}.json", "w"))
    L0, L1 = map(int, a.layers.split("-"))
    hp = f"{a.out}/tmp/hid.r{R}.bf16"
    H = torch.empty(N, cfg.hc_mult, D, dtype=torch.bfloat16).pin_memory()
    if L0 == 0:
        emb = get(PFX + "embed_tokens.weight", dev)
        ims = vision_embeds(cfg_full, data, dev)
        for si, s in enumerate(data.segs):
            ix = torch.from_numpy(s["ix"])
            e = emb[data.tok[ix].to(dev)]
            tl = data.tok[ix]
            for p, _ in s["img"]:
                f = ims[(si, p)]
                lp = int(np.searchsorted(s["ix"], p))     # p = this rank's global row of the image's first token
                m = torch.nonzero(tl[lp:] == cfg_full.image_token_id)[: f.shape[0], 0] + lp
                assert len(m) == f.shape[0], (si, p, len(m), f.shape)
                e[m.to(dev)] = f.to(e.dtype)
            H[ix] = e[:, None].expand(-1, cfg.hc_mult, -1).cpu()
        del emb, ims
    else:
        H.copy_(torch.from_numpy(np.fromfile(hp, np.int16)).view(torch.bfloat16).view(N, cfg.hc_mult, D))
    dist.barrier()
    val_rows = torch.nonzero(data.val)[:, 0]
    blk = NE // W
    for L in range(L0, L1 + 1):
        t0 = time.time()
        lay = build_layer(cfg, L, dev)
        sparse = cfg.mlp_layer_types[L] == "sparse"
        torch.cuda.synchronize(); tl = time.time() - t0; t0 = time.time()
        Hn = torch.empty_like(H)                      # attention-half output for the whole rank
        for s in data.segs:
            ix = torch.from_numpy(s["ix"])
            h = H[ix].to(dev, non_blocking=True)[None]
            post, comb, hs = lay.attn_hc(h)
            hs = lay.input_layernorm(hs)
            n = s["n"]
            am = torch.ones(1, n, dtype=torch.bool, device=dev)
            if lay.block_type == "linear_attention":
                hs = lay.self_attn(hidden_states=hs, attention_mask=am)
            else:
                hs, _, _ = lay.self_attn(hidden_states=hs, attention_mask=am, position_ids=torch.arange(n, device=dev)[None],
                                         position_embeddings=None)
            h2 = post.to(h.dtype).unsqueeze(-1) * hs.unsqueeze(-2) + torch.matmul(comb.to(h.dtype).transpose(-1, -2), h)
            Hn[ix] = h2[0].cpu()
        torch.cuda.synchronize(); ta = time.time() - t0; t0 = time.time()
        Ex = Experts(L, dev) if sparse else None      # after attention: the KDA chunk kernel needs the headroom
        st = Stats(dev) if sparse else None
        trace = [] if (a.trace and sparse) else None
        ev = dict(x=[], ids=[], p=[]) if sparse else None
        for c0 in range(0, N, a.chunk):
            c1 = min(N, c0 + a.chunk)
            h = Hn[c0:c1].to(dev, non_blocking=True)[None]
            post, comb, hs = lay.ffn_hc(h)
            x = lay.post_attention_layernorm(hs)[0]
            if sparse:
                fm = data.fit[c0:c1].to(dev)
                dci = torch.nonzero(dcm[c0:c1])[:, 0].to(dev)
                if st is not None:
                    cm = ctxm[c0:c1].to(dev)
                    gram_(st.Cc, x[cm].float())
                y = moe(lay, Ex, x, st, fm, data.cat[c0:c1].to(dev), dci, dc_scale, trace)
                vm = data.val[c0:c1]
                if vm.any():
                    vmd = vm.to(dev)
                    _, tw, ti = lay.mlp.gate(x[vmd])
                    ev["x"].append(x[vmd].cpu()); ev["ids"].append(ti.cpu()); ev["p"].append(tw.float().cpu())
            else:
                y = lay.mlp(x)
            h2 = post.to(h.dtype).unsqueeze(-1) * y[None].unsqueeze(-2) + torch.matmul(comb.to(h.dtype).transpose(-1, -2), h)
            H[c0:c1] = h2[0].cpu()
        del Hn
        torch.cuda.synchronize(); tm = time.time() - t0; t0 = time.time()
        if sparse:
            del Ex
            torch.cuda.empty_cache()
            write_layer(a, L, st, R, W, blk, dict(T_fit=T_fit, n_ctx=n_ctx, allt=allt, dc_cap=a.dc_cap, corpus=a.corpus,
                                                  wins=len(windows(a.corpus, a.smoke))))
            write_eval(a, L, ev, data, val_rows, R, W)
            if trace is not None:
                ids = torch.cat([t[0] for t in trace]).numpy().astype(np.uint16)
                np.savez(f"{a.trace}/L{L}.r{R}of{W}.npz", ids=ids, w=torch.cat([t[1] for t in trace]).numpy(),
                         xn=torch.cat([t[2] for t in trace]).numpy())
        tw_ = time.time() - t0
        del lay, st
        torch.cuda.empty_cache()
        if L % 4 == 3 or L == L1:
            H.view(torch.int16).numpy().tofile(hp + ".tmp"); os.replace(hp + ".tmp", hp)
            json.dump(dict(next_layer=L + 1), open(f"{a.out}/tmp/ckpt.r{R}.json", "w"))
        if R == 0:
            print(f"[cap37] L{L} {cfg.layer_types[L][:6]} {cfg.mlp_layer_types[L]} load {tl:.0f}s attn {ta:.0f}s "
                  f"mlp {tm:.0f}s write {tw_:.0f}s  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
        dist.barrier()
    if L1 == cfg.num_hidden_layers - 1:
        final(a, cfg, H, data, dev, R, W)
    dist.barrier()
    dist.destroy_process_group()


def write_layer(a, L, st, R, W, blk, info):
    vd = f"{a.out}/stats/L{L}.v1"
    if R == 0:
        os.makedirs(vd, exist_ok=True)
    dist.barrier()
    for j in range(W):                                    # reduce each expert block to its owner rank
        sl = slice(j * blk, (j + 1) * blk)
        for t in (st.A0, st.A2, st.D0, st.D2, st.Dc):
            dist.reduce(t[sl], dst=j)
    for t in (st.Cc, st.Ca, st.g, st.sal):
        dist.reduce(t, dst=0)
    files = {}
    iu = {n: torch.triu_indices(n, n, device=st.A0.device) for n in (D, FF)}
    for k, t, n in (("A2", st.A2, D), ("A0", st.A0, D), ("D2", st.D2, FF), ("D0", st.D0, FF), ("Dc", st.Dc, FF)):
        sb = stride_bytes(n)
        files[k] = dict(file=f"{k}.f32", rows=NE, stride_bytes=sb, packed=npk(n), n=n)
        fn = f"{vd}/{k}.f32"
        if R == 0:
            with open(fn, "wb") as f:
                f.truncate(NE * sb)
    dist.barrier()
    for k, t, n in (("A2", st.A2, D), ("A0", st.A0, D), ("D2", st.D2, FF), ("D0", st.D0, FF), ("Dc", st.Dc, FF)):
        sb = stride_bytes(n)
        fd = os.open(f"{vd}/{k}.f32", os.O_WRONLY)
        for e in range(R * blk, (R + 1) * blk):
            v = t[e][iu[n][0], iu[n][1]].cpu().numpy()
            os.pwrite(fd, v.tobytes(), e * sb)
        os.close(fd)
    if R == 0:
        pk = lambda A: A[iu[D][0], iu[D][1]].cpu().numpy()
        np.save(f"{vd}/C_ctx.npy", pk(st.Cc)); np.save(f"{vd}/C_all.npy", pk(st.Ca))
        np.save(f"{vd}/gdiag.npy", st.g.cpu().numpy())
        sal = st.sal.cpu().numpy(); np.save(f"{vd}/sal.npy", sal)
        sc = sal[:, 0, :4].copy(); np.save(f"{vd}/scalars.npy", sc)
        ess = (sc[:, 2] ** 2 / np.maximum(sc[:, 3], 1e-300)).tolist()
        meta = dict(schema="nestquant-19-stats-v2", layer=L, files=dict(raw=files), T_fit=info["T_fit"], n_ctx=info["n_ctx"],
                    n_routed=sc[:, 0].astype(int).tolist(), ess=ess,
                    shards=[dict(shard=0, T=info["T_fit"], n_ctx=info["n_ctx"], n_dc=int(sum(t[2] for t in info["allt"])),
                                 per_rank=info["allt"], seed=CTX_SEED, corpus=info["corpus"])],
                    protocol=dict(model="zai-org/GLM-5.3-Flash (FP8)", D=D, F=FF, NEXP=NE, sal_cats=SAL_CATS,
                                  sal_cols=["n", "sum_p", "sum_p2", "sum_p4", "sum_p_ynorm", "sum_ynorm"],
                                  grams="fp32 TF32 GEMM on bf16-exact rows; p2 side bf16 hi/lo", gdiag="bf16 gx/ux, Flash clamps",
                                  dc=f"first min(n_ctx_r, {info['dc_cap']}) ctx rows per rank, scaled n_ctx_r/n_dc_r",
                                  ctx="randperm(T_r, seed CTX_SEED + rank)[:T_r // 4] per rank"))
        json.dump(meta, open(f"{vd}/meta.json", "w"))
        lk = f"{a.out}/stats/L{L}"
        if os.path.lexists(lk):
            os.remove(lk)
        os.symlink(f"L{L}.v1", lk)
    dist.barrier()


def write_eval(a, L, ev, data, val_rows, R, W):
    if ev["x"]:
        vr = val_rows.numpy()
        part = dict(x=torch.cat(ev["x"]), ids=torch.cat(ev["ids"]).long(), p=torch.cat(ev["p"]),
                    document_ids=torch.from_numpy(data.docid[vr]), token_positions=torch.from_numpy(data.pos[vr]),
                    domains=[data.dom[i] for i in vr], bnd_think=torch.from_numpy(data.th[vr].astype(np.int8)),
                    bnd_end=torch.from_numpy(data.en[vr].astype(np.int8)))
    else:
        part = None
    torch.save(part, f"{a.out}/tmp/ev.L{L}.r{R}.pt")
    dist.barrier()
    if R == 0:
        parts = [torch.load(f"{a.out}/tmp/ev.L{L}.r{r}.pt", weights_only=False) for r in range(W)]
        parts = [p for p in parts if p is not None]
        out, off = {}, 0
        for k in ("x", "ids", "p", "token_positions", "bnd_think", "bnd_end"):
            out[k] = torch.cat([p[k] for p in parts])
        dids = []
        for p in parts:                                   # globally renumbered document ids
            u, inv = torch.unique(p["document_ids"], return_inverse=True); dids.append(inv + off); off += len(u)
        out["document_ids"] = torch.cat(dids); out["domains"] = sum((p["domains"] for p in parts), [])
        out["layer"] = L; out["protocol"] = dict(role="evaluation only", model="GLM-5.3-Flash", corpus=a.corpus)
        torch.save(out, f"{a.out}/eval/val/layer_{L}.pt")
        for r in range(W):
            os.remove(f"{a.out}/tmp/ev.L{L}.r{r}.pt")
    dist.barrier()


def final(a, cfg, H, data, dev, R, W):
    norm = M.Glm5NextTextRMSNorm(D, cfg.rms_norm_eps).to(dev)
    norm.weight.copy_(get(PFX + "norm.weight", dev, torch.float32))
    norm = norm.bfloat16()
    head = get("lm_head.weight", dev)
    hh = M.Glm5NextTextHyperHead()
    os.makedirs(f"{a.out}/final", exist_ok=True)
    res, top = [], {}
    for si, s in enumerate(data.segs):
        if s["n"] < 2:
            continue
        ix = torch.from_numpy(s["ix"])
        z = F.linear(norm(hh(H[ix].to(dev)[None]))[0], head).float()
        lp = F.log_softmax(z, -1); tgt = data.tok[ix][1:].to(dev)
        ce = -lp[:-1].gather(1, tgt[:, None])[:, 0]; acc = (lp[:-1].argmax(-1) == tgt).float()
        res.append(dict(rank=R, seg=si, split=s["split"], n=s["n"], ce=float(ce.mean()), acc=float(acc.mean()),
                        img=len(s["img"])))
        if s["split"] == "val":
            v, i = lp.topk(64, -1); top[si] = dict(rows=ix, v=v.half().cpu(), i=i.int().cpu())
    torch.save(top, f"{a.out}/final/top64.r{R}.pt")
    json.dump(res, open(f"{a.out}/final/ce.r{R}.json", "w"))
    dist.barrier()
    if R == 0:
        rr = sum((json.load(open(f"{a.out}/final/ce.r{r}.json")) for r in range(W)), [])
        for sp in ("fit", "val"):
            q = [r for r in rr if r["split"] == sp]
            nt = sum(r["n"] - 1 for r in q)
            if nt:
                print(f"[cap37] final {sp}: ppl {np.exp(sum(r['ce'] * (r['n'] - 1) for r in q) / nt):.3f} "
                      f"acc {sum(r['acc'] * (r['n'] - 1) for r in q) / nt:.4f} ({nt} tokens)", flush=True)


if __name__ == "__main__":
    main()
