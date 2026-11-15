"""Inkling-2-Small layer-sequential calibration capture in T19's stats schema (nestquant-19-stats-v2, D=4608, F=2304,
384 experts), adapted from threads/37-flash/capture37g.py.

    torchrun --nproc-per-node 8 cap_ink.py --out /tmp/nq_inkling/cap-txt [--layers 0-47] [--smoke N]

Token-parallel: rank r owns whole windows and runs every layer over them, segment-local (positions restart per
segment).  Per layer:
  attention half per segment (HF Inkling2 layer modules, eager attention with explicit causal/sliding masks)
    -> residual (fp32), MoE input x (bf16), block-state update
  MoE half in G expert-group passes (stats for 384/G experts resident at a time), expert weights dequantised from
    the native NVFP4 (L0-39) / MXFP8 (L40-47) source one expert at a time; outputs accumulated in fp32
  post-MLP short conv per segment, new hidden = (residual + conv(y)).bf16
Teacher = HF Inkling2Experts math: bf16 linears, fp32 softcapped SwiGLU (gate cap 4, up cap 25), fp32 accumulation.
Outputs (ROOT): stats/L{L}.v1 (A0 A2 D0 D2 Dc C_ctx C_all gdiag sal scalars meta), eval/val/layer_{L}.pt,
final/ (val top-64 teacher logprobs, CE), PRIVATE trace/ (never uploaded)."""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

torch.set_grad_enabled(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkdq as Q
from transformers import AutoConfig
from transformers.models.inkling2 import modeling_inkling2 as M

CK = Q.CK
CORP = os.environ.get("INK_CORP", "/tmp/nq_inkling/corpus")
HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.environ.get("INK_MM", "")          # multimodal corpus root (corpus_mm): image windows, vision tower splice
D, FF, NE, TOPK = Q.D, Q.F, Q.NE, 8
CTX_SEED = 20261113
SAL_CATS = ["all"] + [f"{k}/{b}" for k in ("think", "end") for b in ("d1", "d2_4", "d5_16", "d17_32")]
BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 32))
FP32_KEYS = ("e_score_correction_bias", "mlp.gate.weight", "post_attention_convolution", "post_mlp_convolution",
             "k_sconv.conv1d", "v_sconv.conv1d", "attention_input", "mlp_input", "final_input")
GATE_CAP, UP_CAP = 4.0, 25.0


def npk(n):
    return n * (n + 1) // 2


def stride_bytes(n):
    return (npk(n) * 4 + 4095) // 4096 * 4096


# ---------------------------------------------------------------- checkpoint -> HF names (conversion_mapping inkling2_text)
REN = [(r"attn\.wq_du", "self_attn.q_proj"), (r"attn\.wk_dv", "self_attn.k_proj"), (r"attn\.wv_dv", "self_attn.v_proj"),
       (r"attn\.wr_du", "self_attn.r_proj"), (r"attn\.wo_ud", "self_attn.o_proj"), (r"attn\.gate_attn", "self_attn.gate_proj"),
       (r"attn\.attn_sink_proj", "self_attn.sink_proj"),
       (r"^attn\.(q_l2_clip|k_l2_clip|rel_l2_clip|k_sconv|v_sconv|rel_logits_proj)\.", r"self_attn.\1."),
       (r"attn_sconv\.", "post_attention_convolution."), (r"mlp_sconv\.", "post_mlp_convolution."),
       (r"(convolution|sconv)\.weight(?!_)", r"\1.conv1d.weight"), (r"(convolution|sconv)\.static_w", r"\1.static_weight"),
       (r"(convolution|sconv)\.weight_proj", r"\1.weight_proj.weight"), (r"mlp_norm", "post_attention_layernorm"),
       (r"attn_norm", "input_layernorm"), (r"(q|k|v)_input_norm", r"\1_input_layernorm"),
       (r"mlp\.gate\.bias", "mlp.gate.e_score_correction_bias")]


def get(k, dev, fp32=False):
    w = Q.raw(k)
    return w.to(dev, torch.float32 if (fp32 or w.dtype == torch.float32) else torch.bfloat16)


def build_layer(cfg, L, dev):
    cls = M.Inkling2SlidingDecoderLayer if cfg.layer_types[L] == "hybrid_sliding" else M.Inkling2FullDecoderLayer
    with torch.device("meta"):
        lay = cls(cfg, L)
    p = f"model.llm.layers.{L}."
    sd = {}
    for k in Q.idx():
        if not k.startswith(p) or ".mlp.experts." in k:
            continue
        r = k[len(p):]
        for a, b in REN:
            r = re.sub(a, b, r)
        sd[r] = get(k, dev, fp32=any(s in r for s in FP32_KEYS))
    if L > 0:
        sd["attention_input.query"] = get(f"model.llm.attn_res.attn_queries.{L - 1}", dev, True)
    sd["mlp_input.query"] = get(f"model.llm.attn_res.mlp_queries.{L}", dev, True)
    lay.mlp.experts = torch.nn.Identity()
    missing, unexpected = lay.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not m.startswith("mlp.experts")]
    assert not unexpected and not missing, (L, missing, unexpected)
    for n, t in list(lay.named_parameters()) + list(lay.named_buffers()):
        assert t.device.type != "meta", (L, n)
    return lay.eval()


def masks(n, dev, window):
    q = torch.arange(n, device=dev)
    d = q[:, None] - q[None, :]
    ok = d >= 0
    full = torch.zeros(n, n, device=dev, dtype=torch.bfloat16).masked_fill(~ok, float("-inf"))[None, None]
    slide = torch.zeros(n, n, device=dev, dtype=torch.bfloat16).masked_fill(~(ok & (d < window)), float("-inf"))[None, None]
    return full, slide


def attn_half(lay, cfg, h, bs, full_m, slide_m):
    """h [1,n,D] bf16 layer input, bs [1,n,k,D] bf16 block states (before this layer's append).
    Returns residual fp32 [1,n,D], x bf16 [1,n,D] (MoE input)."""
    dtype = h.dtype
    n = h.shape[1]
    pos = torch.arange(n, device=h.device)[None]
    block_residual = h
    if isinstance(lay, M.Inkling2SlidingDecoderLayer):
        hs = lay.attention_input(bs, h)[0] if lay.attention_input is not None else h
        a, _ = lay.self_attn(hidden_states=lay.input_layernorm(hs).to(dtype), attention_mask=slide_m, conv_mask=None,
                             cache=None, position_ids=pos)
    else:
        if lay.attention_input is None:
            qh = kh = vh = h
        else:
            qh, kh, vh = lay.attention_input(bs, h)
        a, _ = lay.self_attn(hidden_states=lay.q_input_layernorm(qh).to(dtype),
                             key_hidden_states=lay.k_input_layernorm(kh).to(dtype),
                             value_hidden_states=lay.v_input_layernorm(vh).to(dtype),
                             attention_mask=full_m, conv_mask=None, cache=None, position_ids=pos)
    a = lay.post_attention_convolution(a, cache=None, conv_mask=None)
    residual = a.float()
    if lay.is_block_boundary:
        bs = torch.cat([bs, block_residual.unsqueeze(-2)], dim=-2)
    else:
        residual = block_residual.float() + residual
    x = lay.post_attention_layernorm(lay.mlp_input(bs, residual)[0]).to(dtype)
    return residual, x


# ---------------------------------------------------------------- data
def cat_of(th, en):
    c = np.zeros(th.shape, np.int64)
    for bi, (lo, hi) in enumerate(BUCKETS):
        c[(th >= lo) & (th <= hi)] = 1 + bi
        c[(en >= lo) & (en <= hi)] = 5 + bi
    return c


def flags(root):
    th = np.load(f"{root}/bnd_think_d.npy").astype(np.int8); en = np.load(f"{root}/bnd_end_d.npy").astype(np.int8)
    both = (th > 0) & (en > 0)
    th[both & (en <= th)] = 0; en[both & (th > 0) & (th < en)] = 0
    return th, en


def windows(smoke, nfit):
    spec = [("traces", f"{CORP}/c2048_traces", nfit[0]), ("c2048", f"{CORP}/c2048", nfit[1])]
    if MM:
        spec = [("mm", MM, nfit[0])]
    out = []
    for g, root, nf in spec:
        sp = json.load(open(f"{root}/split.json"))
        nf = min(nf, sp["fit"][1] - sp["fit"][0]) if not smoke else smoke
        out += [(g, root, w, "fit") for w in range(sp["fit"][0], sp["fit"][0] + nf)]
        a, b = sp["val"]
        out += [(g, root, w, "val") for w in range(a, b if not smoke else a + 1)]
    return out


class Data:
    def __init__(self, wins, rank, world):
        self.wins = [w for i, w in enumerate(wins) if i % world == rank]
        cache = {}
        toks, cats, fitm, valm, segs, docid, pos, dom, ths, ens = [], [], [], [], [], [], [], [], [], []
        o = 0; ndoc = 0; self.woff = []; rks = []
        for g, root, w, split in self.wins:
            if root not in cache:
                cache[root] = dict(tok=np.load(f"{root}/tokens.npy", mmap_mode="r"), seg=np.load(f"{root}/segments.npy", mmap_mode="r"),
                                   fl=flags(root), win=[json.loads(l) for l in open(f"{root}/windows.jsonl")])
            c = cache[root]
            t = np.asarray(c["tok"][w], np.int64); s = np.asarray(c["seg"][w])
            th, en = c["fl"][0][w], c["fl"][1][w]
            valid = s >= 0
            self.woff.append((o, root, w))
            catd = {d.get("segment_id", d.get("doc_index")): d.get("category", g) for d in c["win"][w].get("segments", [])}
            for sid in np.unique(s[s >= 0]):
                ix = np.flatnonzero(s == sid)
                segs.append(dict(ix=ix + o, n=len(ix), split=split))
                docid.append(np.full(len(ix), ndoc)); pos.append(np.arange(len(ix))); ndoc += 1
                dom += [f"{split}:{catd.get(int(sid), g)}"] * len(ix)
            vval = valid
            if g == "mm" and "rk" not in c:
                c["rk"] = np.load(f"{root}/rowkind.npy", mmap_mode="r")
            if g == "mm":                                   # KLD/CE only on text rows (image rows predict placeholders)
                if "rk" not in c:
                    c["rk"] = np.load(f"{root}/rowkind.npy", mmap_mode="r")
                vval = valid & (np.asarray(c["rk"][w]) == 2)
            rks.append(np.asarray(c["rk"][w]) if g == "mm" else np.full(len(t), 2, np.int8))
            toks.append(t); cats.append(cat_of(th, en)); fitm.append(valid & (split == "fit")); valm.append(vval & (split == "val"))
            ths.append(th); ens.append(en)
            o += len(t)
        self.tok = torch.from_numpy(np.concatenate(toks)); self.cat = torch.from_numpy(np.concatenate(cats))
        self.fit = torch.from_numpy(np.concatenate(fitm)); self.val = torch.from_numpy(np.concatenate(valm))
        self.segs = segs; self.N = o
        self.docid = np.zeros(o, np.int64); self.pos = np.zeros(o, np.int64); self.dom = [""] * o
        k = 0
        for sg, di, pp in zip(segs, docid, pos):
            self.docid[sg["ix"]] = di; self.pos[sg["ix"]] = pp
            for i in sg["ix"]:
                self.dom[i] = dom[k]; k += 1
        self.th = np.concatenate(ths); self.en = np.concatenate(ens); self.rk = np.concatenate(rks)


# ---------------------------------------------------------------- MoE + stats
class Stats:
    def __init__(self, ne, dev):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.A0, self.A2 = z(ne, D, D), z(ne, D, D)
        self.D0, self.D2, self.Dc = z(ne, FF, FF), z(ne, FF, FF), z(ne, FF, FF)


def gram_(A, X):
    A.addmm_(X.T, X)


def wgram_(A, X, w):
    wx = X * w[:, None]
    hi = wx.bfloat16().float(); lo = (wx - hi).bfloat16().float()
    A.addmm_(X.T, hi); A.addmm_(X.T, lo)


def teacher(x, Wgu, Wd):
    """HF Inkling2Experts math -> h (down input, bf16), y (bf16), cg^2, cu^2 (fp32: |dh/dg|^2, |dh/du|^2)."""
    gu = F.linear(x, Wgu)
    g, u = gu.chunk(2, -1)
    gf, uf = g.float(), u.float()
    tg, tu = torch.tanh(gf / GATE_CAP), torch.tanh(uf / UP_CAP)
    sg = torch.sigmoid(gf)
    scg, scu = GATE_CAP * tg, UP_CAP * tu
    h = (scg * sg * scu).to(x.dtype)
    y = F.linear(h, Wd)
    cg = scu * ((1 - tg * tg) * sg + scg * sg * (1 - sg))
    cu = scg * sg * (1 - tu * tu)
    return h, y, cg.square(), cu.square()


def route(lay, x):
    _, tw, ti = lay.mlp.gate(x)
    return tw, ti


def moe_pass(lay, Wc, x, Yc, st, e0, e1, fitm, cat, xc, dc_scale, g_acc, sal_acc, trace, hot=None):
    """Experts [e0, e1) over chunk rows x [n,D] bf16; adds routed outputs into Yc (fp32 [n,D], on GPU)."""
    tw, ti = route(lay, x)
    if trace is not None:
        trace.append((ti.short().cpu(), tw.half().cpu(), x.float().square().sum(1).cpu()))
    flat = ti.reshape(-1)
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=NE).tolist()
    rows_all = order // TOPK
    starts = np.r_[0, np.cumsum(cnt)]
    for e in range(e0, e1):
        if cnt[e] == 0 and (st is None or xc is None):
            continue
        if len(Wc[e]) == 4:                       # floating hot set: rows whose block holds e hot use level 4
            sl = order[starts[e]:starts[e + 1]]
            rows = rows_all[starts[e]:starts[e + 1]]
            p = tw.reshape(-1)[sl]
            hm_ = hot[rows, e]
            for sel, (Wgu, Wd) in ((~hm_, Wc[e][:2]), (hm_, Wc[e][2:])):
                if sel.any():
                    r_ = rows[sel]
                    _, y, _, _ = teacher(x[r_], Wgu, Wd)
                    Yc.index_add_(0, r_, y.float() * p[sel][:, None].float())
            continue
        Wgu, Wd = Wc[e]
        j = e - e0
        if cnt[e]:
            sl = order[starts[e]:starts[e + 1]]
            rows = rows_all[starts[e]:starts[e + 1]]
            p = tw.reshape(-1)[sl]
            xe = x[rows]
            h, y, cg2, cu2 = teacher(xe, Wgu, Wd)
            Yc.index_add_(0, rows, y.float() * p[:, None].float())
            if st is not None:
                m = fitm[rows]
                if m.any():
                    xr, hr, pr = xe[m].float(), h[m].float(), p[m].float()
                    p2 = pr.square()
                    gram_(st.A0[j], xr); wgram_(st.A2[j], xr, p2)
                    gram_(st.D0[j], hr); wgram_(st.D2[j], hr, p2)
                    g = g_acc[e]
                    g[0] += (p2 @ cg2[m]).double(); g[1] += cg2[m].sum(0).double()
                    g[2] += (p2 @ cu2[m]).double(); g[3] += cu2[m].sum(0).double()
                    yn = y[m].float().norm(dim=1)
                    v = torch.stack([torch.ones_like(pr), pr, p2, p2.square(), pr * yn, yn], 1).double()
                    sal_acc[e, 0] += v.sum(0)
                    cb = cat[rows][m]; bm = cb > 0
                    if bm.any():
                        sal_acc[e].index_add_(0, cb[bm], v[bm])
        if st is not None and xc is not None:
            h, _, cg2, cu2 = teacher(xc, Wgu, Wd)
            gram_(st.Dc[j], h.float() * dc_scale ** 0.5)
            g_acc[e, 4] += cg2.sum(0).double() * dc_scale; g_acc[e, 5] += cu2.sum(0).double() * dc_scale


_PRED = {}


def float_sets(a, L, lay, X, data, dev):
    """Per-row hot mask [N,384] (GPU bool) from the floating-set predictor, driven by this model's own routing:
    blocks of 16 tokens per segment (= request), set chosen at the end of block b serves block b+1."""
    sys.path.insert(0, os.path.join(HERE, "pred"))
    import train as PT
    if "net" not in _PRED:
        ck = torch.load(a.pred, map_location=dev)
        net = PT.Net(ck["args"]["d"], ck["args"]["nl"]).to(dev).eval(); net.load_state_dict(ck["state"])
        _PRED.update(net=net, stats=PT.layer_stats())
    N = X.shape[0]
    tid = torch.empty(N, TOPK, dtype=torch.long, device=dev); tww = torch.empty(N, TOPK, device=dev); xn = torch.empty(N, device=dev)
    with torch.no_grad():
        for c0 in range(0, N, a.chunk):
            x = X[c0:c0 + a.chunk].to(dev)
            tw, ti = route(lay, x)
            tid[c0:c0 + len(x)] = ti; tww[c0:c0 + len(x)] = tw.float(); xn[c0:c0 + len(x)] = x.float().square().sum(1)
    blk_of = torch.full((N,), -1, dtype=torch.long, device=dev)
    chain, pos = [], []
    nb = 0
    for ci, sg in enumerate(data.segs):
        ix = torch.from_numpy(sg["ix"]).to(dev)
        b = torch.arange(len(ix), device=dev) // 16
        blk_of[ix] = nb + b
        k = int(b[-1]) + 1
        chain.append(torch.full((k,), ci, device=dev, dtype=torch.long)); pos.append(torch.arange(k, device=dev))
        nb += k
    ok = blk_of >= 0
    cnt = torch.zeros(nb, NE, device=dev); sal = torch.zeros(nb, NE, device=dev)
    br = blk_of[ok][:, None].expand(-1, TOPK)
    cnt.index_put_((br.reshape(-1), tid[ok].reshape(-1)), torch.ones(br.numel(), device=dev), accumulate=True)
    sal.index_put_((br.reshape(-1), tid[ok].reshape(-1)), (tww[ok] ** 2 * xn[ok][:, None]).reshape(-1), accumulate=True)
    chain, pos = torch.cat(chain), torch.cat(pos)
    if a.pred_mode == "batch":
        R = PT.sets_for_blocks(_PRED["net"], L, cnt, sal, chain, pos, a.pred_H, a.pred_hm, _PRED["stats"])
    elif a.pred_mode == "ema4":                  # heuristic: salience EMA (half-life 4 blocks), same replay + hysteresis
        mL, prior = _PRED["stats"][L]
        d = dict(cnt=cnt, sal=sal / mL, chain=chain, pos=pos, prior=prior.to(dev))
        X, _, _ = PT.features(d)
        S = torch.expm1(X[..., len(PT.HL) + 1].float()) / 8 + 1e-4 * torch.exp(d["prior"])[None]
        R = PT.replay(S, d["sal"], chain, pos, d["prior"], a.pred_H, a.pred_hm, return_sets=True)
    else:                                        # stream: the published serving/predictor/predictor.py, block by block
        if "sp" not in _PRED:
            sys.path.insert(0, "/tmp/nq_inkling/release/serving/predictor")
            import predictor as SP
            _PRED["sp"] = SP.HotSetPredictor("/tmp/nq_inkling/release/serving/predictor", a.pred_H, a.pred_hm, dev)
        P = _PRED["sp"]
        R = torch.zeros(nb, NE, dtype=torch.bool, device=dev)
        c48 = torch.zeros(48, NE, device=dev); s48 = torch.zeros(48, NE, device=dev)
        for b in range(nb):
            if int(pos[b]) == 0:
                P.reset()
            R[b] = P.hot[L]
            c48[L] = cnt[b]; s48[L] = sal[b]
            P.end_block(c48, s48)
    hot = torch.ones(N, NE, dtype=torch.bool, device=dev)     # pad rows: level 4 (as the fixed-level paths; never scored)
    hot[ok] = R[blk_of[ok]]
    if os.environ.get("INK_DEBUG_HOT"):
        print(f"[hotdbg] L{L} N {N} chunk {a.chunk} ok {int(ok.sum())} hot-frac(ok) {float(hot[ok].float().mean()):.4f} "
              f"R-frac {float(R.float().mean()):.4f} nb {nb}", flush=True)
    return hot


class LazyW:
    """Eval-only (one chunk): load each expert's weights when used instead of caching the whole layer on the GPU."""
    def __init__(self, ex):
        self.ex = ex

    def __getitem__(self, e):
        return self.ex.w(e)


class WSel:
    """Per-expert weights from predecoded NestQuant levels (eval mode)."""

    def __init__(self, a, L, dev):
        if a.wsel == "float":
            lv = [2] * NE
            self.lv = lv; self.dev = dev; self.float = True
            self.src = {v: torch.load(f"{a.wdir}/L{L}.lv{v}.pt", mmap=True) for v in (2, 4)}
            return
        self.float = False
        if a.wsel.startswith("hot:"):
            n = int(a.wsel[4:])
            nr = json.load(open(f"{a.hot_stats}/L{L}/meta.json"))["n_routed"]
            hot = set(sorted(range(NE), key=lambda e: -nr[e])[:n])
            lv = [4 if e in hot else 2 for e in range(NE)]
        else:
            lv = [int(a.wsel)] * NE
        self.lv = lv
        self.src = {v: torch.load(f"{a.wdir}/L{L}.lv{v}.pt", mmap=True) for v in set(lv)}
        self.dev = dev

    def w(self, e):
        if self.float:
            a, b = self.src[2], self.src[4]
            return (a["gu"][e].to(self.dev), a["dn"][e].to(self.dev), b["gu"][e].to(self.dev), b["dn"][e].to(self.dev))
        s = self.src[self.lv[e]]
        return s["gu"][e].to(self.dev), s["dn"][e].to(self.dev)


def splice_vision(data, H, dev, cfg):
    """Replace image placeholder rows (config.image_token_id) with Inkling2 vision tower outputs, as
    Inkling2Model.forward does (masked_scatter after the normed embedding)."""
    sys.path.insert(0, os.path.join(HERE, "corpus_mm"))
    import embed_demo as V
    vm, fcfg = V.load_vision()
    vm = vm.to(dev)
    wl = {}
    n = 0
    with torch.no_grad():
        for o, root, w in data.woff:
            if root not in wl:
                wl[root] = [json.loads(l) for l in open(f"{root}/windows.jsonl")]
            slots = wl[root][w].get("images", [])
            if not slots:
                continue
            pix = [torch.load(f"{root}/{sl['pix_file']}", weights_only=True) for sl in slots]
            pv = torch.cat([p["pixel_values"] for p in pix]).to(dev)
            grid = torch.cat([p["image_grid_thw"] for p in pix]).to(dev)
            feats = vm(pixel_values=pv, image_grid_thw=grid).pooler_output.split(grid.prod(-1).tolist())
            for sl, f in zip(slots, feats):
                a, b = o + sl["start"], o + sl["end"]
                assert bool((data.tok[a:b] == fcfg.image_token_id).all()) and f.shape[0] == b - a
                H[a:b] = f.to(torch.bfloat16).cpu(); n += b - a
    assert n == int((data.tok == fcfg.image_token_id).sum()), (n, int((data.tok == fcfg.image_token_id).sum()))
    print(f"[capink] vision splice: {n} image rows", flush=True)
    del vm; torch.cuda.empty_cache()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--nfit", default="761,1200", help="fit windows: traces,c2048")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--groups", type=int, default=2, help="expert-group passes per layer (stats memory / groups)")
    ap.add_argument("--dc-cap", type=int, default=65536)
    ap.add_argument("--trace", default="")
    ap.add_argument("--no-stats", action="store_true", help="forward only (validation)")
    ap.add_argument("--swap-nibbles", action="store_true", help="validation: deliberately wrong fp4 nibble order")
    ap.add_argument("--wdir", default="", help="predecoded NestQuant experts: WDIR/L{L}.lv{2,4}.pt (bf16 gu [E,2F,D], dn [E,D,F])")
    ap.add_argument("--wsel", default="4", help="2 | 4 | hot:N (top-N experts by routed count in --hot-stats -> level 4)")
    ap.add_argument("--wmax", type=int, default=48, help="only layers < wmax use predecoded weights")
    ap.add_argument("--pred", default="/tmp/nq_inkling/pred/net.pt", help="--wsel float: floating-set predictor")
    ap.add_argument("--pred-H", type=int, default=64)
    ap.add_argument("--pred-hm", type=float, default=0.5)
    ap.add_argument("--pred-mode", default="batch", choices=["batch", "stream", "ema4"],
                    help="batch = train.py sets_for_blocks; stream = published predictor.py; ema4 = salience-EMA heuristic")
    ap.add_argument("--hot-stats", default="/tmp/nq_inkling/cap-txt/stats")
    ap.add_argument("--ref", default="", help="reference capture root: KLD vs its final/top64.r{R}.pt")
    a = ap.parse_args()
    if a.swap_nibbles:
        _u = Q.fp4_unpack
        Q.fp4_unpack = lambda u8, lut: _u(((u8 & 0x0F) << 4) | (u8 >> 4), lut)
    dist.init_process_group("nccl")
    R, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", R % torch.cuda.device_count()); torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    cfg = AutoConfig.from_pretrained(CK).text_config
    cfg._attn_implementation = "eager"
    assert cfg.n_routed_experts == NE and cfg.hidden_size == D and cfg.moe_intermediate_size == FF
    nfit = [int(v) for v in a.nfit.split(",")]
    data = Data(windows(a.smoke, nfit), R, W)
    N = data.N
    fit_idx = torch.nonzero(data.fit)[:, 0]
    T_r = len(fit_idx)
    gen = torch.Generator().manual_seed(CTX_SEED + R)
    ctx = fit_idx[torch.randperm(T_r, generator=gen)[: T_r // 4]]
    n_ctx_r = len(ctx); dc = torch.sort(ctx[: min(n_ctx_r, a.dc_cap)])[0]
    dc_scale = n_ctx_r / max(len(dc), 1)
    ctxm = torch.zeros(N, dtype=torch.bool); ctxm[ctx] = True
    tot = torch.tensor([T_r, n_ctx_r, len(dc), int(data.val.sum()), N], dtype=torch.float64, device=dev)
    allt = [torch.zeros_like(tot) for _ in range(W)]; dist.all_gather(allt, tot)
    allt = [t.tolist() for t in allt]
    T_fit = int(sum(t[0] for t in allt)); n_ctx = int(sum(t[1] for t in allt))
    if R == 0:
        os.makedirs(f"{a.out}/stats", exist_ok=True); os.makedirs(f"{a.out}/eval/val", exist_ok=True)
        print(f"[capink] {len(data.wins)} windows/rank0, T_fit {T_fit}, n_ctx {n_ctx}, per-rank {allt}", flush=True)
    os.makedirs(f"{a.out}/tmp", exist_ok=True)
    if a.trace:
        os.makedirs(a.trace, exist_ok=True)
        json.dump(dict(wins=[(g, w, s) for g, _, w, s in data.wins], N=N), open(f"{a.trace}/windows.r{R}of{W}.json", "w"))
    L0, L1 = map(int, a.layers.split("-"))
    hp = f"{a.out}/tmp/hid.r{R}"
    H = torch.empty(N, D, dtype=torch.bfloat16)
    nb = (L1 + cfg.n_layers_per_attn_res_block) // cfg.n_layers_per_attn_res_block   # max block states needed
    BS = torch.empty(N, nb, D, dtype=torch.bfloat16)
    k_blocks = 0
    if L0 == 0:
        emb = M.Inkling2NormedEmbedding(cfg.vocab_size, D, None, cfg.rms_norm_eps)
        emb.load_state_dict({"weight": Q.raw("model.llm.embed.weight"), "embed_norm.weight": Q.raw("model.llm.embed_norm.weight")})
        emb = emb.to(dev, torch.bfloat16)
        for c0 in range(0, N, 65536):
            H[c0:c0 + 65536] = emb(data.tok[c0:c0 + 65536].to(dev)).cpu()
        del emb
        if MM and not os.environ.get("INK_MM_NOVIS"):
            splice_vision(data, H, dev, cfg)
    else:
        ck = json.load(open(f"{hp}.json")); assert ck["next_layer"] == L0, ck
        k_blocks = ck["k_blocks"]
        H.view(torch.int16).copy_(torch.from_numpy(np.fromfile(f"{hp}.H", np.int16)).view(N, D))
        if k_blocks:
            BS[:, :k_blocks].view(torch.int16).copy_(torch.from_numpy(np.fromfile(f"{hp}.BS", np.int16)).view(N, k_blocks, D))
    dist.barrier()
    val_rows = torch.nonzero(data.val)[:, 0]
    G = 1 if a.no_stats else a.groups
    gsz = NE // G
    assert gsz % W == 0
    win = cfg.sliding_window_size
    for L in range(L0, L1 + 1):
        t0 = time.time()
        lay = build_layer(cfg, L, dev)
        boundary = lay.is_block_boundary
        Res = torch.empty(N, D, dtype=torch.float32)
        X = torch.empty(N, D, dtype=torch.bfloat16)
        mcache = {}
        for s in data.segs:
            ix = torch.from_numpy(s["ix"]); n = s["n"]
            if n not in mcache:
                if len(mcache) > 8:
                    mcache.clear()
                mcache[n] = masks(n, dev, win)
            h = H[ix].to(dev)[None]
            bs = BS[ix, :k_blocks].to(dev)[None]
            res, x = attn_half(lay, cfg, h, bs, *mcache[n])
            Res[ix] = res[0].cpu(); X[ix] = x[0].cpu()
        del mcache
        torch.cuda.synchronize(); ta = time.time() - t0; t0 = time.time()
        Ex = WSel(a, L, dev) if a.wdir and os.path.exists(f"{a.wdir}/L{L}.relerr.json") and L < a.wmax else Q.LayerExperts(L, dev)
        do_stats = not a.no_stats
        g_acc = torch.zeros(NE, 6, FF, dtype=torch.float64, device=dev) if do_stats else None
        sal_acc = torch.zeros(NE, 9, 6, dtype=torch.float64, device=dev) if do_stats else None
        Cc = torch.zeros(D, D, device=dev) if do_stats else None
        Ca = torch.zeros(D, D, device=dev) if do_stats else None
        Y = torch.zeros(N, D, dtype=torch.float32)
        trace = [] if a.trace else None
        ev = dict(x=[], ids=[], p=[])
        xc_all = X[dc].to(dev) if (do_stats and len(dc)) else None
        vfiles = None
        hot_all = None
        if getattr(Ex, "float", False):
            hot_all = float_sets(a, L, lay, X, data, dev)
        for gi in range(G):
            e0, e1 = gi * gsz, (gi + 1) * gsz
            st = Stats(gsz, dev) if do_stats else None
            Wc = LazyW(Ex) if (a.no_stats and N <= a.chunk) else {e: Ex.w(e) for e in range(e0, e1)}   # bf16 group cache
            for c0 in range(0, N, a.chunk):
                c1 = min(N, c0 + a.chunk)
                x = X[c0:c1].to(dev)
                Yc = Y[c0:c1].to(dev)
                fm = data.fit[c0:c1].to(dev)
                if gi == 0 and do_stats:
                    gram_(Ca, x[fm].float()); gram_(Cc, x[ctxm[c0:c1].to(dev)].float())
                moe_pass(lay, Wc, x, Yc, st, e0, e1, fm, data.cat[c0:c1].to(dev), None, 1.0, g_acc, sal_acc,
                         trace if gi == 0 else None, None if hot_all is None else hot_all[c0:c1])
                Y[c0:c1] = Yc.cpu()
                if gi == 0:
                    vm = data.val[c0:c1]
                    if vm.any():
                        vmd = vm.to(dev)
                        tw, ti = route(lay, x[vmd])
                        ev["x"].append(x[vmd].cpu()); ev["ids"].append(ti.cpu()); ev["p"].append(tw.float().cpu())
            if do_stats and xc_all is not None:                 # ctx rows through every expert of the group (Dc, gdiag 4/5)
                for c0 in range(0, len(dc), 16384):
                    xc = xc_all[c0:c0 + 16384]
                    for e in range(e0, e1):
                        Wgu, Wd = Wc[e]
                        h, _, cg2, cu2 = teacher(xc, Wgu, Wd)
                        gram_(st.Dc[e - e0], h.float() * dc_scale ** 0.5)
                        g_acc[e, 4] += cg2.sum(0).double() * dc_scale; g_acc[e, 5] += cu2.sum(0).double() * dc_scale
            del Wc
            if do_stats:
                vfiles = write_group(a, L, st, gi, gsz, R, W)
                del st
                torch.cuda.empty_cache()
        del Ex, xc_all
        torch.cuda.synchronize(); tm = time.time() - t0; t0 = time.time()
        # post-MLP conv per segment, new hidden; block states
        if boundary:
            BS[:, k_blocks] = H
        for s in data.segs:
            ix = torch.from_numpy(s["ix"])
            y = Y[ix].to(dev).bfloat16()[None]                  # HF Experts: fp32 accumulate then .to(bf16)
            yc = lay.post_mlp_convolution(y, cache=None, conv_mask=None)
            H[ix] = (Res[ix].to(dev)[None] + yc).to(torch.bfloat16)[0].cpu()
        if boundary:
            k_blocks += 1
        del Res, X, Y
        if do_stats:
            write_meta(a, L, g_acc, sal_acc, Cc, Ca, R, W, dict(T_fit=T_fit, n_ctx=n_ctx, allt=allt, dc_cap=a.dc_cap,
                                                                 groups=G))
        write_eval(a, L, ev, data, val_rows, R, W)
        if trace is not None:
            np.savez(f"{a.trace}/L{L}.r{R}of{W}.npz", ids=torch.cat([t[0] for t in trace]).numpy().astype(np.uint16),
                     w=torch.cat([t[1] for t in trace]).numpy(), xn=torch.cat([t[2] for t in trace]).numpy())
        tw_ = time.time() - t0
        del lay
        torch.cuda.empty_cache()
        if L % 4 == 3 or L == L1:
            H.view(torch.int16).numpy().tofile(f"{hp}.H.tmp"); os.replace(f"{hp}.H.tmp", f"{hp}.H")
            if k_blocks:
                BS[:, :k_blocks].contiguous().view(torch.int16).numpy().tofile(f"{hp}.BS.tmp"); os.replace(f"{hp}.BS.tmp", f"{hp}.BS")
            json.dump(dict(next_layer=L + 1, k_blocks=k_blocks), open(f"{hp}.json", "w"))
        if R == 0:
            print(f"[capink] L{L} {cfg.layer_types[L]} attn {ta:.0f}s mlp {tm:.0f}s write {tw_:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G {time.strftime('%H:%M:%S')}", flush=True)
        dist.barrier()
    if L1 == cfg.num_hidden_layers - 1:
        final(a, cfg, H, BS, k_blocks, data, dev, R, W)
    dist.barrier()
    dist.destroy_process_group()


def write_group(a, L, st, gi, gsz, R, W):
    vd = f"{a.out}/stats/L{L}.v1"
    blk = gsz // W
    if R == 0 and gi == 0:
        os.makedirs(vd, exist_ok=True)
        for k, n in (("A2", D), ("A0", D), ("D2", FF), ("D0", FF), ("Dc", FF)):
            with open(f"{vd}/{k}.f32", "wb") as f:
                f.truncate(NE * stride_bytes(n))
    dist.barrier()
    for j in range(W):
        sl = slice(j * blk, (j + 1) * blk)
        for t in (st.A0, st.A2, st.D0, st.D2, st.Dc):
            dist.reduce(t[sl], dst=j)
    iu = {n: torch.triu_indices(n, n, device=st.A0.device) for n in (D, FF)}
    for k, t, n in (("A2", st.A2, D), ("A0", st.A0, D), ("D2", st.D2, FF), ("D0", st.D0, FF), ("Dc", st.Dc, FF)):
        sb = stride_bytes(n)
        fd = os.open(f"{vd}/{k}.f32", os.O_WRONLY)
        for jj in range(R * blk, (R + 1) * blk):
            e = gi * gsz + jj
            v = t[jj][iu[n][0], iu[n][1]].cpu().numpy()
            os.pwrite(fd, v.tobytes(), e * sb)
        os.close(fd)
    dist.barrier()


def write_meta(a, L, g_acc, sal_acc, Cc, Ca, R, W, info):
    for t in (Cc, Ca, g_acc, sal_acc):
        dist.reduce(t, dst=0)
    vd = f"{a.out}/stats/L{L}.v1"
    if R == 0:
        iu = torch.triu_indices(D, D, device=Cc.device)
        pk = lambda A: A[iu[0], iu[1]].cpu().numpy()
        np.save(f"{vd}/C_ctx.npy", pk(Cc)); np.save(f"{vd}/C_all.npy", pk(Ca))
        np.save(f"{vd}/gdiag.npy", g_acc.cpu().numpy())
        sal = sal_acc.cpu().numpy(); np.save(f"{vd}/sal.npy", sal)
        sc = sal[:, 0, :4].copy(); np.save(f"{vd}/scalars.npy", sc)
        ess = (sc[:, 2] ** 2 / np.maximum(sc[:, 3], 1e-300)).tolist()
        files = {k: dict(file=f"{k}.f32", rows=NE, stride_bytes=stride_bytes(n), packed=npk(n), n=n)
                 for k, n in (("A2", D), ("A0", D), ("D2", FF), ("D0", FF), ("Dc", FF))}
        meta = dict(schema="nestquant-19-stats-v2", layer=L, files=dict(raw=files), T_fit=info["T_fit"], n_ctx=info["n_ctx"],
                    n_routed=sc[:, 0].astype(int).tolist(), ess=ess,
                    shards=[dict(shard=0, T=info["T_fit"], n_ctx=info["n_ctx"], n_dc=int(sum(t[2] for t in info["allt"])),
                                 per_rank=info["allt"], seed=CTX_SEED, corpus="inkling c2048+traces (GLM-5.3 calib corpus re-templated)")],
                    protocol=dict(model="thinkingmachines/Inkling-2-Small (native NVFP4 L0-39 / MXFP8 L40-47, dequantised)",
                                  D=D, F=FF, NEXP=NE, sal_cats=SAL_CATS,
                                  sal_cols=["n", "sum_p", "sum_p2", "sum_p4", "sum_p_ynorm", "sum_ynorm"],
                                  grams="fp32 TF32 GEMM on bf16-exact rows; p2 side bf16 hi/lo",
                                  gdiag="softcapped SwiGLU derivatives (gate cap 4, up cap 25), fp32",
                                  dc=f"first min(n_ctx_r, {info['dc_cap']}) ctx rows per rank, scaled n_ctx_r/n_dc_r",
                                  ctx="randperm(T_r, seed CTX_SEED + rank)[:T_r // 4] per rank",
                                  groups=info["groups"]))
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
        if parts:
            out, off = {}, 0
            for k in ("x", "ids", "p", "token_positions", "bnd_think", "bnd_end"):
                out[k] = torch.cat([p[k] for p in parts])
            dids = []
            for p in parts:
                u, inv = torch.unique(p["document_ids"], return_inverse=True); dids.append(inv + off); off += len(u)
            out["document_ids"] = torch.cat(dids); out["domains"] = sum((p["domains"] for p in parts), [])
            out["layer"] = L; out["protocol"] = dict(role="evaluation only", model="Inkling-2-Small")
            os.makedirs(f"{a.out}/eval/val", exist_ok=True)
            torch.save(out, f"{a.out}/eval/val/layer_{L}.pt")
        for r in range(W):
            os.remove(f"{a.out}/tmp/ev.L{L}.r{r}.pt")
    dist.barrier()


def final(a, cfg, H, BS, k_blocks, data, dev, R, W):
    fin = M.Inkling2BlockAttentionResidual(cfg).to(dev)
    fin.query.data.copy_(Q.raw("model.llm.attn_res.output_query").float())
    norm = M.Inkling2RMSNorm(D, cfg.rms_norm_eps).to(dev)
    norm.weight.data.copy_(Q.raw("model.llm.norm.weight").float())
    norm = norm.bfloat16()
    head = Q.raw("model.llm.unembed.weight").to(dev, torch.bfloat16)
    V = cfg.unpadded_vocab_size
    os.makedirs(f"{a.out}/final", exist_ok=True)
    res, top = [], {}
    ref = torch.load(f"{a.ref}/final/top64.r{R}.pt") if a.ref else None
    for si, s in enumerate(data.segs):
        if s["n"] < 2:
            continue
        ix = torch.from_numpy(s["ix"])
        h = H[ix].to(dev)[None]; bs = BS[ix, :k_blocks].to(dev)[None]
        z = norm(fin(bs, h)[0].float()).to(torch.bfloat16)[0]
        lg = F.linear(z, head).float()
        lg = (torch.tanh(lg / cfg.final_logit_softcapping) * cfg.final_logit_softcapping)[:, :V]
        lp = F.log_softmax(lg, -1); tgt = data.tok[ix][1:].to(dev)
        ce = -lp[:-1].gather(1, tgt[:, None])[:, 0]; acc = (lp[:-1].argmax(-1) == tgt).float()
        pm = None
        if MM:                                   # score only predictions whose target is a text (caption) row
            pm = torch.zeros(s["n"], dtype=torch.bool, device=dev)
            pm[:-1] = torch.from_numpy(data.rk[ix.numpy()][1:] == 2).to(dev)
            if int(pm.sum()) < 1:
                continue
            ce, acc = ce[pm[:-1]], acc[pm[:-1]]
        n_eff = int(pm.sum()) + 1 if pm is not None else s["n"]
        r = dict(rank=R, seg=si, split=s["split"], n=n_eff, ce=float(ce.mean()), acc=float(acc.mean()))
        if ref is not None and si in ref:
            tv, ti = ref[si]["v"].to(dev).float(), ref[si]["i"].to(dev).long()
            pt = tv.exp(); qs = lp.gather(1, ti)
            kl = (pt * (tv - qs)).sum(1)
            rt = (1 - pt.sum(1)).clamp_min(1e-12); rq = (1 - qs.exp().sum(1)).clamp_min(1e-12)
            kl = kl + rt * (rt.log() - rq.log())
            if pm is not None:
                kl = kl[pm]; ti = ti[pm]; lpm = lp[pm]
            else:
                lpm = lp
            r.update(kld=float(kl.mean()), kld_n=int(kl.numel()), top1=float((ti[:, 0] == lpm.argmax(-1)).float().mean()))
        res.append(r)
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
                print(f"[capink] final {sp}: ppl {np.exp(sum(r['ce'] * (r['n'] - 1) for r in q) / nt):.3f} "
                      f"acc {sum(r['acc'] * (r['n'] - 1) for r in q) / nt:.4f} ({nt} tokens)", flush=True)
        k = [r for r in rr if "kld" in r]
        if k:
            nk = sum(r["kld_n"] for r in k)
            out = dict(kld=sum(r["kld"] * r["kld_n"] for r in k) / nk, top1=sum(r["top1"] * r["kld_n"] for r in k) / nk,
                       n=nk, wsel=a.wsel, wdir=a.wdir, ref=a.ref)
            json.dump(out, open(f"{a.out}/final/kld.json", "w"), indent=1)
            print(f"[capink] KLD vs ref: {out['kld']:.5f} top1 {out['top1']:.4f} ({nk} val tokens)", flush=True)


if __name__ == "__main__":
    main()
