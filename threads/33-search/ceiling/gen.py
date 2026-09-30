#!/usr/bin/env python3
"""T33l task 3: sample K continuations of 64 tokens per prefix from the FP8 GLM-5.3 reference and record the routing
(top-8 ids, combine weights incl. routed_scaling_factor, |x|^2 of the normalised MoE input) of every continuation
token at every sparse layer.  PRIVATE: all outputs stay under /tmp/nestquant/33-search/ceiling.

Single process, D GPUs:
  * backbone (attention, norms, router, shared expert, dense MLPs L0-2) resident on every GPU as block-FP8, dequantised
    per use exactly like T18 nq_io.fp8_dequant (bf16); embed + lm_head (fp32, like T18) resident;
  * routed experts expert-parallel: GPU g owns experts e % D == g, streamed per pass from the page cache by a pool of
    reader threads (pread -> pinned staging -> H2D), prefetched `--depth` layers ahead; as many layers as fit are kept
    resident;
  * MLA with the absorbed form over a latent cache (c = kv_a_layernorm(kv_a[:512]), k_rope roped): prefix cache
    shared by the K+1 sequences of a prefix, continuation cache per sequence; indexer skipped (exact, len <= 2048);
  * per prefix: K sampled continuations (temperature 1.0, top_p 0.95 = generation_config.json) + 1 teacher-forced
    real continuation (validation of the decode path against the T32 trace, and the real-text target).
  gen.py --prefixes prefixes.json --out DIR [--k 8] [--steps 64] [--n-layers 78] [--devices cuda:0,...]
        [--time-limit-min 80] [--resume]"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import nq_io  # noqa: E402

FP8_DIR = os.environ.get("NQ_FP8", "/tmp/nestquant/src/glm53-fp8")
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ raw weight access
class Raw:
    """thread-safe raw tensor reads (own fds + pinned staging per thread)."""

    def __init__(self, idx):
        self.map = idx.map
        self.tl = threading.local()

    def _fd(self, f):
        d = getattr(self.tl, "fd", None)
        if d is None:
            d = self.tl.fd = {}
        if f not in d:
            d[f] = os.open(f, os.O_RDONLY)
        return d[f]

    def _stage(self, n):
        s = getattr(self.tl, "stage", None)
        if s is None or s.numel() < n:
            s = torch.empty(max(n, 48 << 20), dtype=torch.uint8)
            s = self.tl.stage = s.pin_memory() if torch.cuda.is_available() else s
            self.tl.stream = {}
        return s

    def read_many(self, names, dev):
        """-> dict name -> tensor on dev (one packed H2D copy)."""
        metas = [self.map[n] for n in names]
        tot = sum((m[4] + 255) // 256 * 256 for m in metas)
        st = self._stage(tot)
        mv = memoryview(st.numpy())
        o = 0
        offs = []
        for (f, dt, shape, off, nb) in metas:
            fd = self._fd(f)
            got = 0
            while got < nb:
                got += os.preadv(fd, [mv[o + got:o + nb]], off + got)
            offs.append(o)
            o += (nb + 255) // 256 * 256
        if str(dev) == "cpu":
            buf = st[:tot].clone()
        else:
            ss = self.tl.stream.get(dev)
            if ss is None:
                ss = self.tl.stream[dev] = torch.cuda.Stream(device=dev)
            with torch.cuda.stream(ss):
                buf = torch.empty(tot, dtype=torch.uint8, device=dev)
                buf.copy_(st[:tot], non_blocking=True)
            ss.synchronize()
        out = {"__buf__": buf}
        for n, (f, dt, shape, off, nb), o in zip(names, metas, offs):
            out[n] = buf[o:o + nb].view(nq_io.DT[dt]).view(shape)
        return out


def lin(x, w, s=None):
    W = nq_io.fp8_dequant(w, s) if s is not None else w
    return F.linear(x, W.to(x.dtype))


def rms(x, w, eps):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return w * x.to(dt)


def rope_inter(x, cos, sin):
    """HF apply_rotary_pos_emb_interleave on x [..., T, 64] with cos/sin [T, 64] (cat(freqs,freqs))."""
    c = cos[..., : cos.shape[-1] // 2]; s = sin[..., : sin.shape[-1] // 2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# ------------------------------------------------------------------------------------------------ model
class Model:
    def __init__(self, a, devs):
        from transformers import AutoConfig
        self.cfg = cfg = AutoConfig.from_pretrained(FP8_DIR)
        self.devs = devs
        self.D = len(devs)
        self.nl = a.n_layers
        self.eps = cfg.rms_norm_eps
        self.idx = nq_io.SafeIndex(FP8_DIR)
        self.raw = Raw(self.idx)
        self.sparse = [cfg.mlp_layer_types[i] == "sparse" for i in range(self.nl)]
        self.H = cfg.num_attention_heads
        self.scale = cfg.qk_head_dim ** -0.5
        assert cfg.rope_parameters.get("rope_type", "default") == "default"
        self.E = cfg.n_routed_experts
        self.pool = cf.ThreadPoolExecutor(a.threads)
        # rotary
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding
        self.rot = [GlmMoeDsaRotaryEmbedding(cfg).to(d) for d in devs]
        # backbone per device
        self.bb = [[None] * self.nl for _ in devs]
        log(f"loading backbone on {self.D} devices, {self.nl} layers")
        futs = {}
        for g, d in enumerate(devs):
            for li in range(self.nl):
                futs[self.pool.submit(self._load_bb, li, d)] = (g, li)
        for f in cf.as_completed(futs):
            g, li = futs[f]
            self.bb[g][li] = f.result()
        glob_ = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
        self.glob = []
        for d in devs:
            t = {n: self.raw.read_many([n], d)[n] for n in glob_}
            self.glob.append({"emb": t[glob_[0]], "head": t[glob_[1]].float(), "norm": t[glob_[2]]})
            del t[glob_[1]]
        for d in devs:
            if str(d) != "cpu":
                torch.cuda.synchronize(d)
        log("backbone loaded")
        self.resident = {}               # (li) -> per-device dict e -> weights
        self.pending = {}
        self.cycle = [li for li in range(self.nl) if self.sparse[li]]
        self.depth = a.depth

    def _load_bb(self, li, d):
        pre = f"model.layers.{li}."
        names = [n for n in self.idx.names() if n.startswith(pre) and ".experts." not in n and "indexer" not in n]
        t = self.raw.read_many(names, d)
        return {n[len(pre):]: v for n, v in t.items() if n != "__buf__"}

    def W(self, g, li, name):
        b = self.bb[g][li]
        return b[name + ".weight"], b.get(name + ".weight_scale_inv")

    # -------------------------------------------------------------------------------------------- experts
    def _load_experts(self, li, g):
        d = self.devs[g]
        out = {}
        for e in range(g, self.E, self.D):
            p = f"model.layers.{li}.mlp.experts.{e}."
            names = [p + k + s for k in ("gate_proj", "up_proj", "down_proj") for s in (".weight", ".weight_scale_inv")]
            t = self.raw.read_many(names, d)
            out[e] = {k: (t[p + k + ".weight"], t[p + k + ".weight_scale_inv"]) for k in ("gate_proj", "up_proj", "down_proj")}
        return out

    def _load_layer_split(self, li, g, parts=4):
        es = list(range(g, self.E, self.D))
        chunks = [es[i::parts] for i in range(parts)]

        def one(ch):
            d = self.devs[g]
            out = {}
            for e in ch:
                p = f"model.layers.{li}.mlp.experts.{e}."
                names = [p + k + s for k in ("gate_proj", "up_proj", "down_proj") for s in (".weight", ".weight_scale_inv")]
                t = self.raw.read_many(names, d)
                out[e] = {k: (t[p + k + ".weight"], t[p + k + ".weight_scale_inv"]) for k in ("gate_proj", "up_proj", "down_proj")}
                out[e]["__buf__"] = t["__buf__"]
            return out
        return [self.pool.submit(one, ch) for ch in chunks]

    def make_resident(self, n_layers):
        todo = self.cycle[:n_layers]
        log(f"making {len(todo)} sparse layers' experts resident ({todo[:3]}..{todo[-1:]})")
        for li in todo:
            fs = [self._load_layer_split(li, g) for g in range(self.D)]
            self.resident[li] = [{k: v for f in fl for k, v in f.result().items()} for fl in fs]
        log("resident done")

    def schedule(self, li):
        if li in self.resident or li in self.pending:
            return
        self.pending[li] = [self._load_layer_split(li, g) for g in range(self.D)]

    def experts(self, li):
        if li in self.resident:
            return self.resident[li]
        self.schedule(li)
        fs = self.pending.pop(li)
        res = [{k: v for f in fl for k, v in f.result().items()} for fl in fs]
        for g, dd in enumerate(res):              # allocator must not recycle these blocks (loader stream) under
            if self.devs[g].type == "cuda":       # queued compute on the default stream
                cs = torch.cuda.current_stream(self.devs[g])
                for We in dd.values():
                    We["__buf__"].record_stream(cs)
        return res

    def prefetch_after(self, li):
        """schedule the next `depth` streamed sparse layers after li (cyclic over passes)."""
        streamed = [x for x in self.cycle if x not in self.resident]
        if not streamed:
            return
        i0 = next((j for j, x in enumerate(streamed) if x > li), 0)
        for k in range(self.depth):
            self.schedule(streamed[(i0 + k) % len(streamed)])

    # -------------------------------------------------------------------------------------------- attention
    def attn_proj(self, g, li, hn, pos):
        """hn [T, 6144] normed; pos [T] long -> q_abs [T,H,512], q_rope [T,H,64], c [T,512], kr [T,64]."""
        cfg = self.cfg
        b = self.bb[g][li]
        qr = rms(lin(hn, *self.W(g, li, "self_attn.q_a_proj")), b["self_attn.q_a_layernorm.weight"], self.eps)
        q = lin(qr, *self.W(g, li, "self_attn.q_b_proj")).view(-1, self.H, cfg.qk_head_dim)
        q_nope, q_rot = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], -1)
        ckv = lin(hn, *self.W(g, li, "self_attn.kv_a_proj_with_mqa"))
        c, kr = ckv.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], -1)
        c = rms(c, b["self_attn.kv_a_layernorm.weight"], self.eps)
        cos, sin = self.rot[g](hn[None, :, :1], pos[None])
        cos, sin = cos[0], sin[0]                                      # [T, 64]
        q_rot = rope_inter(q_rot.transpose(0, 1), cos, sin).transpose(0, 1)   # [T,H,64]
        kr = rope_inter(kr, cos, sin)
        Wkv = self.kvb(g, li)
        q_abs = torch.einsum("thd,hdc->thc", q_nope, Wkv[:, : cfg.qk_nope_head_dim])      # [T,H,512]
        return q_abs, q_rot.contiguous(), c.contiguous(), kr.contiguous()

    def kvb(self, g, li):
        cfg = self.cfg
        w, s = self.W(g, li, "self_attn.kv_b_proj")
        return nq_io.fp8_dequant(w, s).view(self.H, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)

    def attn_out(self, g, li, olat):
        """olat [T,H,512] -> o_proj(W_uv olat) [T,6144]."""
        cfg = self.cfg
        Wkv = self.kvb(g, li)
        v = torch.einsum("thc,hvc->thv", olat, Wkv[:, cfg.qk_nope_head_dim:])
        return lin(v.reshape(v.shape[0], -1), *self.W(g, li, "self_attn.o_proj"))

    # -------------------------------------------------------------------------------------------- moe
    def moe(self, li, xs, rec):
        """xs: per device normed MoE input [T_g, 6144] bf16 -> per device output (shared + routed) bf16.
        rec(g, ids, w, xn) callback for recording."""
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaTopkRouter
        ids, ws, outs = [], [], []
        for g, x in enumerate(xs):
            gate = self.gate(g, li)
            _, w, i = gate(x)
            ids.append(i); ws.append(w)
            rec(g, i, w, x.float().pow(2).sum(-1))
            b = self.bb[g][li]
            sh = lin(F.silu(lin(x, *self.W(g, li, "mlp.shared_experts.gate_proj"))) *
                     lin(x, *self.W(g, li, "mlp.shared_experts.up_proj")), *self.W(g, li, "mlp.shared_experts.down_proj"))
            outs.append(sh)
        EXP = self.experts(li)
        self.prefetch_after(li)
        sizes = [x.shape[0] for x in xs]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        partials = []
        for h, d in enumerate(self.devs):
            X = torch.cat([x.to(d, non_blocking=True) for x in xs])
            I = torch.cat([i.to(d, non_blocking=True) for i in ids])
            Wt = torch.cat([w.to(d, non_blocking=True) for w in ws])
            flat = I.reshape(-1)
            own = (flat % self.D) == h
            sel = torch.nonzero(own).squeeze(1)
            ex = flat[sel]
            order = torch.argsort(ex, stable=True)
            sel = sel[order]; ex = ex[order]
            cnt = torch.bincount(ex, minlength=self.E).cpu().tolist()
            tok = sel // I.shape[1]
            wt = Wt.reshape(-1)[sel]
            P = torch.zeros(X.shape[0], X.shape[1], dtype=torch.float32, device=d)
            o = 0
            for e in range(h, self.E, self.D):
                n = cnt[e]
                if n == 0:
                    continue
                t = tok[o:o + n]
                We = EXP[h][e]
                xe = X[t]
                y = lin(F.silu(lin(xe, *We["gate_proj"])) * lin(xe, *We["up_proj"]), *We["down_proj"])
                P.index_add_(0, t, y.float() * wt[o:o + n, None].float())
                o += n
            partials.append(P)
        del EXP
        res = []
        for g, d in enumerate(self.devs):
            acc = outs[g].float()
            for h in range(self.D):
                acc = acc + partials[h][offs[g]:offs[g + 1]].to(d, non_blocking=True)
            res.append(acc.to(torch.bfloat16))
        return res

    _gates = {}

    def gate(self, g, li):
        k = (g, li)
        if k not in self._gates:
            from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaTopkRouter
            with torch.device("meta"):
                gx = GlmMoeDsaTopkRouter(self.cfg)
            b = self.bb[g][li]
            gx.load_state_dict({"weight": b["mlp.gate.weight"], "e_score_correction_bias": b["mlp.gate.e_score_correction_bias"]},
                               assign=True)
            self._gates[k] = gx
        return self._gates[k]

    def dense(self, g, li, x):
        return lin(F.silu(lin(x, *self.W(g, li, "mlp.gate_proj"))) * lin(x, *self.W(g, li, "mlp.up_proj")),
                   *self.W(g, li, "mlp.down_proj"))

    def logits(self, g, h):
        x = rms(h, self.glob[g]["norm"], self.eps)
        return x.float() @ self.glob[g]["head"].T


# ------------------------------------------------------------------------------------------------ run
def sample_top_p(logits, gen, top_p=0.95, temp=1.0, n=1):
    """logits [B, V] fp32 -> [B, n] sampled ids (HF top-p: keep smallest set with cum prob >= top_p)."""
    p = torch.softmax(logits / temp, -1)
    sp, si = torch.sort(p, -1, descending=True)
    cs = sp.cumsum(-1)
    drop = (cs - sp) > top_p
    sp = sp.masked_fill(drop, 0)
    sp = sp / sp.sum(-1, keepdim=True)
    j = torch.multinomial(sp, n, replacement=True, generator=gen)
    return si.gather(1, j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--devices", default=",".join(f"cuda:{i}" for i in range(8)))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--resident-layers", type=int, default=-1, help="-1: auto from free memory")
    ap.add_argument("--reserve-gb", type=float, default=14.0)
    ap.add_argument("--time-limit-min", type=float, default=80.0)
    ap.add_argument("--max-prefixes", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=33)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    devs = [torch.device(d) for d in a.devices.split(",")]
    D = len(devs)
    PX = json.load(open(a.prefixes))
    if a.max_prefixes:
        PX = PX[: a.max_prefixes]
    K1 = a.k + 1
    # assignment: prefix i -> device i % D
    per = [[i for i in range(len(PX)) if i % D == g] for g in range(D)]
    M = Model(a, devs)
    sp_layers = [li for li in range(M.nl) if M.sparse[li]]
    nsp = len(sp_layers)
    spi = {li: j for j, li in enumerate(sp_layers)}
    st = []                                   # per-device state
    for g, d in enumerate(devs):
        pl = per[g]
        Ps = [len(PX[i]["prefix"]) for i in pl]
        Pmax = max(Ps)
        s = dict(pl=pl, Ps=torch.tensor(Ps, device=d), Pmax=Pmax, nseq=len(pl) * K1,
                 pc=torch.zeros(M.nl, len(pl), Pmax, 576, dtype=torch.bfloat16, device=d),
                 cc=torch.zeros(M.nl, len(pl) * K1, a.steps, 576, dtype=torch.bfloat16, device=d),
                 tok=torch.zeros(len(pl) * K1, a.steps, dtype=torch.long, device=d),
                 ids=torch.zeros(nsp, len(pl) * K1, a.steps, 8, dtype=torch.uint8, device=d),
                 w=torch.zeros(nsp, len(pl) * K1, a.steps, 8, dtype=torch.float32, device=d),
                 xn=torch.zeros(nsp, len(pl) * K1, a.steps, dtype=torch.float32, device=d),
                 ent=torch.zeros(len(pl) * K1, a.steps, dtype=torch.float32, device=d),
                 pids=torch.zeros(nsp, len(pl), 64, 8, dtype=torch.uint8, device=d),
                 px=torch.zeros(nsp, len(pl), 6144, dtype=torch.bfloat16, device=d),
                 pw=torch.zeros(nsp, len(pl), 8, dtype=torch.float32, device=d),
                 pxn=torch.zeros(nsp, len(pl), dtype=torch.float32, device=d),
                 gen=torch.Generator(device=d).manual_seed(a.seed * 1000 + g), step=0)
        st.append(s)
    mem = [torch.cuda.mem_get_info(d) if d.type == "cuda" else (0, 0) for d in devs]
    per_layer = 32 * 3 * (2048 * 6144) * 1.0 * (8 / D) / 2**30 * 1.02
    if a.resident_layers < 0:
        free = min(m[0] for m in mem) / 2**30 if devs[0].type == "cuda" else 0
        nres = max(0, int((free - a.reserve_gb - (a.depth + 1) * per_layer) / per_layer))
    else:
        nres = a.resident_layers
    log(f"free GB per device {[round(m[0] / 2**30, 1) for m in mem]}; expert GB/layer/device {per_layer:.2f}; resident {nres}")
    ck = f"{a.out}/ckpt.pt"
    if a.resume and os.path.exists(ck):
        C = torch.load(ck, map_location="cpu")
        for g, s in enumerate(st):
            for k in ("pc", "cc", "tok", "ids", "w", "xn", "ent", "pids", "px", "pw", "pxn"):
                s[k].copy_(C[g][k])
            s["gen"].set_state(C[g]["gen"])
            s["step"] = C[g]["step"]
        log(f"resumed at step {st[0]['step']}")
        del C
    M.make_resident(min(nres, len(M.cycle)))
    for li in [x for x in M.cycle if x not in M.resident][: a.depth]:
        M.schedule(li)

    # ---------------------------------------------------------------------------------------- prefill
    def prefill():
        t0 = time.time()
        hs, poss = [], []
        for g, s in enumerate(st):
            d = devs[g]
            toks = torch.cat([torch.tensor(PX[i]["prefix"], dtype=torch.long) for i in s["pl"]]).to(d)
            poss.append(torch.cat([torch.arange(len(PX[i]["prefix"])) for i in s["pl"]]).to(d))
            hs.append(F.embedding(toks, M.glob[g]["emb"]).to(torch.bfloat16))
        for li in range(M.nl):
            xs = []
            for g, s in enumerate(st):
                b = M.bb[g][li]
                hn = rms(hs[g], b["input_layernorm.weight"], M.eps)
                qa, qr, c, kr = M.attn_proj(g, li, hn, poss[g])
                olat = torch.empty(qa.shape, dtype=qa.dtype, device=qa.device)
                o = 0
                for j, P in enumerate(s["Ps"].tolist()):
                    sl = slice(o, o + P)
                    s["pc"][li, j, :P, :512] = c[sl]; s["pc"][li, j, :P, 512:] = kr[sl]
                    for q0 in range(0, P, 512):
                        q1 = min(P, q0 + 512)
                        sc = (torch.einsum("thc,sc->hts", qa[o + q0:o + q1], c[sl]) +
                              torch.einsum("thr,sr->hts", qr[o + q0:o + q1], kr[sl])).float() * M.scale
                        mask = torch.arange(P, device=sc.device)[None, :] > torch.arange(q0, q1, device=sc.device)[:, None]
                        sc = sc.masked_fill(mask[None], float("-inf"))
                        pr = torch.softmax(sc, -1).to(torch.bfloat16)
                        olat[o + q0:o + q1] = torch.einsum("hts,sc->thc", pr, c[sl])
                        del sc, pr
                    o += P
                hs[g] = hs[g] + M.attn_out(g, li, olat)
                del qa, qr, c, kr, olat
                x = rms(hs[g], b["post_attention_layernorm.weight"], M.eps)
                if M.sparse[li]:
                    xs.append(x)
                else:
                    hs[g] = hs[g] + M.dense(g, li, x)
            if M.sparse[li]:
                j = spi[li]

                def rec(g, i, w, xn, j=j):
                    s = st[g]
                    ends = torch.cumsum(s["Ps"], 0)
                    for q, e in enumerate(ends.tolist()):
                        s["pids"][j, q] = i[e - 64:e].to(torch.uint8)
                        s["pw"][j, q] = w[e - 1].float(); s["pxn"][j, q] = xn[e - 1]
                        s["px"][j, q] = xs[g][e - 1]
                outs = M.moe(li, xs, rec)
                for g in range(D):
                    hs[g] = hs[g] + outs[g]
                del outs, xs
            if li % 10 == 0:
                log(f"prefill layer {li} {time.time() - t0:.0f}s")
        # logits at the last prefix position -> first continuation token
        for g, s in enumerate(st):
            ends = torch.cumsum(s["Ps"], 0) - 1
            lg = M.logits(g, hs[g][ends])                           # [npref, V]
            smp = sample_top_p(lg, s["gen"], n=a.k)                 # [npref, k]
            real = torch.tensor([PX[i]["cont"][0] for i in s["pl"]], device=devs[g])
            s["tok"][:, 0] = torch.cat([smp, real[:, None]], 1).reshape(-1)
            lp = torch.log_softmax(lg, -1)
            s["ent"][:, 0] = (-(lp.exp() * lp).sum(-1)).repeat_interleave(K1)
        log(f"prefill done {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------------------------------- decode
    def step(t):
        t0 = time.time()
        hs, poss = [], []
        for g, s in enumerate(st):
            d = devs[g]
            hs.append(F.embedding(s["tok"][:, t], M.glob[g]["emb"]).to(torch.bfloat16))
            poss.append(s["Ps"].repeat_interleave(K1) + t)
        for li in range(M.nl):
            xs = []
            for g, s in enumerate(st):
                b = M.bb[g][li]
                npf = len(s["pl"])
                hn = rms(hs[g], b["input_layernorm.weight"], M.eps)
                qa, qr, c, kr = M.attn_proj(g, li, hn, poss[g])        # [nseq,H,*]
                s["cc"][li, :, t, :512] = c; s["cc"][li, :, t, 512:] = kr
                q = torch.cat([qa, qr], -1)                                # [nseq,H,576]
                qp = q.view(npf, K1 * M.H, 576)
                sp_ = torch.bmm(qp, s["pc"][li].transpose(1, 2)).float() * M.scale      # [npf, K1*H, Pmax]
                pm = torch.arange(s["Pmax"], device=sp_.device)[None, :] >= s["Ps"][:, None]
                sp_ = sp_.masked_fill(pm[:, None, :], float("-inf"))
                sc_ = torch.bmm(q, s["cc"][li, :, : t + 1].transpose(1, 2)).float() * M.scale   # [nseq,H,t+1]
                allsc = torch.cat([sp_.view(npf * K1, M.H, -1), sc_], -1)
                pr = torch.softmax(allsc, -1).to(torch.bfloat16)
                pp = pr[..., : s["Pmax"]].reshape(npf, K1 * M.H, s["Pmax"])
                olat = torch.bmm(pp, s["pc"][li][..., :512]).view(npf * K1, M.H, 512)
                olat = olat + torch.bmm(pr[..., s["Pmax"]:], s["cc"][li, :, : t + 1, :512])
                hs[g] = hs[g] + M.attn_out(g, li, olat)
                x = rms(hs[g], b["post_attention_layernorm.weight"], M.eps)
                if M.sparse[li]:
                    xs.append(x)
                else:
                    hs[g] = hs[g] + M.dense(g, li, x)
            if M.sparse[li]:
                j = spi[li]

                def rec(g, i, w, xn, j=j):
                    st[g]["ids"][j, :, t] = i.to(torch.uint8); st[g]["w"][j, :, t] = w.float(); st[g]["xn"][j, :, t] = xn
                outs = M.moe(li, xs, rec)
                for g in range(D):
                    hs[g] = hs[g] + outs[g]
                del outs, xs
        if t + 1 < a.steps:
            for g, s in enumerate(st):
                lg = M.logits(g, hs[g])
                lp = torch.log_softmax(lg, -1)
                s["ent"][:, t + 1] = -(lp.exp() * lp).sum(-1)
                smp = sample_top_p(lg, s["gen"])[:, 0]
                real = torch.tensor([PX[i]["cont"][t + 1] for i in s["pl"]], device=devs[g])
                nt = smp.view(-1, K1).clone()
                nt[:, -1] = real
                s["tok"][:, t + 1] = nt.reshape(-1)
        for s in st:
            s["step"] = t + 1
        log(f"step {t} {time.time() - t0:.1f}s")

    def save(final):
        C = [{k: (s[k].cpu() if torch.is_tensor(s[k]) else s[k]) for k in
              ("pc", "cc", "tok", "ids", "w", "xn", "ent", "pids", "px", "pw", "pxn", "step")} | {"gen": s["gen"].get_state()}
             for s in st]
        if final:
            out = dict(order=np.concatenate([s["pl"] for s in st]), k=a.k, steps=a.steps, sparse_layers=np.array(sp_layers))
            for k in ("tok", "ids", "w", "xn", "ent", "pids", "px", "pw", "pxn"):
                out[k] = np.concatenate([(c[k].float() if c[k].dtype == torch.bfloat16 else c[k]).numpy() for c in C],
                                        axis=1 if k in ("ids", "w", "xn", "pids", "px", "pw", "pxn") else 0)
            np.savez(f"{a.out}/gen.npz", **out)
            log(f"final -> {a.out}/gen.npz")
        else:
            torch.save(C, ck + ".part"); os.replace(ck + ".part", ck)
            log(f"checkpoint at step {st[0]['step']} -> {ck}")

    if st[0]["step"] == 0 and not (a.resume and os.path.exists(ck)):
        prefill()
    t = st[0]["step"]
    t_step = time.time()
    while t < a.steps:
        step(t)
        t += 1
        el = (time.time() - T0) / 60
        last = (time.time() - t_step) / 60
        t_step = time.time()
        if t < a.steps and el + 1.5 * last + 4.0 > a.time_limit_min:
            save(False)
            log("time limit: exit 3 (resume with --resume)")
            os._exit(3)
    save(True)
    os._exit(0)


if __name__ == "__main__":
    main()
