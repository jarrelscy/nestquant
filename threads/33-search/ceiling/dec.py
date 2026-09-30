#!/usr/bin/env python3
"""T33l fp8dec: long-context FP8 GLM-5.3 decoder / KV-carry teacher-forcer with the DSA indexer, recording the
routing (top-8 ids, combine weights incl. routed_scaling_factor, |x|^2 of the normalised MoE input) of EVERY token
(prompt + decode) at every sparse layer, in full context.  PRIVATE: outputs stay on this box.

vs gen.py (T33l ceiling, <=2048 exact indexer skip):
  * DSA indexer implemented (HF GlmMoeDsaIndexer semantics: interleaved-rope q/k, LayerNorm k, relu(q.k)*d^-.5
    weighted by weights_proj*H^-.5, causal top-2048; 'shared' layers reuse the previous 'full' layer's selection);
    attention = MLA absorbed form over the gathered top-k latent rows (== dense causal when len <= 2048);
  * flat (unpadded) per-device latent + indexer-key caches, variable prompt lengths, LPT device balancing;
  * prefill layer-major over all prompts (query chunks, token-chunked MoE, experts fetched once per layer);
  * expert streaming: --stream page (default) = pageable mmap -> H2D per device (sbench: 21 GB/s, bitwise OK);
    --stream reg = cudaHostRegister(Portable|ReadOnly) of the mmap'd safetensors files
    (refcounted, unregistered after the layer's copies complete; GPU DMAs straight from the page cache) or
    --stream pread (gen.py path, --threads readers); one host sync per layer for the expert counts;
  * decode: temperature 1.0 / top_p 0.95 (generation_config), seeded per device; natural stop on any stop id
    (<|endoftext|>/<|user|>/<|observation|>), --n-dec = cap, real per-task decode length in index.json;
    --mode tf: teacher-force task["tf"] tokens (prefill only; KV-carry capture);
  * checkpoint/resume of the full state (caches + records) across GPU holds.
  dec.py --tasks tasks.json --out DIR [--mode gen|tf] [--n-dec 2048] [--stream reg|pread] ...
tasks.json: [{"id": str, "prompt": [ids]} (+ "tf": [ids] for --mode tf)]"""
import argparse
import concurrent.futures as cf
import json
import mmap
import os
import sys
import threading
import time
import atexit

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_io  # noqa: E402
import gen as G  # noqa: E402
from gen import lin, rms, rope_inter, log, sample_top_p  # noqa: E402

TOPK = 2048


# ------------------------------------------------------------------------------------------------ host registration
class Reg:
    """refcounted cudaHostRegister of whole mmap'd safetensors files (read-only, portable)."""

    def __init__(self):
        self.L = threading.Lock()
        self.ent = {}
        self.cr = torch.cuda.cudart()
        self.t_reg = 0.0
        atexit.register(self.close_all)

    def get(self, f):
        with self.L:
            e = self.ent.get(f)
            mine = e is None
            if mine:
                e = self.ent[f] = {"ev": threading.Event(), "ref": 0, "err": None}
            e["ref"] += 1
        if mine:
            try:
                fd = os.open(f, os.O_RDONLY)
                m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                os.close(fd)
                a = np.frombuffer(m, dtype=np.uint8)
                t0 = time.time()
                err = self.cr.cudaHostRegister(a.ctypes.data, a.nbytes, 0x01 | 0x08)
                self.t_reg += time.time() - t0
                if int(err) != 0:
                    raise RuntimeError(f"cudaHostRegister {f}: {err}")
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    e.update(mm=m, a=a, host=torch.from_numpy(a), ptr=a.ctypes.data)
            except Exception as ex:          # noqa: BLE001
                e["err"] = ex
            e["ev"].set()
        else:
            e["ev"].wait()
        if e["err"] is not None:
            raise e["err"]
        return e["host"]

    def put(self, f):
        with self.L:
            e = self.ent[f]
            e["ref"] -= 1
            if e["ref"] > 0:
                return
            del self.ent[f]
        self._drop(e)

    def _drop(self, e):
        if "ptr" in e:
            self.cr.cudaHostUnregister(e["ptr"])
            del e["host"], e["a"]
            try:
                e["mm"].close()
            except BufferError:
                pass

    def close_all(self):
        with self.L:
            es = list(self.ent.values()); self.ent.clear()
        for e in es:
            self._drop(e)

    def live_gb(self):
        with self.L:
            return sum(e["a"].nbytes for e in self.ent.values() if "a" in e) / 2**30


# ------------------------------------------------------------------------------------------------ model
class DModel(G.Model):
    def __init__(self, a, devs):
        self.stream_mode = a.stream
        self.reg = Reg() if a.stream == "reg" else None
        super().__init__(a, devs)
        cfg = self.cfg
        self.full = [t == "full" for t in cfg.indexer_types[: self.nl]]
        self.mtp = None
        if getattr(a, "mtp", False):                 # MTP layer 78 (draft for speculative decoding)
            self.mtp = cfg.num_hidden_layers
            for g, d in enumerate(devs):
                self.bb[g].append(self._load_bb(self.mtp, d))
            self.sparse = self.sparse + [True]
            self.full = self.full + [True]            # own indexer (keys over the MTP cache)
            self.cycle.append(self.mtp)
        self.nlc = self.nl + (1 if self.mtp is not None else 0)     # cache layers
        self.fidx = {}
        for li in range(self.nlc):
            if self.full[li]:
                self.fidx[li] = len(self.fidx)
        self.nfull = len(self.fidx)
        self.IH, self.ID = cfg.index_n_heads, cfg.index_head_dim
        self.lp = cf.ThreadPoolExecutor(self.D)      # loader threads (reg mode)
        self.ls = [torch.cuda.Stream(device=d) for d in devs]
        self.t_wait = 0.0

    _dq = {}

    def Wd(self, g, li, name):
        """dequantised bf16 weight (cached until clear_dq(); bitwise == lin()'s on-the-fly dequant)."""
        k = (g, li, name)
        if k not in self._dq:
            w, s = self.W(g, li, name)
            self._dq[k] = nq_io.fp8_dequant(w, s) if s is not None else w
        return self._dq[k]

    def clear_dq(self):
        self._dq.clear()

    def L(self, x, g, li, name):
        return F.linear(x, self.Wd(g, li, name).to(x.dtype))

    def kvb(self, g, li):
        cfg = self.cfg
        return self.Wd(g, li, "self_attn.kv_b_proj").view(self.H, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)

    def attn_out(self, g, li, olat):
        cfg = self.cfg
        v = torch.einsum("thc,hvc->thv", olat, self.kvb(g, li)[:, cfg.qk_nope_head_dim:])
        return self.L(v.reshape(v.shape[0], -1), g, li, "self_attn.o_proj")

    def dense(self, g, li, x):
        return self.L(F.silu(self.L(x, g, li, "mlp.gate_proj")) * self.L(x, g, li, "mlp.up_proj"), g, li, "mlp.down_proj")

    def _load_bb(self, li, d):
        pre = f"model.layers.{li}."
        names = [n for n in self.idx.names() if n.startswith(pre) and ".experts." not in n]
        t = self.raw.read_many(names, d)
        return {n[len(pre):]: v for n, v in t.items() if n != "__buf__"}

    # -------------------------------------------------------------------------------------------- experts
    def _exp_names(self, li, g):
        return [(e, k, s, f"model.layers.{li}.mlp.experts.{e}.{k}{s}") for e in range(g, self.E, self.D)
                for k in ("gate_proj", "up_proj", "down_proj") for s in (".weight", ".weight_scale_inv")]

    def _load_reg(self, li, g):
        d = self.devs[g]
        ns = self._exp_names(li, g)
        metas = [self.idx.map[n] for *_, n in ns]
        offs, o = [], 0
        for m in metas:
            offs.append(o); o += (m[4] + 255) // 256 * 256
        files = sorted({m[0] for m in metas})
        hosts = {f: self.reg.get(f) for f in files}
        s = self.ls[g]
        with torch.cuda.stream(s):
            buf = torch.empty(o, dtype=torch.uint8, device=d)      # allocated on the loader stream (like gen.py)
            for (f, dt, shape, off, nb), bo in zip(metas, offs):
                buf[bo:bo + nb].copy_(hosts[f][off:off + nb], non_blocking=True)
        s.synchronize()
        del hosts
        for f in files:
            self.reg.put(f)
        out = {}
        for (e, k, sfx, n), (f, dt, shape, off, nb), bo in zip(ns, metas, offs):
            t = buf[bo:bo + nb].view(nq_io.DT[dt]).view(shape)
            out.setdefault(e, {"__buf__": buf}).setdefault(k, [None, None])[0 if sfx == ".weight" else 1] = t
        for e in out:
            for k in ("gate_proj", "up_proj", "down_proj"):
                out[e][k] = tuple(out[e][k])
        return out

    _mm = {}
    _mml = threading.Lock()

    def _mmap(self, f):
        with self._mml:
            if f not in self._mm:
                fd = os.open(f, os.O_RDONLY)
                self._mm[f] = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                os.close(fd)
            return self._mm[f]

    def _load_page(self, li, g):
        """pageable mmap (page cache) -> H2D via driver staging, one loader thread per device (sbench mode P:
        21 GB/s aggregate, bitwise == pread; no pinned host memory, no registration)."""
        d = self.devs[g]
        ns = self._exp_names(li, g)
        metas = [self.idx.map[n] for *_, n in ns]
        offs, o = [], 0
        for m in metas:
            offs.append(o); o += (m[4] + 255) // 256 * 256
        s = self.ls[g]
        with torch.cuda.stream(s):
            buf = torch.empty(o, dtype=torch.uint8, device=d)
            for (f, dt, shape, off, nb), bo in zip(metas, offs):
                host = torch.from_numpy(np.frombuffer(self._mmap(f), dtype=np.uint8, count=nb, offset=off))
                buf[bo:bo + nb].copy_(host)
        s.synchronize()
        out = {}
        for (e, k, sfx, n), (f, dt, shape, off, nb), bo in zip(ns, metas, offs):
            t = buf[bo:bo + nb].view(nq_io.DT[dt]).view(shape)
            out.setdefault(e, {"__buf__": buf}).setdefault(k, [None, None])[0 if sfx == ".weight" else 1] = t
        for e in out:
            for k in ("gate_proj", "up_proj", "down_proj"):
                out[e][k] = tuple(out[e][k])
        return out

    def _load_layer_split(self, li, g, parts=4):
        if self.stream_mode == "reg":
            return [self.lp.submit(self._load_reg, li, g)]
        if self.stream_mode == "page":
            return [self.lp.submit(self._load_page, li, g)]
        return super()._load_layer_split(li, g, parts=max(1, self.pool._max_workers // self.D))

    def experts(self, li):
        t0 = time.time()
        r = super().experts(li)
        self.t_wait += time.time() - t0
        return r

    # -------------------------------------------------------------------------------------------- attention pieces
    def qkv(self, g, li, hn, pos):
        """-> q [T,H,576] (absorbed nope | roped), c [T,512], kr [T,64], qres [T,2048], cos, sin."""
        cfg = self.cfg
        b = self.bb[g][li]
        qres = rms(self.L(hn, g, li, "self_attn.q_a_proj"), b["self_attn.q_a_layernorm.weight"], self.eps)
        q = self.L(qres, g, li, "self_attn.q_b_proj").view(-1, self.H, cfg.qk_head_dim)
        q_nope, q_rot = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], -1)
        ckv = self.L(hn, g, li, "self_attn.kv_a_proj_with_mqa")
        c, kr = ckv.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], -1)
        c = rms(c, b["self_attn.kv_a_layernorm.weight"], self.eps)
        cos, sin = self.rot[g](hn[None, :, :1], pos[None])
        cos, sin = cos[0], sin[0]
        q_rot = rope_inter(q_rot.transpose(0, 1), cos, sin).transpose(0, 1)
        kr = rope_inter(kr, cos, sin)
        Wkv = self.kvb(g, li)
        q_abs = torch.einsum("thd,hdc->thc", q_nope, Wkv[:, : cfg.qk_nope_head_dim])
        return torch.cat([q_abs, q_rot], -1), c, kr, qres, cos, sin

    def idx_k(self, g, li, hn, cos, sin):
        """indexer key [T,128] (LayerNorm(wk hn), rope on the first 64 dims, interleaved)."""
        b = self.bb[g][li]
        k = self.L(hn, g, li, "self_attn.indexer.wk")
        k = F.layer_norm(k, (self.ID,), b["self_attn.indexer.k_norm.weight"], b["self_attn.indexer.k_norm.bias"], 1e-6)
        kr_, kp = k.split([self.cfg.qk_rope_head_dim, self.ID - self.cfg.qk_rope_head_dim], -1)
        return torch.cat([rope_inter(kr_, cos, sin), kp], -1)

    def idx_q(self, g, li, hn, qres, cos, sin):
        """indexer query [T,IH,128] and head weights [T,IH] fp32 (incl. IH^-0.5)."""
        b = self.bb[g][li]
        q = self.L(qres, g, li, "self_attn.indexer.wq_b").view(-1, self.IH, self.ID)
        qr_, qp = q.split([self.cfg.qk_rope_head_dim, self.ID - self.cfg.qk_rope_head_dim], -1)
        qr_ = rope_inter(qr_.transpose(0, 1), cos, sin).transpose(0, 1)
        w = F.linear(hn.to(b["self_attn.indexer.weights_proj.weight"].dtype), b["self_attn.indexer.weights_proj.weight"])
        return torch.cat([qr_, qp], -1), w.float() * self.IH ** -0.5

    # -------------------------------------------------------------------------------------------- moe
    def moe_chunked(self, li, xs, rec, chunk=16384):
        """xs per device [T_g, 6144]; experts fetched once; shared + routed in token chunks -> per device bf16
        (fp32 shared + fp32 partials summed, then bf16: == gen.py numerics)."""
        ids, ws = [], []
        for g, x in enumerate(xs):
            gate = self.gate(g, li)
            i_l, w_l, xn_l = [], [], []
            for c0 in range(0, x.shape[0], chunk):
                xc = x[c0:c0 + chunk]
                _, w, i = gate(xc)
                i_l.append(i); w_l.append(w); xn_l.append(xc.float().pow(2).sum(-1))
            i, w = torch.cat(i_l), torch.cat(w_l)
            rec(g, i, w, torch.cat(xn_l))
            ids.append(i); ws.append(w)
        EXP = self.experts(li)
        self.prefetch_after(li)
        outs = [torch.empty_like(x) for x in xs]
        nmax = max(x.shape[0] for x in xs)
        for c0 in range(0, nmax, chunk):
            xc = [x[c0:c0 + chunk] for x in xs]
            ic = [i[c0:c0 + chunk] for i in ids]
            wc = [w[c0:c0 + chunk] for w in ws]
            sizes = [x.shape[0] for x in xc]
            offs = np.concatenate([[0], np.cumsum(sizes)])
            prep = []
            for h, d in enumerate(self.devs):                 # phase 1: gather + counts on every device (async)
                X = torch.cat([x.to(d, non_blocking=True) for x in xc])
                I = torch.cat([i.to(d, non_blocking=True) for i in ic])
                Wt = torch.cat([w.to(d, non_blocking=True) for w in wc])
                flat = I.reshape(-1)
                sel = torch.nonzero((flat % self.D) == h).squeeze(1)
                ex = flat[sel]
                order = torch.argsort(ex, stable=True)
                sel = sel[order]; ex = ex[order]
                prep.append((X, sel // I.shape[1], Wt.reshape(-1)[sel], torch.bincount(ex, minlength=self.E)))
            cnts = [p[3].cpu().tolist() for p in prep]        # one host sync per device, all queued already
            partials = []
            for h, d in enumerate(self.devs):                 # phase 2: expert FFNs
                X, tok, wt, _ = prep[h]
                P = torch.zeros(X.shape[0], X.shape[1], dtype=torch.float32, device=d)
                o = 0
                for e in range(h, self.E, self.D):
                    n = cnts[h][e]
                    if n == 0:
                        continue
                    t = tok[o:o + n]
                    We = EXP[h][e]
                    xe = X[t]
                    y = lin(F.silu(lin(xe, *We["gate_proj"])) * lin(xe, *We["up_proj"]), *We["down_proj"])
                    P.index_add_(0, t, y.float() * wt[o:o + n, None].float())
                    o += n
                partials.append(P)
            for g, d in enumerate(self.devs):
                if sizes[g] == 0:
                    continue
                x = xc[g]
                acc = self.L(F.silu(self.L(x, g, li, "mlp.shared_experts.gate_proj")) *
                             self.L(x, g, li, "mlp.shared_experts.up_proj"), g, li, "mlp.shared_experts.down_proj").float()
                for h in range(self.D):
                    acc = acc + partials[h][offs[g]:offs[g + 1]].to(d, non_blocking=True)
                outs[g][c0:c0 + chunk] = acc.to(torch.bfloat16)
            del prep, partials
        del EXP
        return outs


# ------------------------------------------------------------------------------------------------ attention kernels
def sel_topk(qi, wi, KI, gidx, valid):
    """qi [n,IH,128], wi [n,IH], KI flat [N,128], gidx [n,L] flat rows, valid [n,L] -> (pos [n,k], ok [n,k])."""
    Kg = KI[gidx]                                                     # [n,L,128]
    sc = torch.relu(torch.bmm(qi.float(), Kg.float().transpose(1, 2)) * (qi.shape[-1] ** -0.5))   # [n,IH,L]
    isc = torch.bmm(wi[:, None, :], sc)[:, 0]                          # [n,L]
    isc = isc.masked_fill(~valid, float("-inf"))
    k = min(TOPK, isc.shape[1])
    v, p = isc.topk(k, dim=-1)
    return p, torch.isfinite(v)


def sel_topk_seq(qi, wi, Ks, valid):
    """one sequence's query chunk: qi [n,IH,128], wi [n,IH], Ks [L,128] its keys, valid [n,L] -> (pos, ok) [n,k]."""
    sc = torch.relu(torch.einsum("nhd,ld->nhl", qi.float(), Ks.float()) * (qi.shape[-1] ** -0.5))
    isc = torch.bmm(wi[:, None, :], sc)[:, 0]
    del sc
    isc = isc.masked_fill(~valid, float("-inf"))
    v, p = isc.topk(min(TOPK, isc.shape[1]), dim=-1)
    return p, torch.isfinite(v)


def attend(q, Cl, fi, ok, scale):
    """q [n,H,576], Cl flat [N,576], fi [n,k] flat rows, ok [n,k] -> olat [n,H,512]."""
    Cg = Cl[fi]                                                        # [n,k,576]
    s = torch.bmm(q, Cg.transpose(1, 2)).float() * scale              # [n,H,k]
    s = s.masked_fill(~ok[:, None, :], float("-inf"))
    pr = torch.softmax(s, -1).to(torch.bfloat16)
    return torch.bmm(pr, Cg[..., :512])


# ------------------------------------------------------------------------------------------------ run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["gen", "tf", "force"], default="gen",
                    help="gen: sample; tf: prefill prompt+tf (KV-carry capture); force: decode path over task['tf']")
    ap.add_argument("--n-dec", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--devices", default=",".join(f"cuda:{i}" for i in range(8)))
    ap.add_argument("--stream", choices=["page", "reg", "pread"], default="page")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--resident-layers", type=int, default=-1)
    ap.add_argument("--reserve-gb", type=float, default=10.0)
    ap.add_argument("--time-limit-min", type=float, default=85.0)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--qchunk", type=int, default=256)
    ap.add_argument("--prefill-tokens", type=int, default=65536, help="prompt tokens per device per prefill wave")
    ap.add_argument("--prompt-ent", action="store_true", help="also store next-token entropy at prompt positions")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many decode steps (bench)")
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="ban stop ids for the first N decode tokens (default 0 = natural stop)")
    ap.add_argument("--mtp", action="store_true", help="speculative decoding with the MTP layer (exact: "
                    "speculative sampling against the top-p target distribution)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    devs = [torch.device(d) for d in a.devices.split(",")]
    D = len(devs)
    TK = json.load(open(a.tasks))
    if a.max_tasks:
        TK = TK[: a.max_tasks]
    if a.mode == "force":                   # decode-path teacher forcing: n_dec = len(tf) (equal for all tasks)
        a.n_dec = len(TK[0]["tf"]); assert all(len(t["tf"]) == a.n_dec for t in TK)
    ndec = 0 if a.mode == "tf" else a.n_dec
    seqs = [list(t["prompt"]) + (list(t["tf"]) if a.mode == "tf" else []) for t in TK]
    P = [len(s) for s in seqs]
    cap = [p + ndec + (2 if a.mtp else 0) for p in P]
    assert not (a.mtp and a.mode != "gen")
    # LPT balance of cache tokens over devices
    order = sorted(range(len(TK)), key=lambda i: -cap[i])
    load = [0] * D
    per = [[] for _ in range(D)]
    for i in order:
        g = int(np.argmin(load)); per[g].append(i); load[g] += cap[i]
    log(f"{len(TK)} tasks, prompt tokens {sum(P)} (max {max(P)}), decode {ndec}/task; cache tokens per device {load}")
    M = DModel(a, devs)
    cfg = M.cfg
    sp_layers = [li for li in range(M.nl) if M.sparse[li]]
    nsp = len(sp_layers)
    spi = {li: j for j, li in enumerate(sp_layers)}
    eos = torch.tensor(cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id])
    st = []
    for g, d in enumerate(devs):
        pl = per[g]
        Ps = [P[i] for i in pl]
        caps = [cap[i] for i in pl]
        off = np.concatenate([[0], np.cumsum(caps)]).astype(np.int64)
        N = int(off[-1])
        s = dict(pl=pl, Ps=Ps, off=off, N=N,
                 off_t=torch.tensor(off[:-1], device=d), P_t=torch.tensor(Ps, device=d),
                 C=torch.zeros(M.nlc, N, 576, dtype=torch.bfloat16, device=d),
                 KI=torch.zeros(M.nfull, N, M.ID, dtype=torch.bfloat16, device=d),
                 tok=torch.zeros(N, dtype=torch.long, device=d),
                 rid=torch.zeros(nsp, N, 8, dtype=torch.uint8, device=d),
                 rw=torch.zeros(nsp, N, 8, dtype=torch.float32, device=d),
                 rxn=torch.zeros(nsp, N, dtype=torch.float32, device=d),
                 ent=torch.zeros(N, dtype=torch.float32, device=d),
                 gen=torch.Generator(device=d).manual_seed(a.seed * 1000 + g), step=0,
                 done=torch.zeros(len(pl), dtype=torch.bool, device=d),
                 dlen=torch.full((len(pl),), ndec, dtype=torch.long, device=d))
        if a.mtp:
            s.update(cur=torch.zeros(len(pl), dtype=torch.long, device=d),        # decode index of the next unfed token
                     draft=torch.zeros(len(pl), dtype=torch.long, device=d),
                     q=torch.zeros(len(pl), cfg.vocab_size, dtype=torch.float32, device=d),
                     nacc=torch.zeros((), dtype=torch.long, device=d), nprop=torch.zeros((), dtype=torch.long, device=d))
        for j, i in enumerate(pl):
            s["tok"][off[j]:off[j] + P[i]] = torch.tensor(seqs[i], dtype=torch.long)
            if a.mode == "force":
                s["tok"][off[j] + P[i]:off[j] + P[i] + ndec] = torch.tensor(TK[i]["tf"], dtype=torch.long)
        st.append(s)
    kv_gb = max(s["N"] for s in st) * (M.nlc * 576 * 2 + M.nfull * M.ID * 2 + nsp * 44 + 12) / 2**30
    mem = [torch.cuda.mem_get_info(d) for d in devs]
    per_layer = 32 * 3 * (2048 * 6144) * (8 / D) / 2**30 * 1.02
    free = min(m[0] for m in mem) / 2**30
    nres = a.resident_layers if a.resident_layers >= 0 else max(0, int((free - a.reserve_gb - (a.depth + 1) * per_layer) / per_layer))
    log(f"state {kv_gb:.1f} GB/device (allocated); free GB {[round(m[0] / 2**30, 1) for m in mem]}; resident {nres}")
    ck = f"{a.out}/ckpt"
    CKK = ("C", "KI", "tok", "rid", "rw", "rxn", "ent", "done", "dlen") + (("cur", "draft", "q", "nacc", "nprop") if a.mtp else ())
    assert not (a.mtp and a.min_tokens), "--min-tokens is not supported with --mtp"
    if a.resume and os.path.exists(f"{ck}/meta.json"):
        meta = json.load(open(f"{ck}/meta.json"))
        assert meta["per"] == per, "task assignment changed"
        for g, s in enumerate(st):
            for k in CKK:
                for li in range(s[k].shape[0]) if s[k].dim() == 3 else [None]:
                    fn = f"{ck}/g{g}_{k}{'' if li is None else f'_{li}'}.npy"
                    x = torch.from_numpy(np.load(fn)).to(devs[g])
                    (s[k] if li is None else s[k][li]).copy_(x.view(s[k].dtype) if x.dtype != s[k].dtype else x)
            s["gen"].set_state(torch.load(f"{ck}/g{g}_gen.pt"))
            s["step"] = meta["step"]
        log(f"resumed at decode step {st[0]['step']}")
    M.make_resident(min(nres, len(M.cycle)))
    for li in [x for x in M.cycle if x not in M.resident][: a.depth]:
        M.schedule(li)

    def rec_into(j, fpos):
        def rec(g, i, w, xn):
            s = st[g]
            s["rid"][j, fpos[g]] = i.to(torch.uint8); s["rw"][j, fpos[g]] = w.float(); s["rxn"][j, fpos[g]] = xn
        return rec

    def sample(g, lg, dec_idx):
        s = st[g]
        lp = torch.log_softmax(lg.float(), -1)
        ent = -(lp.exp() * lp).sum(-1)
        if dec_idx < a.min_tokens:          # optional min_tokens: no stop id before it
            lg = lg.clone(); lg[:, eos.to(lg.device)] = float("-inf")
        return sample_top_p(lg.float(), s["gen"], top_p=a.top_p, temp=a.temp)[:, 0], ent

    def mark_stop(g, nt, dec_idx, js=None):
        """nt = decode token dec_idx for local seqs js (all if None). A stop id ends the chain: decode steps
        0..dec_idx-1 are real (their routing is recorded), the stop token itself is never fed back."""
        s = st[g]
        js = torch.arange(len(s["Ps"]), device=nt.device) if js is None else js
        hit = torch.isin(nt, eos.to(nt.device)) & ~s["done"][js]
        s["dlen"][js] = torch.where(hit, torch.full_like(s["dlen"][js], dec_idx), s["dlen"][js])
        s["done"][js] |= hit

    # ---------------------------------------------------------------------------------------- prefill
    def prefill_layer(li, hs, pos, fpos, seg, wv, tks):
        """attention of layer li over the wave (keys of every prompt token first, then query chunks, in place on hs);
        dense MLP in place; returns the MoE inputs (sparse layers) or None.  tks[g] = the full layer's selection."""
        xs = []
        for g, s in enumerate(st):
            b = M.bb[g][li]
            T_ = hs[g].shape[0]
            if T_ == 0:
                xs.append(hs[g]); continue
            for c0 in range(0, T_, 8192):                 # keys of every prompt token first
                hn = rms(hs[g][c0:c0 + 8192], b["input_layernorm.weight"], M.eps)
                _, c, kr, _, cos, sin = M.qkv(g, li, hn, pos[g][c0:c0 + 8192])
                fr = fpos[g][c0:c0 + 8192]
                s["C"][li, fr, :512] = c; s["C"][li, fr, 512:] = kr
                if M.full[li]:
                    s["KI"][M.fidx[li], fr] = M.idx_k(g, li, hn, cos, sin)
                del hn, c, kr
            if M.full[li]:
                tks[g] = (torch.zeros(T_, TOPK, dtype=torch.int32, device=devs[g]),
                          torch.zeros(T_, TOPK, dtype=torch.bool, device=devs[g]))
            for jj, j in enumerate(wv[g]):
                p = s["Ps"][j]; o0 = int(s["off"][j]); r0 = int(seg[g][jj])
                for q0 in range(0, p, a.qchunk):
                    q1 = min(p, q0 + a.qchunk)
                    rr = slice(r0 + q0, r0 + q1)
                    hn = rms(hs[g][rr], b["input_layernorm.weight"], M.eps)
                    q, _, _, qres, cos, sin = M.qkv(g, li, hn, pos[g][rr])
                    L = q1                                # keys 0..q1-1 cover every causal key of the chunk
                    ar = torch.arange(L, device=q.device)
                    valid = ar[None, :] <= torch.arange(q0, q1, device=q.device)[:, None]
                    if L <= TOPK:                          # every causal key selected (full and shared layers)
                        pk, ok = ar[None, :].expand(q1 - q0, L), valid
                        if M.full[li]:
                            tks[g][0][rr, :L] = ar.to(torch.int32)[None]; tks[g][1][rr, :L] = valid
                    elif M.full[li]:
                        qi, wi = M.idx_q(g, li, hn, qres, cos, sin)
                        pk, ok = sel_topk_seq(qi, wi, s["KI"][M.fidx[li], o0:o0 + L], valid)
                        tks[g][0][rr] = pk.to(torch.int32); tks[g][1][rr] = ok
                        del qi, wi
                    else:
                        pk, ok = tks[g][0][rr].long(), tks[g][1][rr]
                    ol = attend(q, s["C"][li], o0 + pk, ok, M.scale)
                    hs[g][rr] += M.attn_out(g, li, ol)         # rows already consumed (keys cached above)
                    del q, qres, hn, ol
            x = rms(hs[g], b["post_attention_layernorm.weight"], M.eps)
            if M.sparse[li]:
                xs.append(x)
            else:
                for c0 in range(0, T_, 16384):
                    hs[g][c0:c0 + 16384] += M.dense(g, li, x[c0:c0 + 16384])
                del x
        return xs if M.sparse[li] else None

    def norec(g, i, w, xn):
        pass

    def mtp_in(g, h, nxt):
        """MTP input: eh_proj([enorm(emb(next token)), hnorm(main final hidden)])."""
        b = M.bb[g][M.mtp]
        e = rms(F.embedding(nxt, M.glob[g]["emb"]).to(torch.bfloat16), b["enorm.weight"], M.eps)
        return M.L(torch.cat([e, rms(h, b["hnorm.weight"], M.eps)], -1), g, M.mtp, "eh_proj")

    def mtp_logits(g, h):
        x = rms(h, M.bb[g][M.mtp]["shared_head.norm.weight"], M.eps)
        return x.float() @ M.glob[g]["head"].T

    def topp(lg):
        """[n,V] logits -> top-p filtered probs [n,V] fp32 (== sample_top_p's distribution)."""
        pr = torch.softmax(lg.float() / a.temp, -1)
        sp, si = torch.sort(pr, -1, descending=True)
        drop = (sp.cumsum(-1) - sp) > a.top_p
        sp = sp.masked_fill(drop, 0)
        sp = sp / sp.sum(-1, keepdim=True)
        return torch.zeros_like(pr).scatter_(1, si, sp)

    def prefill_wave(wv):
        """wv[g] = list of local seq indices on device g for this wave (layer-major over the wave)."""
        t0 = time.time()
        hs, pos, fpos, seg = [], [], [], []
        for g, s in enumerate(st):
            d = devs[g]
            js = wv[g]
            fp = torch.cat([torch.arange(int(s["off"][j]), int(s["off"][j]) + s["Ps"][j]) for j in js]).to(d) if js \
                else torch.zeros(0, dtype=torch.long, device=d)
            fpos.append(fp)
            pos.append(torch.cat([torch.arange(s["Ps"][j]) for j in js]).to(d) if js else fp.clone())
            hs.append(F.embedding(s["tok"][fp], M.glob[g]["emb"]).to(torch.bfloat16))
            seg.append(np.concatenate([[0], np.cumsum([s["Ps"][j] for j in js])]).astype(np.int64))
        tks = [None] * D
        for li in range(M.nl):
            M.clear_dq()
            xs = prefill_layer(li, hs, pos, fpos, seg, wv, tks)
            if M.sparse[li]:
                outs = M.moe_chunked(li, xs, rec_into(spi[li], fpos))
                for g in range(D):
                    hs[g] += outs[g]
                del outs, xs
            if li % 10 == 0 or li == M.nl - 1:
                log(f"prefill layer {li} {time.time() - t0:.0f}s (expert wait {M.t_wait:.0f}s)")
        M.clear_dq()
        for g, s in enumerate(st):
            if not wv[g]:
                continue
            last = torch.tensor(seg[g][1:] - 1, device=devs[g])
            if a.prompt_ent:
                for c0 in range(0, hs[g].shape[0], 4096):
                    lp = torch.log_softmax(M.logits(g, hs[g][c0:c0 + 4096]), -1)
                    s["ent"][fpos[g][c0:c0 + 4096]] = -(lp.exp() * lp).sum(-1)
                    del lp
            if ndec and a.mode == "gen":
                lg = M.logits(g, hs[g][last])
                nt, _ = sample(g, lg, 0)
                js = torch.tensor(wv[g], device=devs[g])
                s["tok"][s["off_t"][js] + s["P_t"][js]] = nt
                mark_stop(g, nt, 0, js)
        if a.mtp and ndec:                          # MTP prefill: position j <- (h_j, x_{j+1}), j = 0..P-1
            hm = [mtp_in(g, hs[g], s["tok"][fpos[g] + 1]) if hs[g].shape[0] else hs[g] for g, s in enumerate(st)]
            del hs
            tkm = [None] * D
            xs = prefill_layer(M.mtp, hm, pos, fpos, seg, wv, tkm)
            outs = M.moe_chunked(M.mtp, xs, norec)
            for g, s in enumerate(st):
                if not wv[g]:
                    continue
                hm[g] += outs[g]
                last = torch.tensor(seg[g][1:] - 1, device=devs[g])
                js = torch.tensor(wv[g], device=devs[g])
                q = topp(mtp_logits(g, hm[g][last]))
                s["q"][js] = q
                s["draft"][js] = torch.multinomial(q, 1, generator=s["gen"])[:, 0]
            del outs, xs, hm
            M.clear_dq()
            log(f"prefill MTP layer {time.time() - t0:.0f}s")
        log(f"prefill wave done {time.time() - t0:.0f}s")

    def prefill():
        waves = []                                 # per device: greedy chunks of <= prefill_tokens prompt tokens
        for g, s in enumerate(st):
            w, cur, n = [], [], 0
            for j, p in enumerate(s["Ps"]):
                if cur and n + p > a.prefill_tokens:
                    w.append(cur); cur, n = [], 0
                cur.append(j); n += p
            if cur:
                w.append(cur)
            waves.append(w)
        nw = max(len(w) for w in waves)
        log(f"prefill: {nw} waves (<= {a.prefill_tokens} prompt tokens per device per wave)")
        for k in range(nw):
            prefill_wave([w[k] if k < len(w) else [] for w in waves])

    # ---------------------------------------------------------------------------------------- decode
    def step(t):
        t0 = time.time(); w0 = M.t_wait
        hs, pos, fpos = [], [], []
        for g, s in enumerate(st):
            p = s["P_t"] + t
            fp = s["off_t"] + p
            pos.append(p); fpos.append(fp)
            hs.append(F.embedding(s["tok"][fp], M.glob[g]["emb"]).to(torch.bfloat16))
        sel = [None] * D
        for li in range(M.nl):
            M.clear_dq()
            xs = []
            for g, s in enumerate(st):
                b = M.bb[g][li]
                n = len(s["Ps"])
                hn = rms(hs[g], b["input_layernorm.weight"], M.eps)
                q, c, kr, qres, cos, sin = M.qkv(g, li, hn, pos[g])
                s["C"][li, fpos[g], :512] = c; s["C"][li, fpos[g], 512:] = kr
                L = int(max(s["Ps"])) + t + 1
                ar = torch.arange(L, device=q.device)
                valid = ar[None, :] <= pos[g][:, None]
                if M.full[li]:
                    s["KI"][M.fidx[li], fpos[g]] = M.idx_k(g, li, hn, cos, sin)
                    if L <= TOPK:
                        sel[g] = (ar[None, :].expand(n, L), valid)
                    else:
                        gidx = (s["off_t"][:, None] + ar[None, :]).clamp_max(s["N"] - 1)
                        qi, wi = M.idx_q(g, li, hn, qres, cos, sin)
                        sel[g] = sel_topk(qi, wi, s["KI"][M.fidx[li]], gidx, valid)
                pk, ok = sel[g]
                fi = (s["off_t"][:, None] + pk).clamp_max(s["N"] - 1)
                ol = attend(q, s["C"][li], fi, ok, M.scale)
                hs[g] = hs[g] + M.attn_out(g, li, ol)
                x = rms(hs[g], b["post_attention_layernorm.weight"], M.eps)
                if M.sparse[li]:
                    xs.append(x)
                else:
                    hs[g] = hs[g] + M.dense(g, li, x)
            if M.sparse[li]:
                outs = M.moe_chunked(li, xs, rec_into(spi[li], fpos))
                for g in range(D):
                    hs[g] = hs[g] + outs[g]
                del outs, xs
        M.clear_dq()
        for g, s in enumerate(st):
            lg = M.logits(g, hs[g])
            if a.mode == "force":
                nt = None
                lp = torch.log_softmax(lg.float(), -1); ent = -(lp.exp() * lp).sum(-1)
            else:
                nt, ent = sample(g, lg, t + 1)
            s["ent"][fpos[g]] = ent
            if t + 1 < ndec and nt is not None:
                live = ~s["done"]                           # never overwrite a finished chain's stop token
                s["tok"][fpos[g][live] + 1] = nt[live]
                mark_stop(g, nt, t + 1)
            s["step"] = t + 1
        nd = sum(int(s["done"].sum()) for s in st)
        log(f"step {t} {time.time() - t0:.1f}s done {nd}/{len(TK)} (expert wait {M.t_wait - w0:.1f}s, "
            f"reg live {M.reg.live_gb() if M.reg else 0:.0f} GB, reg time {M.reg.t_reg if M.reg else 0:.0f}s)")

    def dec_layer(li, hs, pos, fpos, offr, Lh, sel):
        """one decode layer over rows (hs[g] [R,6144]; pos/fpos/offr [R]; Lh[g] = host max(pos)+1): attention (+ dense
        MLP) in place; returns the MoE inputs for sparse layers. sel[g] = the current full layer's selection."""
        xs = []
        for g, s in enumerate(st):
            b = M.bb[g][li]
            hn = rms(hs[g], b["input_layernorm.weight"], M.eps)
            q, c, kr, qres, cos, sin = M.qkv(g, li, hn, pos[g])
            s["C"][li, fpos[g], :512] = c; s["C"][li, fpos[g], 512:] = kr
            L = Lh[g]
            ar = torch.arange(L, device=q.device)
            valid = ar[None, :] <= pos[g][:, None]
            if M.full[li]:
                s["KI"][M.fidx[li], fpos[g]] = M.idx_k(g, li, hn, cos, sin)
                if L <= TOPK:
                    sel[g] = (ar[None, :].expand(len(pos[g]), L), valid)
                else:
                    gidx = (offr[g][:, None] + ar[None, :]).clamp_max(s["N"] - 1)
                    qi, wi = M.idx_q(g, li, hn, qres, cos, sin)
                    sel[g] = sel_topk(qi, wi, s["KI"][M.fidx[li]], gidx, valid)
            pk, ok = sel[g]
            fi = (offr[g][:, None] + pk).clamp_max(s["N"] - 1)
            ol = attend(q, s["C"][li], fi, ok, M.scale)
            hs[g] = hs[g] + M.attn_out(g, li, ol)
            x = rms(hs[g], b["post_attention_layernorm.weight"], M.eps)
            if M.sparse[li]:
                xs.append(x)
            else:
                hs[g] = hs[g] + M.dense(g, li, x)
        return xs

    def step_spec(t):
        """one speculative step per live chain: main model on [x_n, draft] at positions n, n+1 (exact speculative
        sampling vs the top-p target), then the MTP layer on the accepted positions -> next draft."""
        t0 = time.time(); w0 = M.t_wait
        hs, pos, fpos, offr, Lh = [], [], [], [], []
        for g, s in enumerate(st):
            p0 = s["P_t"] + s["cur"]
            ps = torch.stack([p0, p0 + 1], 1).reshape(-1)
            orr = s["off_t"].repeat_interleave(2)
            toks = torch.stack([s["tok"][s["off_t"] + p0], s["draft"]], 1).reshape(-1)
            pos.append(ps); offr.append(orr); fpos.append(orr + ps)
            hs.append(F.embedding(toks, M.glob[g]["emb"]).to(torch.bfloat16))
            Lh.append(int(ps.max()) + 1)
        sel = [None] * D
        for li in range(M.nl):
            M.clear_dq()
            xs = dec_layer(li, hs, pos, fpos, offr, Lh, sel)
            if M.sparse[li]:
                outs = M.moe_chunked(li, xs, rec_into(spi[li], fpos))
                for g in range(D):
                    hs[g] = hs[g] + outs[g]
                del outs, xs
        M.clear_dq()
        xnext, last = [], []
        for g, s in enumerate(st):
            n = len(s["Ps"])
            lg = M.logits(g, hs[g])
            lp = torch.log_softmax(lg.float(), -1)
            s["ent"][fpos[g]] = -(lp.exp() * lp).sum(-1)
            del lp
            pt = topp(lg); del lg
            p0d, p1d = pt[0::2], pt[1::2]
            d, qd = s["draft"], s["q"]
            ratio = p0d.gather(1, d[:, None])[:, 0] / qd.gather(1, d[:, None])[:, 0].clamp_min(1e-30)
            u = torch.rand(n, generator=s["gen"], device=devs[g])
            acc = u < ratio
            res = (p0d - qd).clamp_min(0)
            rs = res.sum(-1, keepdim=True)
            res = torch.where(rs > 0, res / rs.clamp_min(1e-30), p0d)
            x1 = torch.where(acc, d, torch.multinomial(res, 1, generator=s["gen"])[:, 0])
            x2 = torch.multinomial(p1d, 1, generator=s["gen"])[:, 0]
            del pt, p0d, p1d, res
            live = ~s["done"]
            cur = s["cur"]
            p0 = s["P_t"] + cur
            e_ = eos.to(x1.device)
            st1 = torch.isin(x1, e_)
            two = acc & ~st1
            st2 = two & torch.isin(x2, e_)
            w1 = live
            w2 = live & two
            s["tok"][(s["off_t"] + p0 + 1)[w1]] = x1[w1]
            s["tok"][(s["off_t"] + p0 + 2)[w2]] = x2[w2]
            ncur = cur + 1 + two.long()
            dl = torch.where(st1, cur + 1, torch.where(st2, cur + 2, torch.full_like(cur, ndec))).clamp_max(ndec)
            fin = live & (st1 | st2 | (ncur >= ndec))
            s["dlen"] = torch.where(fin, dl, s["dlen"])
            s["done"] = s["done"] | fin
            # finished chains park at cur = dlen: their later (ignored) rows write only non-real positions >= P+dlen
            s["cur"] = torch.where(fin, dl, torch.where(live, ncur, cur))
            s["nprop"] += live.sum(); s["nacc"] += (live & acc).sum()
            xnext.append(torch.stack([x1, x2], 1).reshape(-1))
            last.append(torch.arange(n, device=devs[g]) * 2 + two.long())
            s["step"] = t + 1
        # MTP: position j <- (h_j, x_{j+1}) for the two rows; next draft from the last accepted row
        hm = [mtp_in(g, hs[g], xnext[g]) for g in range(D)]
        del hs
        xs = dec_layer(M.mtp, hm, pos, fpos, offr, Lh, [None] * D)
        outs = M.moe_chunked(M.mtp, xs, norec)
        for g, s in enumerate(st):
            h = (hm[g] + outs[g])[last[g]]
            q = topp(mtp_logits(g, h))
            s["q"] = q
            s["draft"] = torch.multinomial(q, 1, generator=s["gen"])[:, 0]
        del outs, xs, hm
        M.clear_dq()
        nd = sum(int(s["done"].sum()) for s in st)
        na = sum(int(s["nacc"]) for s in st); npp = sum(int(s["nprop"]) for s in st)
        adv = sum(int(s["cur"].sum()) for s in st)
        log(f"spec step {t} {time.time() - t0:.1f}s done {nd}/{len(TK)} accept {na / max(npp, 1):.3f} "
            f"decode tokens {adv} (expert wait {M.t_wait - w0:.1f}s, "
            f"reg live {M.reg.live_gb() if M.reg else 0:.0f} GB, reg time {M.reg.t_reg if M.reg else 0:.0f}s)")

    def save(final):
        os.makedirs(ck, exist_ok=True)
        if final:
            out = {}
            for g, s in enumerate(st):
                for j, i in enumerate(s["pl"]):
                    o0, o1 = int(s["off"][j]), int(s["off"][j + 1])
                    out[i] = (o0, o1, g)
            tasks_out = []
            dl = {}
            for g, s in enumerate(st):
                for j, i in enumerate(s["pl"]):
                    dl[i] = (int(s["dlen"][j]), bool(s["done"][j]))
            for g, s in enumerate(st):
                np.save(f"{a.out}/tok.g{g}.npy", s["tok"].cpu().numpy().astype(np.int32))
                np.save(f"{a.out}/ent.g{g}.npy", s["ent"].cpu().numpy())
                for k in ("rid", "rw", "rxn"):
                    np.save(f"{a.out}/{k}.g{g}.npy", s[k].cpu().numpy())
            for i in range(len(TK)):
                o0, o1, g = out[i]
                tasks_out.append(dict(id=TK[i].get("id", str(i)), g=g, off=o0, end=o1, prompt_len=P[i],
                                      n_dec=dl[i][0], stopped=dl[i][1], n_dec_cap=ndec,
                                      gen_seed=a.seed * 1000 + g))
            json.dump(dict(tasks=tasks_out, sparse_layers=sp_layers, mode=a.mode, n_dec=ndec, seed=a.seed,
                           temp=a.temp, top_p=a.top_p, min_tokens=a.min_tokens, stop_ids=eos.tolist(),
                           note="real decode positions per task: prompt_len .. prompt_len+n_dec-1 (routing/ent); "
                                "tok[prompt_len+n_dec] is the stop token when stopped"),
                      open(f"{a.out}/index.json", "w"))
            log(f"final -> {a.out}")
            return
        for g, s in enumerate(st):
            for k in CKK:
                if s[k].dim() == 3:
                    for li in range(s[k].shape[0]):
                        x = s[k][li].cpu()
                        np.save(f"{ck}/g{g}_{k}_{li}.npy", (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).numpy())
                else:
                    np.save(f"{ck}/g{g}_{k}.npy", s[k].cpu().numpy())
            torch.save(s["gen"].get_state(), f"{ck}/g{g}_gen.pt")
        json.dump(dict(step=st[0]["step"], per=per), open(f"{ck}/meta.json.part", "w"))
        os.replace(f"{ck}/meta.json.part", f"{ck}/meta.json")
        log(f"checkpoint at step {st[0]['step']} -> {ck}")

    if not (a.resume and os.path.exists(f"{ck}/meta.json")):
        prefill()
    t = st[0]["step"]
    t_step = time.time()
    nsteps = ndec                           # step t processes decode token t (token 0 from the prefill logits)
    while t < nsteps:
        if a.mtp:
            step_spec(t)
        else:
            step(t)
        t += 1
        el = (time.time() - G.T0) / 60
        last = (time.time() - t_step) / 60
        t_step = time.time()
        if a.max_steps and t >= a.max_steps:
            break
        if a.mode == "gen" and all(bool(s["done"].all()) for s in st):
            log(f"all chains stopped at step {t}")
            break
        if t < nsteps and el + 1.5 * last + 6.0 > a.time_limit_min:
            save(False)
            log("time limit: exit 3 (resume with --resume)")
            if M.reg:
                M.reg.close_all()
            os._exit(3)
    if a.max_steps and a.mode == "gen":      # bench stop: real decode length = tokens actually fed
        for s in st:
            s["dlen"] = torch.minimum(s["dlen"], s["cur"] if a.mtp else torch.full_like(s["dlen"], t))
    save(True)
    if M.reg:
        M.reg.close_all()
    os._exit(0)


if __name__ == "__main__":
    main()
