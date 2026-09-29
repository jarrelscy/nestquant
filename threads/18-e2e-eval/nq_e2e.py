#!/usr/bin/env python3
"""NestQuant thread 18: full-model KLD / top-1 / ppl of GLM-5.3 with quantised routed experts,
against the FP8 reference (block-FP8 weights dequantised to bf16, bf16 arithmetic).

Only routed experts differ between reference and candidates; attention, dense MLPs (L0-2), shared
experts, router, norms, embeddings and lm_head are the FP8 reference for every stream.

Protocol (same as glm52/tools/capture53/eval3_kld.py, the 2026-09 campaign):
  teacher-forced, non-overlapping SEQ=2048 windows (== index_topk, so the DSA indexer selects every
  causal token and dense causal SDPA is exact -> indexers skipped), windows sharded win[RANK::WORLD],
  each rank streams all 78 decoder layers once (MTP layer 78 excluded).

Multi-stream: one pass carries the reference stream plus K candidate streams (separate residual
streams, separate routing).  Per layer the backbone is read once; per expert the FP8 reference is read
once and every stream that routed tokens to that expert runs through it (candidates get their own
weights from the quantiser plug-in, see quantisers.py).  Candidates therefore never need their own
copy of the backbone, and the reference is computed bit-identically in the same process.

Metrics per (candidate, corpus): KLD(ref||cand) mean nats (+ per-window stderr, per-token p50/p90/
p99), top-1 agreement with ref argmax, ppl(ref), ppl(cand).  Per layer (inline reference only):
residual-stream relative divergence ||h_c-h_r||/||h_r||, router top-8 set agreement, and (with
--local-err) router-weighted relative routed-expert output L2 on the candidate's own inputs.

Subcommands
  prep  [--default] [NAME=SRC ...]   tokenise corpora into $NQ_OUT/corpora/NAME.npy (int32, flat)
        SRC = text file | npytail:/path/tokens.npy:NTOK (last NTOK tokens of a token array)
  run   --corpora a,b,... --cand NAME=SPEC [...] [--n-layers N] [--max-windows W] [--local-err]
        [--ref inline|cache] [--tag T]            (per rank; RANK/WORLD env)
  merge [--tag T]                                 aggregate ranks -> table + $NQ_OUT/results/T.json
  predecode --cand NAME=SPEC [--layers 3-77] [--out DIR]   decode once -> DIR/NAME (use as dir:DIR/NAME)
Env: NQ_FP8 (FP8 checkpoint dir), NQ_OUT (scratch), NQ_VRAM_GB (per-process cap), NQ_SEQ, RANK, WORLD.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq_io  # noqa: E402
import quantisers  # noqa: E402

FP8_DIR = os.environ.get("NQ_FP8", "/tmp/nestquant/src/glm53-fp8")
OUT = os.environ.get("NQ_OUT", "/tmp/nestquant/18-e2e")
SEQ = int(os.environ.get("NQ_SEQ", "2048"))
RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD", "1"))
EVALSETS = f"{OUT}/evalsets"
NQ_TRAIN = "/home/coder/git/orbit-duet/runs/glm53_training_15m_v2/tokens.npy"


def log(m):
    print(f"[r{RANK} {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------ corpora
DEFAULT_CORPORA = {
    # in-distribution for NestQuant's calibration mixture: tail of the frozen 15M training corpus
    # (seeded doc permutation, packed; fits must only use windows before the tail -- see REPORT.md)
    "nq-tail": f"npytail:{NQ_TRAIN}:262144",
    # earlier campaign's corpora (restored from ~/glm53-evalsets-backup.tar.gz)
    "vllm-docs": f"{EVALSETS}/glm52-heldout-xl.txt",       # code/docs held-out (ID-ish, continuity)
    "wikitext": f"{EVALSETS}/glm52-neutral-wikitext.txt",  # OOD prose: the calib-bias canary
    "github": f"{EVALSETS}/glm52-neutral-github.txt",      # OOD code committed after model cutoff
}


def cmd_prep(a):
    os.makedirs(f"{OUT}/corpora", exist_ok=True)
    items = dict(DEFAULT_CORPORA) if a.default else {}
    for s in a.items:
        k, _, v = s.partition("=")
        items[k] = v
    tok = None
    man = {}
    for name, src in items.items():
        if src.startswith("npywin:"):            # npywin:/path/tokens.npy:lo:hi:n  (n evenly spaced rows of [lo, hi))
            _, path, lo, hi, nn = src.split(":")
            t = np.load(path, mmap_mode="r")
            rows = np.linspace(int(lo), int(hi) - 1, int(nn)).round().astype(int)
            assert len(set(rows.tolist())) == len(rows) and t.shape[1] == SEQ
            ids = np.asarray(t[rows], dtype=np.int32).reshape(-1)
            man_rows = rows.tolist()
        elif src.startswith("npytail:"):
            _, path, n = src.split(":")
            t = np.load(path, mmap_mode="r").reshape(-1)
            ids = np.asarray(t[len(t) - int(n):], dtype=np.int32)
        else:
            if tok is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(a.tokenizer or FP8_DIR)
            ids = np.asarray(tok.encode(open(src, errors="ignore").read(),
                                        add_special_tokens=False), dtype=np.int32)
        np.save(f"{OUT}/corpora/{name}.npy", ids)
        man[name] = {"src": src, "tokens": int(len(ids)), "windows": int(len(ids) // SEQ),
                     "sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
        if src.startswith("npywin:"):
            man[name]["rows"] = man_rows
        log(f"{name}: {len(ids)} tokens, {len(ids)//SEQ} windows of {SEQ}")
    old = {}
    mp = f"{OUT}/corpora/manifest.json"
    if os.path.exists(mp):
        old = json.load(open(mp))
    old.update(man)
    json.dump(old, open(mp, "w"), indent=1)


WIN_IDS = []
OVERRIDES = {}


def load_windows(names, max_windows):
    seqs, groups, shas = [], [], []
    for g, n in enumerate(names):
        ids = np.load(f"{OUT}/corpora/{n}.npy")
        nw = len(ids) // SEQ
        if max_windows:
            nw = min(nw, max_windows)
        w = torch.from_numpy(ids[: nw * SEQ].astype(np.int32)).view(nw, SEQ)[RANK::WORLD]
        seqs.append(w)
        WIN_IDS.append((n, list(range(nw))[RANK::WORLD]))
        groups += [g] * w.shape[0]
        shas.append(hashlib.sha256(ids[: nw * SEQ].tobytes()).hexdigest()[:16])
        log(f"corpus {n}: {nw} windows, rank takes {w.shape[0]}")
    return torch.cat(seqs), groups, shas


# ------------------------------------------------------------------------------------------ model
def load_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(FP8_DIR)
    cfg._attn_implementation = "sdpa"
    assert SEQ <= cfg.index_topk, "SEQ must be <= index_topk for the exact indexer skip"
    return cfg


class Backbone:
    def __init__(self, cfg, fp8, dev):
        self.cfg, self.fp8, self.dev = cfg, fp8, dev
        self.names = sorted(fp8.idx.names())

    def build(self, li):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
        with torch.device("meta"):
            layer = GlmMoeDsaDecoderLayer(self.cfg, li)
        layer.self_attn.indexer = None          # exact at SEQ <= index_topk (all causal keys kept)
        sparse = self.cfg.mlp_layer_types[li] == "sparse"
        if sparse:
            layer.mlp.experts = torch.nn.Module()   # routed experts are streamed per expert
        pre = f"model.layers.{li}."
        sd = {}
        for n in self.names:
            if not n.startswith(pre):
                continue
            k = n[len(pre):]
            if ".experts." in n or "indexer" in k or k.endswith("weight_scale_inv"):
                continue
            sd[k] = self.fp8.tensor(n, self.dev)
        layer.load_state_dict(sd, strict=True, assign=True)
        return layer.eval(), sparse


def ffn(x, W):
    g = torch.nn.functional.linear(x, W["gate_proj"].to(x.dtype))
    u = torch.nn.functional.linear(x, W["up_proj"].to(x.dtype))
    return torch.nn.functional.linear(torch.nn.functional.silu(g) * u, W["down_proj"].to(x.dtype))


def moe_multi(layer, li, flats, qs, fp8, dev, stats, local_err, chunk, keep=None):
    """flats: per-stream [T,H] bf16 residual streams (updated in place: f += MoE(norm(f)));
    qs: per-stream quantiser (Ref for the reference).  Routing and the shared expert run in `chunk`-token
    slabs; routed experts gather + normalise their own tokens, so no [T,H] normed copy is kept and every expert's
    weights are transferred once per layer (for all streams).  Extra memory: one [T,H] output buffer per stream."""
    norm, E = layer.post_attention_layernorm, layer.mlp.gate.num_experts
    T = flats[0].shape[0]
    route, outs = [], []
    for f in flats:
        ids_l, w_l = [], []
        out = torch.empty_like(f)
        if keep is not None:
            kx = torch.empty(f.shape, dtype=f.dtype, pin_memory=True)
        for c0 in range(0, T, chunk):
            x = norm(f[c0:c0 + chunk])
            _, w, i = layer.mlp.gate(x)
            ids_l.append(i)
            w_l.append(w)
            out[c0:c0 + chunk] = layer.mlp.shared_experts(x)
            if keep is not None:
                kx[c0:c0 + chunk] = x.cpu()
            del x
        i, w = torch.cat(ids_l), torch.cat(w_l)
        if keep is not None:                     # hidden-state dump: MoE input (post-norm), router, shared output
            keep["x"].append(kx)
            keep["ids"].append(i.to(torch.uint8).cpu())
            keep["p"].append(w.float().cpu())
            keep["shared"].append(out.cpu())
        ids = i.reshape(-1)
        order = torch.argsort(ids, stable=True)
        tok = torch.arange(T, device=dev).repeat_interleave(i.shape[1])[order]
        offs = [0] + torch.bincount(ids, minlength=E).cumsum(0).tolist()
        route.append((tok, w.reshape(-1)[order], offs, i))
        outs.append(out)
    ref_i = route[0][3] if getattr(qs[0], "is_ref", False) else None
    for s in range(1, len(flats)):
        if ref_i is not None:                    # router top-8 set agreement vs reference stream
            same = (route[s][3].unsqueeze(-1) == ref_i.unsqueeze(-2)).any(-1).float().mean()
            stats[s]["route_agree"][li] = float(same)
    lerr = [[0.0, 0.0] for _ in flats]
    for e in range(E):
        if all(r[2][e] == r[2][e + 1] for r in route):
            continue
        cache = {}

        def ref():
            if "w" not in cache:
                cache["w"] = fp8.expert(li, e, dev)
            return cache["w"]
        for s, (f, q) in enumerate(zip(flats, qs)):
            tok, wts, offs, _ = route[s]
            a, b = offs[e], offs[e + 1]
            if a == b:
                continue
            supplied = False
            if getattr(q, "is_ref", False):
                W = ref()
            elif not getattr(q, "active", lambda _l: True)(li):   # layers= restriction: reference by design
                W = ref()
                stats[s]["ref_by_design"] = stats[s].get("ref_by_design", 0) + 1
            else:
                W = q.expert(li, e, ref)
                if W is None:
                    W = ref()
                    stats[s]["fallback"] += 1
                else:
                    stats[s]["supplied"] += 1
                    supplied = True
            xe = norm(f[tok[a:b]])
            y = ffn(xe, W)
            if local_err and supplied:          # routed-expert output error, supplied experts only
                yr = ffn(xe, ref()).float()
                p = wts[a:b].float()
                lerr[s][0] += float((p * (y.float() - yr).pow(2).sum(-1)).sum())
                lerr[s][1] += float((p * yr.pow(2).sum(-1)).sum())
            outs[s].index_add_(0, tok[a:b], (y * wts[a:b, None]).to(f.dtype))
            del xe, y
        cache.clear()
    for s, f in enumerate(flats):
        if lerr[s][1] > 0:
            stats[s]["local_rel_l2"][li] = math.sqrt(lerr[s][0] / lerr[s][1])
        f += outs[s]
        if keep is not None:                     # hidden-state dump: MoE block output (routed + shared)
            keep["moe_out"].append(outs[s].cpu())
    del outs


# ------------------------------------------------------------------------------------------ run
def cmd_run(a):
    dev = "cuda:0"
    if os.environ.get("NQ_VRAM_GB"):
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["NQ_VRAM_GB"]) / tot))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config()
    nl = a.n_layers or cfg.num_hidden_layers
    names = a.corpora.split(",")
    seqs, groups, shas = load_windows(names, a.max_windows)
    N = seqs.shape[0]
    key = hashlib.sha256(json.dumps([names, shas, nl, SEQ, WORLD, a.max_windows]).encode()
                         ).hexdigest()[:12]
    cdir = f"{OUT}/refcache/{key}"
    cfile = f"{cdir}/hid_r{RANK}.pt"
    use_cache = a.ref == "cache" and os.path.exists(cfile)

    qs = [] if use_cache else [quantisers.make("ref", "ref")]
    for c in a.cand:
        n, _, spec = c.partition("=")
        qs.append(quantisers.make(n, spec))
    for q in qs:
        q.dev = dev
    if a.expert_override:
        tgt = set(a.override_streams.split(",")) if a.override_streams else {q.name for q in qs[1:]}
        ov = dict(x.split("=", 1) for x in a.expert_override)
        ov = {int(k): v for k, v in ov.items()}
        for i, q in enumerate(qs):
            if q.name in tgt:
                qs[i] = quantisers.Override(q, ov)
                qs[i].dev = dev
        OVERRIDES.update({"streams": sorted(tgt), "layers": {L: {"dir": d, "sha256": quantisers.dir_sha(d, L)}
                                                            for L, d in ov.items()}})
        log(f"expert overrides {OVERRIDES}")
    stats = [{"fallback": 0, "supplied": 0, "route_agree": {}, "local_rel_l2": {},
              "rel_div": {}} for _ in qs]
    log(f"{N} windows x {SEQ}; streams {[q.name for q in qs]}; layers {nl}; "
        f"ref={'cache ' + cfile if use_cache else 'inline'}")

    fp8 = nq_io.FP8Model(FP8_DIR)
    bb = Backbone(cfg, fp8, dev)
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaRotaryEmbedding, GlmMoeDsaRMSNorm)
    rot = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(SEQ, device=dev).view(1, -1)
    cos_sin = rot(torch.empty(1, SEQ, 1, device=dev, dtype=torch.bfloat16), pos)
    topk_all = pos.to(torch.int32).view(1, 1, SEQ).expand(a.attn_chunk, SEQ, SEQ)
    ids_dev = seqs.to(dev).long()
    emb = fp8.tensor("model.embed_tokens.weight", dev)
    h0 = torch.nn.functional.embedding(ids_dev, emb).to(torch.bfloat16)
    del emb
    hid = [h0] + [h0.clone() for _ in qs[1:]]
    torch.cuda.empty_cache()

    t_start = time.time()
    tl = {}
    dump = set()
    if a.dump_layers:
        lo, _, hi = a.dump_layers.partition("-")
        dump = set(range(int(lo), int(hi or lo) + 1))
        from safetensors.torch import save_file
        assert getattr(qs[0], "is_ref", False), "dump needs the inline reference stream"
        sname = lambda s: "fp8" if s == 0 else qs[s].name              # noqa: E731
        want = set(a.dump_what.split(","))

        def dsave(li, kind, s, t):
            if kind not in want:
                return
            d = f"{a.dump_dir}/L{li}"
            os.makedirs(d, exist_ok=True)
            nm = f"{kind}_{sname(s)}"
            f = f"{d}/{nm}.r{RANK}of{WORLD}.safetensors"
            t = t.reshape(N * SEQ, -1).contiguous().cpu()
            save_file({nm: t}, f + ".part", metadata={"layer": str(li), "kind": kind, "stream": sname(s),
                                                       "spec": qs[s].spec})
            os.rename(f + ".part", f)
    for li in range(nl):
        t0 = time.time()
        layer, sparse = bb.build(li)
        if li in dump:
            for s_ in range(len(hid)):
                dsave(li, "h_in", s_, hid[s_])              # residual entering layer li
        for q in qs:
            q.begin_layer(li, dev)
        with torch.no_grad():
            for h in hid:                                   # attention, per stream
                for s0 in range(0, N, a.attn_chunk):
                    s1 = min(s0 + a.attn_chunk, N)
                    att, _, _ = layer.self_attn(
                        hidden_states=layer.input_layernorm(h[s0:s1]),
                        position_embeddings=cos_sin, attention_mask=None,
                        position_ids=pos.expand(s1 - s0, -1),
                        prev_topk_indices=topk_all[: s1 - s0])
                    h[s0:s1] += att
                    del att
            flats = [h.view(N * SEQ, -1) for h in hid]
            keep = hmid = None
            if li in dump:
                hmid = [h.cpu() for h in hid]               # residual after attention (MoE block input, pre-norm)
                for s_ in range(len(hid)):
                    dsave(li, "h_mid", s_, hmid[s_])
                keep = {k: [] for k in ("x", "ids", "p", "shared", "moe_out")}
            if not sparse:
                for f in flats:
                    for t0_ in range(0, f.shape[0], a.moe_chunk):
                        t1_ = min(t0_ + a.moe_chunk, f.shape[0])
                        f[t0_:t1_] += layer.mlp(layer.post_attention_layernorm(f[t0_:t1_]))
            else:
                moe_multi(layer, li, flats, qs, fp8, dev, stats, a.local_err, a.moe_chunk, keep)
            if li in dump:
                if keep and keep["x"]:
                    for kind, ts in keep.items():           # x / ids / p / shared / moe_out per stream
                        for s_, t in enumerate(ts):
                            dsave(li, kind, s_, t)
                for s_ in range(len(hid)):
                    dsave(li, "h_out", s_, hid[s_])         # residual leaving layer li
                for s_ in range(1, len(hid)):               # FP8 layer output minus this stream's post-attn residual
                    d = (hid[0].float() - hmid[s_].to(dev).float()).to(torch.bfloat16)
                    dsave(li, "d_ref", s_, d)
                    del d
                del keep, hmid
            if not use_cache:
                r = hid[0].float()
                den = float(r.pow(2).sum())
                for s in range(1, len(hid)):
                    stats[s]["rel_div"][li] = math.sqrt(float((hid[s].float() - r).pow(2).sum()) / den)
                del r
        for q in qs:
            q.end_layer(li)
        del layer
        torch.cuda.empty_cache()
        tl[li] = time.time() - t0
        msg = " ".join(f"{q.name}:div={stats[s]['rel_div'].get(li, 0):.4f}"
                       for s, q in enumerate(qs) if s > 0 or use_cache)
        log(f"layer {li} {tl[li]:.1f}s peakVRAM {torch.cuda.max_memory_allocated()/2**30:.1f}G {msg}")
        if dump and li >= max(dump) and a.dump_stop:
            break
    if dump:
        json.dump({"rank": RANK, "world": WORLD, "seq": SEQ, "tokens": N * SEQ, "windows": WIN_IDS,
                   "corpora": names, "corpus_sha": shas, "streams": {q.name: q.spec for q in qs},
                   "layers": sorted(dump), "what": sorted(want), "overrides": OVERRIDES,
                   "dtype": "bfloat16 (ids uint8 [T,8], p float32 [T,8] incl. routed_scaling_factor)",
                   "layout": "[windows*SEQ, ...], rank-local windows in order (manifest windows)",
                   "definitions": {"h_in": "residual entering the layer", "h_mid": "residual after attention",
                                   "x": "post_attention_layernorm(h_mid) = MoE input", "ids/p": "router top-8 / weights",
                                   "shared": "shared-expert output on x", "moe_out": "shared + routed output",
                                   "h_out": "residual leaving the layer (= h_mid + moe_out, bf16 add)",
                                   "d_ref_S": "fp32(h_out_fp8) - fp32(h_mid_S), stored bf16"},
                   "stats": [{k: v for k, v in st.items() if k in ("fallback", "supplied", "ref_by_design")}
                             for st in stats], "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())},
                  open(f"{a.dump_dir}/manifest.r{RANK}of{WORLD}.json", "w"), indent=1)
        if a.dump_stop:
            log(f"dump done -> {a.dump_dir}")
            return

    # ---- reference cache
    if use_cache:
        href = torch.load(cfile, map_location=dev)
        cand = list(range(len(qs)))
    else:
        href = hid[0]
        cand = list(range(1, len(qs)))
        os.makedirs(cdir, exist_ok=True)
        if os.path.exists(cfile):
            old = torch.load(cfile, map_location=dev)
            eq = bool(torch.equal(old, href))
            log(f"inline reference vs cached reference bitwise equal: {eq}")
            stats[0]["ref_cache_bitwise_equal"] = eq
        elif a.save_ref:
            torch.save(href, cfile)
            log(f"saved reference final hidden -> {cfile}")

    # ---- logits + metrics
    norm = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev)
    norm.load_state_dict({"weight": fp8.tensor("model.norm.weight", dev)})
    head = fp8.tensor("lm_head.weight", dev).float()      # [V, H] fp32
    ng = len(names)
    res = {q.name: {"groups": [{"ce_ref": 0.0, "ce": 0.0, "ntok": 0, "klsum": 0.0,
                                "agree": 0, "win_kl": []} for _ in range(ng)]}
           for i, q in enumerate(qs) if i in cand}
    tokkl = {qs[i].name: [] for i in cand}
    P = a.pos_chunk
    with torch.no_grad():
        for w in range(N):
            g = groups[w]
            hr = norm(href[w])[:-1]
            tgt = ids_dev[w, 1:]
            hc = [norm(hid[i][w])[:-1] for i in cand]
            wk = [0.0] * len(cand)
            for p0 in range(0, SEQ - 1, P):
                p1 = min(p0 + P, SEQ - 1)
                lr = torch.log_softmax(hr[p0:p1].float() @ head.T, -1)
                ce_r = float(-lr.gather(1, tgt[p0:p1, None]).sum())
                am_r = lr.argmax(-1)
                for j, i in enumerate(cand):
                    lc = torch.log_softmax(hc[j][p0:p1].float() @ head.T, -1)
                    kl = (lr.exp() * (lr - lc)).sum(-1)
                    G = res[qs[i].name]["groups"][g]
                    G["ce_ref"] += ce_r
                    G["ce"] += float(-lc.gather(1, tgt[p0:p1, None]).sum())
                    G["klsum"] += float(kl.sum())
                    G["agree"] += int((lc.argmax(-1) == am_r).sum())
                    G["ntok"] += p1 - p0
                    wk[j] += float(kl.sum())
                    tokkl[qs[i].name].append(kl.half().cpu().numpy())
                    del lc, kl
                del lr
            for j, i in enumerate(cand):
                res[qs[i].name]["groups"][g]["win_kl"].append(wk[j] / (SEQ - 1))
    rdir = f"{OUT}/results/{a.tag}"
    os.makedirs(rdir, exist_ok=True)
    for j, i in enumerate(cand):
        q = qs[i]
        res[q.name].update({"spec": q.spec, "stats": stats[i], "tokkl_groups": None,
                            "extra": {k: getattr(q, k) for k in ("n_hi", "n_lo") if hasattr(q, k)}})
        np.save(f"{rdir}/tokkl_{q.name}_r{RANK}.npy", np.concatenate(tokkl[q.name]))
    json.dump({"rank": RANK, "world": WORLD, "corpora": names, "corpus_sha": shas, "seq": SEQ,
               "n_layers": nl, "groups_per_window": groups, "ref": "cache" if use_cache else "inline",
               "ref_stats": stats[0] if not use_cache else None,
               "layer_seconds": tl, "wall_seconds": time.time() - t_start, "results": res},
              open(f"{rdir}/r{RANK}.json", "w"))
    for name, r in res.items():
        for g, G in enumerate(r["groups"]):
            if G["ntok"]:
                log(f"{name:12s} {names[g]:10s} ppl_ref {math.exp(G['ce_ref']/G['ntok']):8.4f} "
                    f"ppl {math.exp(G['ce']/G['ntok']):8.4f} KLD {G['klsum']/G['ntok']:.5f} "
                    f"top1 {100*G['agree']/G['ntok']:.2f}%  (fallback experts {r['stats']['fallback']})")


def cmd_predecode(a):
    """Decode a candidate once into a dir: layout (GPU-parallel via RANK/WORLD), so the eval pass only
    reads bf16/fp16 weights.  Experts are sharded (L*E + e) % WORLD == RANK; one file per (layer, rank),
    written .part then renamed (the dir: plug-in only globs finished *.safetensors).  fp32 decodes are
    stored as fp16 when that is lossless (NestQuant's kernel decode is fp16), else bf16."""
    from safetensors.torch import save_file
    dev = "cuda:0"
    if os.environ.get("NQ_VRAM_GB"):
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["NQ_VRAM_GB"]) / tot))
    cfg = load_config()
    n, _, spec = a.cand.partition("=")
    q = quantisers.make(n, spec)
    q.dev = dev
    fp8 = nq_io.FP8Model(FP8_DIR)
    lo, _, hi = a.layers.partition("-")
    layers = [li for li in range(int(lo), int(hi or lo) + 1) if li >= cfg.first_k_dense_replace]
    E = cfg.n_routed_experts
    out = f"{a.out}/{n}"
    os.makedirs(out, exist_ok=True)
    lossy = 0
    for li in layers:
        f = f"{out}/layer_{li:03d}.r{RANK}of{WORLD}.safetensors"
        if os.path.exists(f):
            continue
        t0 = time.time()
        q.begin_layer(li, dev)
        tens, miss = {}, 0
        for e in range(E):
            if (li * E + e) % WORLD != RANK:
                continue
            cache = {}

            def ref():
                if "w" not in cache:
                    cache["w"] = fp8.expert(li, e, dev)
                return cache["w"]
            W = q.expert(li, e, ref)
            if W is None:
                miss += 1
                continue
            for pn, w in W.items():
                if w.dtype == torch.float32:
                    h = w.half()
                    if torch.equal(h.float(), w):
                        w = h
                    else:
                        w, lossy = w.bfloat16(), lossy + 1
                tens[f"model.layers.{li}.mlp.experts.{e}.{pn}.weight"] = w.contiguous().cpu()
        q.end_layer(li)
        if tens:
            save_file(tens, f + ".part")
            os.rename(f + ".part", f)
        log(f"L{li}: {len(tens) // 3} experts written, {miss} missing, {time.time() - t0:.1f}s")
        del tens
    if lossy:
        log(f"WARNING: {lossy} fp32 tensors were not fp16-exact -> stored bf16")


def st_same(f, g, blk=1 << 26):
    """safetensors files equal: identical data section bytes (everything after the header) and equal header JSON.
    The header is compared parsed because safetensors serialises the __metadata__ map in hash order (differs between
    processes for the same dict), so raw header bytes are not reproducible even for the same writer."""
    import struct
    with open(f, "rb") as x, open(g, "rb") as y:
        nx, ny = struct.unpack("<Q", x.read(8))[0], struct.unpack("<Q", y.read(8))[0]
        if nx != ny or json.loads(x.read(nx)) != json.loads(y.read(ny)):
            return False
        if os.fstat(x.fileno()).st_size != os.fstat(y.fileno()).st_size:
            return False
        while True:
            bx, by = x.read(blk), y.read(blk)
            if bx != by:
                return False
            if not bx:
                return True


def cmd_predecode_nq(a):
    """NestQuant-specific predecode: one rotated_levels() per expert serves both levels.  Writes
    {out}/nq2/layer_LLL.rRofW.safetensors (every expert, if 2 in --levels) and {out}/nq4/... (experts in --l4-set,
    or all).  fp16 always (the fp32 decode incl. the low-rank term, rounded once; |w| << 65504 is asserted)."""
    from safetensors.torch import save_file
    dev = os.environ.get("NQ_DEV", "cuda:0")
    if os.environ.get("NQ_VRAM_GB") and dev != "cpu":
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["NQ_VRAM_GB"]) / tot))
    sys.path.insert(0, quantisers.NQ12)
    import nq_decode as D
    cfg = load_config()
    E = cfg.n_routed_experts
    levels = [int(x) for x in a.levels.split(",")]
    l4 = quantisers.load_level4_set(a.l4_set) if a.l4_set != "all" else None
    lo, _, hi = a.layers.partition("-")
    layers = [li for li in range(int(lo), int(hi or lo) + 1) if li >= cfg.first_k_dense_replace]
    for lv in levels:
        os.makedirs(f"{a.out}/nq{lv}", exist_ok=True)
    nexp = {lv: 0 for lv in levels}
    writer = None
    verify = {lv: [0, 0] for lv in levels}
    werr = []
    if a.fast:
        import nq_fastdec as F
    t_all = time.time()
    for li in layers:
        want = {}
        for lv in levels:
            f = f"{a.out}/nq{lv}/layer_{li:03d}.r{RANK}of{WORLD}.safetensors"
            ex = [e for e in range(E) if (li * E + e) % WORLD == RANK
                  and (lv != 4 or l4 is None or e in l4.get(li, ()))]
            if ex and not os.path.exists(f):
                want[lv] = (f, set(ex))
        if not want:
            continue
        t0 = time.time()
        todo = sorted(set().union(*[v[1] for v in want.values()]))
        tens = {lv: {} for lv in want}
        if a.fast:                                   # batched GPU decoder from the TP shard files (nq_fastdec)
            if writer is not None:
                writer.join()                        # at most one layer of fp16 tensors in RAM besides the current one
            if werr:
                raise werr[0]
            ls = F.LayerShards(a.root, li)
            for i in range(0, len(todo), a.batch):
                chunk = todo[i:i + a.batch]
                lvs = tuple(lv for lv in want if any(e in want[lv][1] for e in chunk))
                out = F.decode_experts([ls.art(e) for e in chunk], lvs, dev, batch_had=True)
                for lv in lvs:
                    for k, e in enumerate(chunk):
                        if e not in want[lv][1]:
                            continue
                        for pn, w in zip(quantisers.PROJ, out[lv][k]):
                            assert torch.isfinite(w).all() and float(w.abs().max()) < 6e4, (li, e, pn)
                            tens[lv][f"model.layers.{li}.mlp.experts.{e}.{pn}.weight"] = w.half().contiguous().cpu()
                        nexp[lv] += 1
                del out
            t_dec = time.time() - t0

            def write(tens=tens, want=want, li=li, t_dec=t_dec, t0=t0, n=len(todo)):
                t_w = time.time()
                for lv, (f, ex) in want.items():
                    save_file(tens[lv], f + ".part", metadata={"root": a.root, "level": str(lv)})
                    os.rename(f + ".part", f)
                t_w = time.time() - t_w
                ver = ""
                if a.verify_against:                 # byte-for-byte vs an earlier predecode, then drop our copy
                    for lv, (f, ex) in want.items():
                        ref = f"{a.verify_against}/nq{lv}/{os.path.basename(f)}"
                        same = os.path.exists(ref) and st_same(f, ref)
                        verify[lv][0 if same else 1] += 1
                        ver += f" nq{lv}:{'IDENTICAL' if same else 'DIFFERENT'}"
                        if same:
                            os.remove(f)
                log(f"L{li}: decoded {n} experts in {t_dec:.1f}s, wrote " +
                    " ".join(f"nq{lv}:{len(tens[lv]) // 3}" for lv in want) + f" in {t_w:.1f}s{ver}")
            import threading

            def guarded(write=write):
                try:
                    write()
                except BaseException as ex:          # surfaced by the main thread after join
                    werr.append(ex)
            writer = threading.Thread(target=guarded)
            writer.start()
            del tens
            continue
        for e in todo:
            f = quantisers.nq_artifact_path(a.root, li, e)
            if f is None:
                raise SystemExit(f"missing artifact L{li} E{e} under {a.root}")
            art = torch.load(f, map_location="cpu", weights_only=False)
            rot = {p: D.rotated_levels(art[p], dev) for p in ("gate", "up", "down")}
            perm = art.get("meta", {}).get("inter_perm")
            for lv in want:
                if e not in want[lv][1]:
                    continue
                W = [D.decode_matrix(art[p], lv, dev, rot=rot[p]) for p in ("gate", "up", "down")]
                if perm is not None:
                    inv = torch.argsort(torch.as_tensor(perm, device=dev))
                    W = [W[0][inv], W[1][inv], W[2][:, inv]]
                for pn, w in zip(quantisers.PROJ, W):
                    assert torch.isfinite(w).all() and float(w.abs().max()) < 6e4, (li, e, pn)
                    tens[lv][f"model.layers.{li}.mlp.experts.{e}.{pn}.weight"] = w.half().contiguous().cpu()
                nexp[lv] += 1
            del rot, art
        for lv, (f, ex) in want.items():
            save_file(tens[lv], f + ".part", metadata={"root": a.root, "level": str(lv)})
            os.rename(f + ".part", f)
        log(f"L{li}: decoded {len(todo)} experts, wrote " +
            " ".join(f"nq{lv}:{len(tens[lv]) // 3}" for lv in want) + f" in {time.time() - t0:.1f}s")
        del tens
    if writer is not None:
        writer.join()
    if werr:
        raise werr[0]
    log(f"done: experts written {nexp} in {time.time() - t_all:.1f}s" +
        (f"; byte-compare [identical, different] per level {verify}" if a.verify_against else ""))


def cmd_merge(a):
    rdir = f"{OUT}/results/{a.tag}"
    parts = [json.load(open(p)) for p in sorted(glob.glob(f"{rdir}/r*.json"))]
    names = parts[0]["corpora"]
    out = {"tag": a.tag, "ranks": len(parts), "corpora": names, "table": {}}
    print(f"{a.tag}: {len(parts)} ranks, corpora {names}, n_layers {parts[0]['n_layers']}")
    print(f"{'cand':14} {'corpus':10} {'ntok':>8} {'ppl_ref':>8} {'ppl':>8} {'KLD':>9} {'±se':>8} "
          f"{'p99':>7} {'top1%':>7} {'fallbk':>7}")
    for cand in parts[0]["results"]:
        # token kl in rank order, groups known from per-window groups
        for g, n in enumerate(names):
            A = {"ce_ref": 0.0, "ce": 0.0, "ntok": 0, "klsum": 0.0, "agree": 0, "win_kl": []}
            fb = 0
            tk = []
            for p in parts:
                r = p["results"][cand]
                G = r["groups"][g]
                for k in ("ce_ref", "ce", "ntok", "klsum", "agree"):
                    A[k] += G[k]
                A["win_kl"] += G["win_kl"]
                fb += r["stats"]["fallback"]
                kl = np.load(f"{rdir}/tokkl_{cand}_r{p['rank']}.npy").astype(np.float32)
                gw = np.repeat(np.array(p["groups_per_window"]), p["seq"] - 1)
                tk.append(kl[gw == g])
            if not A["ntok"]:
                continue
            tk = np.concatenate(tk)
            wk = np.array(A["win_kl"])
            row = {"ntok": A["ntok"], "ppl_ref": math.exp(A["ce_ref"] / A["ntok"]),
                   "ppl": math.exp(A["ce"] / A["ntok"]), "kld": A["klsum"] / A["ntok"],
                   "kld_se": float(wk.std(ddof=1) / math.sqrt(len(wk))) if len(wk) > 1 else float("nan"),
                   "kld_p50": float(np.percentile(tk, 50)), "kld_p90": float(np.percentile(tk, 90)),
                   "kld_p99": float(np.percentile(tk, 99)), "top1": 100 * A["agree"] / A["ntok"],
                   "fallback_expert_calls": fb}
            out["table"].setdefault(cand, {})[n] = row
            print(f"{cand:14} {n:10} {row['ntok']:8d} {row['ppl_ref']:8.4f} {row['ppl']:8.4f} "
                  f"{row['kld']:9.5f} {row['kld_se']:8.5f} {row['kld_p99']:7.3f} {row['top1']:7.3f} {fb:7d}")
        # per-layer stats, averaged over ranks (each rank holds ~1/WORLD of every corpus)
        per = {}
        for key in ("rel_div", "route_agree", "local_rel_l2"):
            acc = {}
            for p in parts:
                for l, v in p["results"][cand]["stats"][key].items():
                    acc.setdefault(int(l), []).append(v)
            per[key] = {l: float(np.mean(v)) for l, v in sorted(acc.items())}
        if per["rel_div"] or per["route_agree"]:
            rd = per["rel_div"]
            inc = {l: rd[l] - rd.get(l - 1, 0.0) for l in rd}
            bands = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}
            bs = {}
            for b, rg in bands.items():
                ra = [per["route_agree"][l] for l in rg if l in per["route_agree"]]
                le = [per["local_rel_l2"][l] for l in rg if l in per["local_rel_l2"]]
                bs[b] = {"route_top8_agree": float(np.mean(ra)) if ra else None,
                         "local_rel_l2_mean": float(np.mean(le)) if le else None,
                         "rel_div_increment_sum": float(sum(inc[l] for l in rg if l in inc))}
            top = sorted(inc, key=lambda l: -inc[l])[:10]
            out.setdefault("per_layer", {})[cand] = dict(per, rel_div_increment=inc, bands=bs,
                                                         top_increment_layers=top,
                                                         extra=[p["results"][cand].get("extra") for p in parts])
            print(f"   {cand} bands: " + " | ".join(
                f"{b} route {v['route_top8_agree'] if v['route_top8_agree'] is None else round(100 * v['route_top8_agree'], 2)}% "
                f"lerr {v['local_rel_l2_mean'] if v['local_rel_l2_mean'] is None else round(v['local_rel_l2_mean'], 4)} "
                f"dDiv {v['rel_div_increment_sum']:.3f}" for b, v in bs.items()))
            print(f"   {cand} largest divergence increments: " + " ".join(f"L{l}:+{inc[l]:.3f}" for l in top))
    json.dump(out, open(f"{OUT}/results/{a.tag}.json", "w"), indent=1)
    print(f"-> {OUT}/results/{a.tag}.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--default", action="store_true")
    p.add_argument("--tokenizer")
    p.add_argument("items", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("--corpora", required=True)
    r.add_argument("--cand", action="append", default=[])
    r.add_argument("--n-layers", type=int, default=0)
    r.add_argument("--max-windows", type=int, default=0, help="cap per corpus (before sharding)")
    r.add_argument("--ref", choices=["inline", "cache"], default="inline")
    r.add_argument("--save-ref", action="store_true")
    r.add_argument("--local-err", action="store_true")
    r.add_argument("--attn-chunk", type=int, default=4)
    r.add_argument("--moe-chunk", type=int, default=16384, help="tokens per router/shared-expert slab")
    r.add_argument("--pos-chunk", type=int, default=512)
    r.add_argument("--tag", default="run")
    r.add_argument("--dump-layers", help="a-b: save h_in / h_mid / moe_out / h_out of every stream at these layers")
    r.add_argument("--dump-dir", default=f"{OUT}/hdump")
    r.add_argument("--dump-stop", action="store_true", help="stop after the last dump layer (no metrics)")
    r.add_argument("--dump-what", default="x,ids,p,shared,moe_out,d_ref,h_in,h_out",
                   help="subset of h_in,h_mid,x,ids,p,shared,moe_out,h_out,d_ref (files L{L}/{kind}_{stream}.rRofW)")
    r.add_argument("--expert-override", action="append", default=[],
                   help="L=DIR: layer L's routed experts decoded (nq_fastdec) from DIR's E{E}.pt artifacts, at the "
                        "level the stream's mix picks; applies to --override-streams")
    r.add_argument("--override-streams", default="", help="comma list of candidate names (default: all candidates)")
    m = sub.add_parser("merge")
    m.add_argument("--tag", default="run")
    d = sub.add_parser("predecode")
    d.add_argument("--cand", required=True, help="NAME=SPEC")
    d.add_argument("--layers", default="3-77")
    d.add_argument("--out", default=f"{OUT}/predecoded")
    n = sub.add_parser("predecode-nq")
    n.add_argument("--root", default="/tmp/nestquant/nq-encode-v1")
    n.add_argument("--levels", default="2,4")
    n.add_argument("--l4-set", default="all", help="nq_defset.py JSON (level-4 subset) or 'all'")
    n.add_argument("--layers", default="3-77")
    n.add_argument("--out", default=f"{OUT}/predecoded")
    n.add_argument("--fast", action="store_true", help="batched GPU decoder (nq_fastdec) from the tp*.safetensors")
    n.add_argument("--batch", type=int, default=4, help="--fast: experts per decode call")
    n.add_argument("--verify-against", help="--fast: compare each written file byte-for-byte with DIR/nq{lv}/same name, "
                                            "delete ours if identical (keeps different ones for inspection)")
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "merge": cmd_merge, "predecode": cmd_predecode,
     "predecode-nq": cmd_predecode_nq}[a.cmd](a)


if __name__ == "__main__":
    main()
