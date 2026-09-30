#!/usr/bin/env python3
"""T33 draft capture (PRIVATE outputs only, under $DRAFT_OUT = /tmp/nestquant/33-search/draft/private/cap).
FP8 reference teacher-forced forward (T18 nq_e2e machinery, ref stream only, NQ_SHARD=contig -> same windows/order as
T32 trace), per sparse layer L:
  lg   [T,256] fp16  raw router logits (fp32 linear of post_attention_layernorm(h_mid)) of every position
  rms  [T]     fp32  rms of h_mid (pre-norm MoE block input residual)
  sw   [T,256] fp16  "token swap" probe: router logits of norm(h_mid_t - e_t + e_{t+1})  (next token known at refresh)
  ids  [T,8]   uint8 reference top-8 (parity vs T32 trace)
then the MTP layer 78 (DeepSeek/GLM MTP: eh_proj(cat(enorm(emb x_{t+1}), hnorm(h_t))) -> decoder layer -> shared_head
norm -> lm_head), chained NSTEP times (vLLM-style: step j>1 feeds MTP's own output hidden + emb(draft)).  Chained steps
are batched teacher-forced over anchors (step-j attention keys = step-j entries of earlier anchors: approximation of the
serve's per-request KV).  Per anchor t (position of the last token of a refresh block):
  m{j}  [T,6144] fp16  MTP output residual of step j (the state of position t+j)
  d{j}  [T] int32      draft token for position t+j+1, p{j} its prob;  hn: final-normed main hidden [T,6144] fp16
Diagnostics: main top-1 acc vs x_{t+1}, draft acc vs x_{t+j+1} for both hidden variants (normed/un-normed) at step 1."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
os.environ.pop("NQ_TRACE_DIR", None)
import nq_e2e as E  # noqa: E402
import nq_io  # noqa: E402
import quantisers  # noqa: E402

OUTD = os.environ.get("DRAFT_OUT", "/tmp/nestquant/33-search/draft/private/cap")
NSTEP = int(os.environ.get("DRAFT_NSTEP", "4"))
CORPORA = os.environ.get("DRAFT_CORPORA", "calib-fit,glm52-heldout").split(",")
R, W = E.RANK, E.WORLD
F = torch.nn.functional


def save(name, **kw):
    f = f"{OUTD}/{name}.r{R}of{W}.npz"
    np.savez(f + ".part.npz", **kw)
    os.rename(f + ".part.npz", f)


def main():
    assert E.SHARD == "contig" and not E.TRACE_DIR
    os.makedirs(OUTD, exist_ok=True)
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = E.load_config()
    seqs, groups, shas = E.load_windows(CORPORA, 0)
    N, SEQ = seqs.shape[0], E.SEQ
    T = N * SEQ
    json.dump({"rank": R, "world": W, "seq": SEQ, "windows": E.WIN_IDS, "corpora": CORPORA, "corpus_sha": shas},
              open(f"{OUTD}/windows.r{R}of{W}.json", "w"))
    fp8 = nq_io.FP8Model(E.FP8_DIR)
    bb = E.Backbone(cfg, fp8, dev)
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (GlmMoeDsaRotaryEmbedding, GlmMoeDsaRMSNorm,
                                                                      GlmMoeDsaDecoderLayer)
    rot = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(SEQ, device=dev).view(1, -1)
    cos_sin = rot(torch.empty(1, SEQ, 1, device=dev, dtype=torch.bfloat16), pos)
    AC, MC = 4, 16384
    topk_all = pos.to(torch.int32).view(1, 1, SEQ).expand(AC, SEQ, SEQ)
    ids_dev = seqs.to(dev).long()
    emb_w = fp8.tensor("model.embed_tokens.weight", dev)
    h = F.embedding(ids_dev, emb_w).to(torch.bfloat16)                       # [N, SEQ, H]
    e_cur = h.view(T, -1).clone()
    nxt = torch.cat([ids_dev[:, 1:], ids_dev[:, -1:]], 1)                   # x_{t+1} (last position: dummy)
    e_nxt = F.embedding(nxt, emb_w).to(torch.bfloat16).view(T, -1)
    ref = quantisers.make("ref", "ref"); ref.dev = dev
    stats = [{"fallback": 0, "supplied": 0, "route_agree": {}, "local_rel_l2": {}, "rel_div": {}}]
    t0 = time.time()

    def attn(layer, hh):
        for s0 in range(0, N, AC):
            s1 = min(s0 + AC, N)
            att, _, _ = layer.self_attn(hidden_states=layer.input_layernorm(hh[s0:s1]), position_embeddings=cos_sin,
                                        attention_mask=None, position_ids=pos.expand(s1 - s0, -1),
                                        prev_topk_indices=topk_all[: s1 - s0])
            hh[s0:s1] += att
            del att

    NL = int(os.environ.get("DRAFT_NL", cfg.num_hidden_layers))
    hfile = f"{OUTD}/hfinal_nl{NL}.r{R}of{W}.pt"
    done_layers = os.path.exists(hfile)
    if done_layers:
        h = torch.load(hfile, map_location=dev)
        E.log(f"resumed final hidden from {hfile}")
    for li in range(NL if done_layers else 0, NL):
        layer, sparse = bb.build(li)
        ref.begin_layer(li, dev)
        with torch.no_grad():
            attn(layer, h)
            f = h.view(T, -1)
            if sparse:
                norm, gate = layer.post_attention_layernorm, layer.mlp.gate
                Wg = gate.weight.float()
                lg, sw, rms, ids = [], [], [], []
                for c0 in range(0, T, MC):
                    fc = f[c0:c0 + MC]
                    x = norm(fc)
                    lg.append(F.linear(x.float(), Wg).half().cpu())
                    _, _, i = gate(x)
                    ids.append(i.to(torch.uint8).cpu())
                    rms.append(fc.float().pow(2).mean(-1).sqrt().cpu())
                    xs = norm(fc - e_cur[c0:c0 + MC] + e_nxt[c0:c0 + MC])
                    sw.append(F.linear(xs.float(), Wg).half().cpu())
                    del x, xs
                save(f"L{li}", lg=torch.cat(lg).numpy(), sw=torch.cat(sw).numpy(), rms=torch.cat(rms).numpy(),
                     ids=torch.cat(ids).numpy())
                E.moe_multi(layer, li, [f], [ref], fp8, dev, stats, False, MC, seq=SEQ, groups=groups)
            else:
                for c0 in range(0, T, MC):
                    f[c0:c0 + MC] += layer.mlp(layer.post_attention_layernorm(f[c0:c0 + MC]))
        ref.end_layer(li)
        del layer
        torch.cuda.empty_cache()
        E.log(f"layer {li} done {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G")

    if not done_layers:
        torch.save(h, hfile + ".part"); os.rename(hfile + ".part", hfile)
    # ---------------------------------------------------------------- head + MTP
    def rmsnorm(name):
        n = GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev)
        n.load_state_dict({"weight": fp8.tensor(name, dev)})
        return n
    with torch.no_grad():
        fnorm = rmsnorm("model.norm.weight")
        head = fp8.tensor("lm_head.weight", dev).to(torch.bfloat16)
        hraw = h.view(T, -1)
        hn = torch.cat([fnorm(hraw[c0:c0 + MC]) for c0 in range(0, T, MC)]).to(torch.bfloat16)

        def argmax_head(x):
            d, p = [], []
            for c0 in range(0, T, 4096):
                lp = torch.log_softmax((x[c0:c0 + 4096].to(torch.bfloat16) @ head.T).float(), -1)
                mx = lp.max(-1)
                d.append(mx.indices); p.append(mx.values.exp())
                del lp
            return torch.cat(d), torch.cat(p)
        valid = torch.ones(N, SEQ, dtype=torch.bool, device=dev)
        d0, _ = argmax_head(hn)
        acc0 = float((d0.view(N, SEQ)[:, :-1] == ids_dev[:, 1:]).float().mean())
        E.log(f"main head top1 acc vs x_(t+1) {acc0:.4f}")
        save("head", hn=hn.half().cpu().numpy(), d0=d0.int().cpu().numpy())
        pre = "model.layers.78."
        with torch.device("meta"):
            ml = GlmMoeDsaDecoderLayer(cfg, 77)
        ml.self_attn.indexer = None
        ml.mlp.experts = torch.nn.Module()
        sd = {}
        for n in bb.names:
            if n.startswith(pre):
                k = n[len(pre):]
                if ".experts." in n or "indexer" in k or k.endswith("weight_scale_inv") or \
                        k.split(".")[0] in ("eh_proj", "enorm", "hnorm", "shared_head"):
                    continue
                sd[k] = fp8.tensor(n, dev)
        ml.load_state_dict(sd, strict=True, assign=True)
        ml.eval()
        enorm, hnorm, snorm = rmsnorm(pre + "enorm.weight"), rmsnorm(pre + "hnorm.weight"), \
            rmsnorm(pre + "shared_head.norm.weight")
        eh = fp8.tensor(pre + "eh_proj.weight", dev).to(torch.bfloat16)
        ref.begin_layer(78, dev)

        def mtp(hprev, tok):
            """hprev [T,H] (per anchor), tok [N,SEQ] ids -> MTP output residual [T,H]"""
            e = F.embedding(tok, emb_w).to(torch.bfloat16).view(T, -1)
            z = torch.cat([F.linear(torch.cat([enorm(e[c0:c0 + MC]).to(torch.bfloat16),
                                               hnorm(hprev[c0:c0 + MC]).to(torch.bfloat16)], -1), eh)
                           for c0 in range(0, T, MC)]).view(N, SEQ, -1)
            del e
            attn(ml, z)
            fz = z.view(T, -1)
            E.moe_multi(ml, 78, [fz], [ref], fp8, dev, stats, False, MC, seq=SEQ, groups=groups)
            return fz

        def shift_acc(d, j):
            """draft d at anchor t predicts x_{t+j+1}"""
            dd = d.view(N, SEQ)
            return float((dd[:, :SEQ - j - 1] == ids_dev[:, j + 1:]).float().mean())
        res = {"main_acc": acc0}
        variants = {"normed": hn, "raw": hraw}
        m1 = {}
        for vn, hv in variants.items():
            m = mtp(hv, nxt)
            d, p = argmax_head(torch.cat([snorm(m[c0:c0 + MC]) for c0 in range(0, T, MC)]))
            res[f"step1_{vn}_acc"] = shift_acc(d, 1)
            E.log(f"MTP step1 hidden={vn}: draft acc vs x_(t+2) {res[f'step1_{vn}_acc']:.4f}")
            m1[vn] = (m, d, p)
        best = max(variants, key=lambda v: res[f"step1_{v}_acc"])
        res["variant"] = best
        m, d, p = m1[best]
        del m1
        if os.environ.get("DRAFT_SKIP1") != "1":
            save("mtp1", m=m.half().cpu().numpy(), d=d.int().cpu().numpy(), p=p.float().cpu().numpy())
        CH = os.environ.get("DRAFT_CHAIN", "normed")   # step j>1 hidden input: MTP output (raw) or snorm(output)
        res["chain"] = CH
        for j in range(2, NSTEP + 1):
            hin = m if CH == "raw" else torch.cat([snorm(m[c0:c0 + MC]) for c0 in range(0, T, MC)])
            m = mtp(hin, d.view(N, SEQ))
            d, p = argmax_head(torch.cat([snorm(m[c0:c0 + MC]) for c0 in range(0, T, MC)]))
            res[f"step{j}_acc"] = shift_acc(d, j)
            E.log(f"MTP step{j}: draft acc vs x_(t+{j + 1}) {res[f'step{j}_acc']:.4f}")
            save(f"mtp{j}", m=m.half().cpu().numpy(), d=d.int().cpu().numpy(), p=p.float().cpu().numpy())
        res["wall"] = time.time() - t0
        json.dump(res, open(f"{OUTD}/diag.r{R}of{W}.json", "w"), indent=1)
        E.log(f"done {res}")


if __name__ == "__main__":
    main()
