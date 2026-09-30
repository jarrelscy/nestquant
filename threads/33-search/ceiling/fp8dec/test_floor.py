#!/usr/bin/env python3
"""T33l: why is dec.py-vs-T32-trace top-8 overlap ~90%?  CPU, real tokens (glm52-heldout window W, first T tokens), layers
0..3: HF GlmMoeDsa decoder layers in fp32 (truth) and bf16 (= nq_e2e/T32 trace numerics: dequantised-FP8 weights, bf16
arithmetic, HF non-absorbed MLA) vs dec.py's bf16 path (absorbed MLA, Wd dequant) -> L3 router (fp32, HF TopkRouter) on
each variant's post_attention_layernorm output.  Reports rel err of the MoE input vs fp32 and pairwise top-8 overlap,
+ HF-bf16 / mine vs the T32 trace L3 ids at the same positions.
  CUDA_VISIBLE_DEVICES= test_floor.py [T=512] [W=3]"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/ceiling")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import dec as Dm  # noqa: E402
import gen as G  # noqa: E402
import nq_io  # noqa: E402
import t32lib as T32  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as HM  # noqa: E402
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("NT", "16")))
T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LS = 3
cfg = AutoConfig.from_pretrained(G.FP8_DIR); cfg._attn_implementation = "eager"
idx = nq_io.SafeIndex(G.FP8_DIR); raw = G.Raw(idx)
M = object.__new__(Dm.DModel)
M.cfg = cfg; M.H = cfg.num_attention_heads; M.eps = cfg.rms_norm_eps; M.scale = cfg.qk_head_dim ** -0.5
M.IH, M.ID = cfg.index_n_heads, cfg.index_head_dim
M.idx = idx; M.raw = raw; M.nl = 78; M._dq = {}; M._gates = {}
M.rot = [HM.GlmMoeDsaRotaryEmbedding(cfg)]
M.bb = [{li: M._load_bb(li, torch.device("cpu")) for li in range(LS + 1)}]
M.full = [t == "full" for t in cfg.indexer_types]
tok = np.load("/tmp/nestquant/18-e2e/corpora/glm52-heldout.npy")[W * 2048: W * 2048 + T].astype(np.int64)
emb = raw.read_many(["model.embed_tokens.weight"], torch.device("cpu"))["model.embed_tokens.weight"]
h0 = torch.nn.functional.embedding(torch.from_numpy(tok), emb).to(torch.bfloat16)
pos = torch.arange(T)


def sd_of(li, pre):
    out = {}
    b = M.bb[0][li]
    for k, v in b.items():
        if not k.startswith(pre) or k.endswith("weight_scale_inv"):
            continue
        s = b.get(k.replace(".weight", ".weight_scale_inv"))
        out[k[len(pre):]] = nq_io.fp8_dequant(v, s) if (s is not None and k.endswith(".weight")) else v
    return out


def hf_run(dt):
    h = h0.to(dt)[None]
    cos, sin = M.rot[0](h.float(), pos[None])
    pe = (cos.to(dt), sin.to(dt))
    tk = None
    for li in range(LS):
        L = HM.GlmMoeDsaDecoderLayer(cfg, li)
        L.load_state_dict({k: v.to(dt) for k, v in sd_of(li, "").items()}, strict=True)
        L = L.to(dt)
        h, tk_ = L(h, None, position_ids=pos[None], position_embeddings=pe, prev_topk_indices=tk)
        tk = tk_ if tk_ is not None else tk
    A = HM.GlmMoeDsaAttention(cfg, LS)
    sd = {k: v.to(dt) for k, v in sd_of(LS, "self_attn.").items()}
    assert not A.load_state_dict(sd, strict=False).missing_keys
    A = A.to(dt)
    n1 = HM.GlmMoeDsaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); n1.weight.data = M.bb[0][LS]["input_layernorm.weight"].to(dt)
    n2 = HM.GlmMoeDsaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); n2.weight.data = M.bb[0][LS]["post_attention_layernorm.weight"].to(dt)
    o, _, _ = A(n1(h), pe, None, position_ids=pos[None], prev_topk_indices=tk)
    h = h + o
    return n2(h)[0]


def mine():
    h = h0.clone()
    for li in range(LS + 1):
        b = M.bb[0][li]
        hn = G.rms(h, b["input_layernorm.weight"], M.eps)
        q, c, kr, *_ = M.qkv(0, li, hn, pos)
        C = torch.cat([c, kr], -1)
        out = torch.empty(T, cfg.hidden_size, dtype=torch.bfloat16)
        for q0 in range(0, T, 256):
            q1 = min(T, q0 + 256); ar = torch.arange(q1)
            ok = ar[None, :] <= torch.arange(q0, q1)[:, None]
            out[q0:q1] = M.attn_out(0, li, Dm.attend(q[q0:q1], C, ar[None].expand(q1 - q0, q1), ok, M.scale))
        h = h + out
        x = G.rms(h, b["post_attention_layernorm.weight"], M.eps)
        if li < LS:
            h = h + M.dense(0, li, x)
        M.clear_dq()
    return x


x32 = hf_run(torch.float32); print("hf fp32 done", flush=True)
x16 = hf_run(torch.bfloat16); print("hf bf16 done", flush=True)
xm = mine(); print("mine done", flush=True)
gate = G.Model.gate(M, 0, LS)
ids = {k: gate(v)[2].sort(-1)[0] for k, v in (("fp32", x32), ("hf_bf16", x16), ("mine", xm))}
tr = T32.load_layer(LS, "glm52-heldout")[0][W * 2048: W * 2048 + T].astype(np.int64)
ids["trace"] = torch.from_numpy(np.sort(tr, 1))


def ov(a, b):
    return float(np.mean([len(set(a[i].tolist()) & set(b[i].tolist())) / 8 for i in range(T)]))


def rel(a):
    return float((a.float() - x32.float()).norm() / x32.float().norm())
print(f"T={T} W={W} L{LS} MoE-input rel err vs fp32: hf_bf16 {rel(x16):.2e}  mine {rel(xm):.2e}  mine-vs-hf_bf16 "
      f"{float((xm.float() - x16.float()).norm() / x16.float().norm()):.2e}")
ks = list(ids)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"L{LS} top-8 overlap {ks[i]:8s} vs {ks[j]:8s} {100 * ov(ids[ks[i]], ids[ks[j]]):6.2f}%")
