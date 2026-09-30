#!/usr/bin/env python3
"""CPU unit test: dec.py prefill attention (DSA indexer top-2048 + gathered MLA) vs HF GlmMoeDsaAttention (eager),
real layer weights (full layer LF and the following shared layer), random normed hidden states, T > 2048.
  CUDA_VISIBLE_DEVICES= test_indexer.py [T] [LF]"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec as Dm  # noqa: E402
import gen as G  # noqa: E402
import nq_io  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as HM  # noqa: E402

torch.set_grad_enabled(False)
import faulthandler, time  # noqa: E402
faulthandler.dump_traceback_later(240, repeat=True)
T0 = time.time()


def tp(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
torch.set_num_threads(int(os.environ.get("NT", "16")))
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
LF = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cfg = AutoConfig.from_pretrained(G.FP8_DIR)
cfg._attn_implementation = "eager"
idx = nq_io.SafeIndex(G.FP8_DIR)
raw = G.Raw(idx)
dev = torch.device("cpu")

M = object.__new__(Dm.DModel)
M.cfg = cfg; M.H = cfg.num_attention_heads; M.eps = cfg.rms_norm_eps; M.scale = cfg.qk_head_dim ** -0.5
M.IH, M.ID = cfg.index_n_heads, cfg.index_head_dim
M.idx = idx; M.raw = raw; M.nl = 78
M.rot = [HM.GlmMoeDsaRotaryEmbedding(cfg)]
layers = [LF, LF + 1]
M.bb = [{li: M._load_bb(li, dev) for li in layers}]
M._dq = {}
M.full = [t == "full" for t in cfg.indexer_types]
assert cfg.indexer_types[LF] == "full" and cfg.indexer_types[LF + 1] == "shared"

torch.manual_seed(0)
hn = (torch.randn(T, cfg.hidden_size) * 1.0).to(torch.bfloat16)          # stand-in for input_layernorm output
b0 = M.bb[0][LF]
hn = G.rms(hn, b0["input_layernorm.weight"], M.eps)
pos = torch.arange(T)


def hf_attn(li, prev=None):
    A = HM.GlmMoeDsaAttention(cfg, li).float()          # fp32 reference (CPU bf16 bmm is ~100x slower)
    sd = {}
    for k, v in M.bb[0][li].items():
        if not k.startswith("self_attn.") or k.endswith("weight_scale_inv"):
            continue
        kk = k[len("self_attn."):]
        s = M.bb[0][li].get(k.replace(".weight", ".weight_scale_inv"))
        sd[kk] = (nq_io.fp8_dequant(v, s) if (s is not None and k.endswith(".weight")) else v).float()
    missing = A.load_state_dict(sd, strict=False)
    assert not missing.missing_keys, missing.missing_keys
    h32 = hn.float()[None]
    cos, sin = M.rot[0](h32, pos[None])
    out, _, tk = A(h32, (cos.float(), sin.float()), None, position_ids=pos[None], prev_topk_indices=prev)
    return out[0], tk


def mine(li, tks):
    q, c, kr, qres, cos, sin = M.qkv(0, li, hn, pos)
    C = torch.cat([c, kr], -1)
    out = torch.empty(T, cfg.hidden_size, dtype=torch.bfloat16)
    if M.full[li]:
        KI = M.idx_k(0, li, hn, cos, sin)
        qi, wi = M.idx_q(0, li, hn, qres, cos, sin)
        tks = (torch.zeros(T, Dm.TOPK, dtype=torch.long), torch.zeros(T, Dm.TOPK, dtype=torch.bool))
    for q0 in range(0, T, 256):
        q1 = min(T, q0 + 256)
        L = q1
        ar = torch.arange(L)
        valid = ar[None, :] <= torch.arange(q0, q1)[:, None]
        if L <= Dm.TOPK:
            pk, ok = ar[None].expand(q1 - q0, L), valid
            if cfg.indexer_types[li] == "full":
                tks[0][q0:q1, :L] = ar[None]; tks[1][q0:q1, :L] = valid
        elif cfg.indexer_types[li] == "full":
            pk, ok = Dm.sel_topk_seq(qi[q0:q1], wi[q0:q1], KI[:L], valid)
            tks[0][q0:q1] = pk; tks[1][q0:q1] = ok
        else:
            pk, ok = tks[0][q0:q1], tks[1][q0:q1]
        ol = Dm.attend(q[q0:q1], C, pk, ok, M.scale)
        out[q0:q1] = M.attn_out(0, li, ol)
    return out, tks


def dense(li):
    q, c, kr, *_ = M.qkv(0, li, hn, pos)
    C = torch.cat([c, kr], -1)
    out = torch.empty(T, cfg.hidden_size, dtype=torch.bfloat16)
    for q0 in range(0, T, 256):
        q1 = min(T, q0 + 256)
        ar = torch.arange(q1)
        ok = ar[None, :] <= torch.arange(q0, q1)[:, None]
        out[q0:q1] = M.attn_out(0, li, Dm.attend(q[q0:q1], C, ar[None].expand(q1 - q0, q1), ok, M.scale))
    return out


def rel(a, b, sl):
    a, b = a[sl].float(), b[sl].float()
    return float((a - b).norm() / b.norm())


tp("loaded")
hf0, tk_hf = hf_attn(LF); tp("hf0")
my0, tks = mine(LF, None); tp("my0")
hf1, _ = hf_attn(LF + 1, prev=tk_hf); tp("hf1")
my1, _ = mine(LF + 1, tks); tp("my1")
dn0 = dense(LF); tp("dense")
lo, hi = slice(0, 2048), slice(2048, T)
# selection agreement for positions >= 2048
agree = []
for t in range(2048, T, 7):
    a = set(tk_hf[0, t].tolist()); b = set(tks[0][t][tks[1][t]].tolist())
    agree.append(len(a & b) / len(a))
print(f"T={T} L{LF} full : rel|mine-HF| pos<2048 {rel(my0, hf0, lo):.2e}  pos>=2048 {rel(my0, hf0, hi):.2e}  "
      f"| dense-vs-HF pos>=2048 {rel(dn0, hf0, hi):.2e}  | top-2048 set agreement {np.mean(agree):.5f} (min {np.min(agree):.4f})")
print(f"T={T} L{LF + 1} shared: rel|mine-HF| pos<2048 {rel(my1, hf1, lo):.2e}  pos>=2048 {rel(my1, hf1, hi):.2e}")
