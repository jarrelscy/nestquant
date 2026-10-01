"""T34: emulation of vLLM's fp8_ds_mla KV cache (SM120 serve format, 656 B/token/layer) for the nq_e2e harness.

Source (jarrelscy SM120 fork, /home/coder/git/vllm-mimo-v26-arvq-sm120 @88c942331):
  csrc/libtorch_stable/cache_kernels.cu concat_and_cache_ds_mla_kernel (write), per token:
    bytes [0,512)   e4m3 of kv_c / tile_scale, 4 tiles of 128  (__nv_cvt_float_to_fp8(x/scale, SATFINITE, E4M3), RNE)
    bytes [512,528) 4 fp32 tile scales: s = max(max|x_tile| / 448, FLT_MIN), then rounded UP to the smallest power
                    of two >= s (ilogbf/ldexpf; PR #10, commit 2dce07864, pinned by tests/sm120_correctness/tier1_kernel/
                    test_ds_mla_kv_write.py).  Upstream vLLM (glm52/vllm) stores the raw fp32 s: mode suffix ":raw".
    bytes [528,656) k_pe (post-RoPE) bf16, verbatim  -> lossless.
  read (flashinfer_mla_sparse_sm120.py, trtllm sparse-MLA kernel, kv_scale_format arbitrary_fp32 for GLM):
    K = V = e4m3 * scale (exact in bf16/fp32 for pow2 scales); the query's 512-d absorbed latent is quantised
    kernel-side to e4m3 with per-(head, 128-tile) pow2 scales exp2(ceil(log2(max(amax, 1e-4) / 448))) -- the
    "kernel-matched Q quantization" of vllm/model_executor/layers/quantization/arvq_reference.paged_attention;
    the q rope part stays unquantised.  All tokens (prefill included) read the fp8 cache (forward_mqa only).
Modes (NQ_KVQ):
  kv    KV round-trip only, HF (non-absorbed) attention otherwise unchanged: kv_c = kv_a_layernorm(.) is
        round-tripped before kv_b_proj (== absorbed K/V reading the dequantised cache, exact algebra).
  kvq   full serve emulation: absorbed MLA (q_lat = q_nope W_UK in bf16, as vLLM's bmm), Q latent e4m3 quant,
        fp32 attention over [dequant kv_c | k_pe] (arvq_reference.sparse_attention arithmetic), o_lat -> bf16 ->
        W_UV (bf16) -> o_proj.
  append ":raw" for the upstream raw-fp32 tile scale.  DSA indexer: skipped by the harness (SEQ 2048 = index_topk,
  all causal keys selected), so vLLM's fp8 indexer K cache (indexer_k_quant_and_cache) cannot change the selection.
"""
import types

import torch

FLT_MIN = torch.finfo(torch.float32).tiny
E4M3_MAX = 448.0


def _pow2_up(s):
    """Smallest power of two >= s (s > 0 normal fp32): ilogbf / ldexpf / 'if (p2 < s) p2 *= 2'."""
    m, e = torch.frexp(s)                                   # s = m * 2^e, m in [0.5, 1)
    return torch.where(m == 0.5, s, torch.ldexp(torch.ones_like(s), e))


def e4m3_satfinite(x):
    """__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3): RNE, clamp to +-448 (NaN stays NaN)."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def ds_mla_pack_latent(kv_c, pow2=True):
    """kv_c [..., 512] bf16 -> (e4m3 [..., 512], fp32 scales [..., 4])  == cache bytes [0, 528)."""
    x = kv_c.float().unflatten(-1, (4, 128))
    s = torch.clamp(x.abs().amax(-1) / E4M3_MAX, min=FLT_MIN)
    if pow2:
        s = _pow2_up(s)
    q = e4m3_satfinite(x / s.unsqueeze(-1))
    return q.flatten(-2), s


def ds_mla_unpack_latent(q, s, dtype=torch.bfloat16):
    return (q.float().unflatten(-1, (4, 128)) * s.unsqueeze(-1)).flatten(-2).to(dtype)


def ds_mla_roundtrip(kv_c, pow2=True):
    return ds_mla_unpack_latent(*ds_mla_pack_latent(kv_c, pow2), dtype=kv_c.dtype)


def quant_q_latent(q_lat):
    """arvq_reference.paged_attention kernel-matched Q quantisation of the 512-d latent, per (token, head) tile."""
    t = q_lat.float().unflatten(-1, (4, 128))
    qs = torch.exp2(torch.ceil(torch.log2(t.abs().amax(-1, keepdim=True).clamp_min(1e-4) / E4M3_MAX)))
    return ((t / qs).to(torch.float8_e4m3fn).float() * qs).flatten(-2)


def attend_latent(q_lat, q_rot, kv_c, k_rot, scale, kv=True, q=True, pow2=True):
    """Absorbed causal MLA core over the (emulated) fp8_ds_mla cache.  q_lat [B,H,T,512] bf16, q_rot [B,H,T,64],
    kv_c [B,T,512] bf16 (post kv_a_layernorm), k_rot [B,T,64] (post RoPE) -> o_lat [B,H,T,512] fp32."""
    B, H, T, R = q_lat.shape
    ql = quant_q_latent(q_lat) if q else q_lat.float()
    kv_d = ds_mla_roundtrip(kv_c, pow2).float() if kv else kv_c.float()
    k_pe = k_rot.to(torch.bfloat16).float()                                  # cache stores k_pe as 16-bit verbatim
    qf = torch.cat((ql, q_rot.float()), -1)
    k = torch.cat((kv_d, k_pe), -1).unsqueeze(1).expand(B, H, T, -1)
    return torch.nn.functional.scaled_dot_product_attention(qf, k, kv_d.unsqueeze(1).expand(B, H, T, R),
                                                            is_causal=True, scale=scale)


def _forward_kv(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, position_ids=None,
                prev_topk_indices=None, **kw):
    """HF GlmMoeDsaAttention.forward (transformers glm_moe_dsa, indexer None, sdpa, attention_mask None) with the
    cached latent round-tripped through fp8_ds_mla."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave
    B, T = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(B, T, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c = ds_mla_roundtrip(self.kv_a_layernorm(k_pass), self._kvq_pow2)
    k_pass = self.kv_b_proj(kv_c).view(B, T, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(B, 1, T, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    # == HF path with attention_mask None: causal additive mask (index mask all-False for full causal top-k)
    kpos = torch.arange(T, device=hidden_states.device)
    mask = hidden_states.new_zeros((B, 1, T, T)).masked_fill(
        kpos[None, None, None, :] > position_ids[:, None, :, None], torch.finfo(hidden_states.dtype).min)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    out, _ = sdpa_attention_forward(self, query_states, key_states, value_states, mask, dropout=0.0,
                                    scaling=self.scaling)
    return self.o_proj(out.reshape(B, T, -1).contiguous()), None, prev_topk_indices


def _forward_kvq(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, position_ids=None,
                 prev_topk_indices=None, **kw):
    """Absorbed MLA as vLLM + fp8_ds_mla cache + kernel-matched Q latent quantisation (see module doc)."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave
    B, T = hidden_states.shape[:-1]
    H, dn, dr, dv, R = self.num_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.kv_lora_rank
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(B, T, H, self.qk_head_dim).transpose(1, 2)       # [B,H,T,dq]
    q_pass, q_rot = torch.split(q_states, [dn, dr], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [R, dr], dim=-1)
    kv_c = self.kv_a_layernorm(k_pass)                                                      # [B,T,R] bf16
    k_rot = k_rot.view(B, 1, T, dr)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    W = self.kv_b_proj.weight.view(H, dn + dv, R)
    W_UK, W_UV = W[:, :dn], W[:, dn:]                                                        # [H,dn,R], [H,dv,R]
    q_lat = torch.matmul(q_pass, W_UK)                                                       # [B,H,T,R] bf16 (bmm)
    o_lat = attend_latent(q_lat, q_rot, kv_c, k_rot[:, 0], self.scaling, self._kvq_kv, self._kvq_q, self._kvq_pow2)
    o = torch.matmul(o_lat.to(hidden_states.dtype), W_UV.transpose(1, 2))                   # [B,H,T,dv] bf16
    return self.o_proj(o.transpose(1, 2).reshape(B, T, H * dv).contiguous()), None, prev_topk_indices


def install(attn, mode):
    """mode: kv | kvq | absorbed (kvq path, no quantisation: absorption-numerics control) [+ ':raw']."""
    base, _, sfx = mode.partition(":")
    assert base in ("kv", "kvq", "absorbed") and sfx in ("", "raw"), mode
    assert attn.indexer is None, "KV emulation assumes the harness's exact indexer skip"
    attn._kvq_pow2 = sfx != "raw"
    attn._kvq_kv = base != "absorbed"
    attn._kvq_q = base == "kvq"
    attn.forward = types.MethodType(_forward_kv if base == "kv" else _forward_kvq, attn)
    return attn
