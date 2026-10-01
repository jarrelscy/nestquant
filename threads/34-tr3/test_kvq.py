"""T34 CPU unit check of kvq.py vs the SM120 fork's normative references (numpy fp_codecs + arvq_reference)."""
import importlib.util, math, sys
import numpy as np, torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/34-tr3")
V = "/home/coder/git/vllm-mimo-v26-arvq-sm120"
sys.path.insert(0, f"{V}/tests/sm120_correctness")
from common.fp_codecs import F32, FLT_MIN, f32_to_fp8_e4m3_satfinite   # noqa
sp = importlib.util.spec_from_file_location("arvq_ref", f"{V}/vllm/model_executor/layers/quantization/arvq_reference.py")
AR = importlib.util.module_from_spec(sp); sp.loader.exec_module(AR)
import kvq
torch.manual_seed(0); g = np.random.default_rng(0)

def ref_pack(kv_row, pe_row, pow2=True):
    """test_ds_mla_kv_write._ref_pack + the kernel's pow2 round-up (ilogbf/ldexpf)."""
    out = np.zeros(656, np.uint8); x = kv_row.astype(F32)
    for t in range(4):
        v = x[t * 128:(t + 1) * 128]
        s = np.maximum((np.max(np.abs(v)) / np.float32(448.0)).astype(F32), FLT_MIN)
        if pow2:
            p2 = np.float32(math.ldexp(1.0, math.frexp(float(s))[1] - 1))
            if p2 < s: p2 = np.float32(p2 * 2)
            s = p2
        out[t * 128:(t + 1) * 128] = f32_to_fp8_e4m3_satfinite((v / s).astype(F32))
        out[512 + 4 * t:516 + 4 * t] = np.frombuffer(np.float32(s).tobytes(), np.uint8)
    out[528:] = torch.from_numpy(pe_row).to(torch.bfloat16).view(torch.uint16).numpy().view(np.uint8)
    return out

# 1. cache bytes: random magnitudes per tile, zero tile, tiny values, exact pow2 amax
N = 512
kv = torch.randn(N, 512) * torch.exp(torch.randn(N, 4).repeat_interleave(128, 1) * 4)
kv[0, :128] = 0; kv[1] *= 1e-30; kv[2, :128] = 0; kv[2, 5] = 448.0 * 2 ** -3
kv = kv.bfloat16(); pe = torch.randn(N, 64).bfloat16()
for pow2 in (True, False):
    q, s = kvq.ds_mla_pack_latent(kv, pow2)
    mine = torch.cat([q.view(torch.uint8), s.contiguous().view(torch.uint8).view(N, 16), pe.view(torch.uint8).view(N, 128)], 1).numpy()
    ref = np.stack([ref_pack(kv[i].float().numpy(), pe[i].float().numpy(), pow2) for i in range(N)])
    nb = int((mine != ref).sum())
    print(f"pack pow2={pow2}: byte mismatches {nb} / {ref.size}"); assert nb == 0
    deq = kvq.ds_mla_roundtrip(kv, pow2)
    refd = (torch.from_numpy(ref[:, :512].copy()).view(torch.float8_e4m3fn).float().view(N, 4, 128)
            * torch.from_numpy(ref[:, 512:528].copy()).view(torch.float32).view(N, 4, 1)).view(N, 512)
    assert torch.equal(deq, refd.bfloat16()), "unpack differs"
    rel = ((deq.float() - kv.float()).norm() / kv.float().norm()).item()
    print(f"  roundtrip bitwise == reference unpack; dequant exact in bf16: {torch.equal(refd.bfloat16().float(), refd)}; rel err {rel:.4f}")
cache = torch.from_numpy(np.stack([ref_pack(kv[i].float().numpy(), pe[i].float().numpy()) for i in range(64)]))

# 2. attention core vs arvq_reference.paged_attention (kernel-matched Q quant, fp32 sparse attention) on a real cache
T, H = 64, 4
q_lat = (torch.randn(T, H, 512) * 3).bfloat16(); q_pe = torch.randn(T, H, 64).bfloat16()
kvc, kpe = kv[:T], pe[:T]
idx = torch.arange(T)[None].expand(T, T).clone(); idx[idx > torch.arange(T)[:, None]] = -1
sc = 192 ** -0.5
ro, _ = AR.paged_attention(torch.cat([q_lat, q_pe], -1), cache, idx, sc)
mo = kvq.attend_latent(q_lat.transpose(0, 1)[None], q_pe.transpose(0, 1)[None], kvc[None], kpe[None], sc)[0].transpose(0, 1)
d = (mo.bfloat16().float() - ro.float()).abs().max().item(); rr = ((mo - ro.float()).norm() / ro.float().norm()).item()
print(f"attend_latent(kvq) vs arvq_reference.paged_attention: max|d| {d:.2e} rel {rr:.2e} (bf16 out)"); assert rr < 4e-3
mo0 = kvq.attend_latent(q_lat.transpose(0, 1)[None], q_pe.transpose(0, 1)[None], kvc[None], kpe[None], sc, kv=False, q=False)[0].transpose(0, 1)
print(f"  quant effect on this toy: rel {((mo0 - ro.float()).norm() / ro.float().norm()).item():.2e}")

# 3. real HF GlmMoeDsaAttention (tiny config): kv path with identity round-trip == HF forward bitwise; absorbed ~= HF
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding
cfg = AutoConfig.from_pretrained("/tmp/nestquant/src/glm53-fp8"); cfg._attn_implementation = "sdpa"
cfg.hidden_size = 256; cfg.num_attention_heads = 4; cfg.q_lora_rank = 128
attn = GlmMoeDsaAttention(cfg, 3).bfloat16().eval()
for p_ in attn.parameters(): torch.nn.init.normal_(p_, std=0.05)
with torch.no_grad():
    for n_, p_ in attn.named_parameters():
        if "layernorm" in n_: p_.fill_(1.0)
attn.indexer = None
T = 256; x = torch.randn(2, T, 256).bfloat16(); pos = torch.arange(T)[None].expand(2, -1)
cs = GlmMoeDsaRotaryEmbedding(cfg)(x, pos); topk = torch.arange(T)[None, None].expand(2, T, T)
kw = dict(hidden_states=x, position_embeddings=cs, attention_mask=None, position_ids=pos, prev_topk_indices=topk)
with torch.no_grad():
    y0 = attn(**kw)[0]
    orig_rt = kvq.ds_mla_roundtrip; kvq.ds_mla_roundtrip = lambda t, p=True: t
    kvq.install(attn, "kv"); y1 = attn(**kw)[0]; kvq.ds_mla_roundtrip = orig_rt
    print("kv path with identity round-trip == HF forward bitwise:", torch.equal(y0, y1)); assert torch.equal(y0, y1)
    y2 = attn(**kw)[0]
    kvq.install(attn, "absorbed"); y3 = attn(**kw)[0]
    kvq.install(attn, "kvq"); y4 = attn(**kw)[0]
    r = lambda a: ((a.float() - y0.float()).norm() / y0.float().norm()).item()
    print(f"rel vs HF: kv {r(y2):.2e}  absorbed(no quant) {r(y3):.2e}  kvq {r(y4):.2e}"); assert r(y3) < 1e-2
print("ALL OK")
