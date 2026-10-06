---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: image-text-to-text
tags:
- vision
- multimodal
- image-text-to-text
- quantization
- moe
- glm
---

# GLM-5.3-Flash NestQuant 1.5-4 bit

GLM-5.3-Flash with its routed experts quantized to NestQuant, a nested two-level format. Each expert has a
**1.5-bit base** plus a residual plane that lifts it to **4 bits**. At serving time an expert moves from 1.5 bit to
4 bit by loading its residual plane on top of the base, and the base bytes stay the same.

This build targets a **128 GB Mac with 96 GiB usable** for weights, KV cache and runtime. 96 GiB is the macOS default
GPU wired limit on a 128 GB machine (75% of RAM). The native vision tower is included, so the model still takes images.

**Status: complete.** All 42 expert layers, the backbone, vision tower, MTP layer and the floating-set predictor are
uploaded. This repo cannot be loaded with stock transformers, vLLM or MLX. The serving kernel and loader will be
published separately.

## Sizing at 96 GiB

Sizes in GB (1e9 bytes), Mac setup without MTP.

| Part | Format | Size in memory |
|---|---|---|
| Routed experts, layers 3-44 (42 x 288), 1.5-bit base | NestQuant base | 59.5 GB |
| 4-bit residual planes, 19 fixed experts per layer | NestQuant residual | 6.6 GB |
| 4-bit residual planes, 63 floating experts per layer (32K context) | NestQuant residual | 22.0 GB |
| Backbone: attention, dense MLP, shared experts, router, norms, mHC, embeddings, lm_head | MLX q8 at load (fp8/bf16 as shipped) | 9.6 GB (15.2 GB as shipped) |
| Vision tower + merger | bf16 | 1.1 GB |
| KV cache + KDA state | | 0.5 GB at 32K, 1.7 GB at 128K |
| MLX workspace, process runtime, allocator slack, predictor state | | 3.6 GB |

- One 4-bit slot across all 42 layers costs 350 MB.
- At 32K context, 19 fixed + 63 floating = **82 of 288 experts per layer run at 4 bit** (28%). At 128K, 19 + 59 = 78.
- The **fixed** 19 per layer always stay at 4 bit.
- The **floating** experts are picked by the predictor in `serving/predictor/` ahead of routing.
- The hot pool is sized at launch, so the same files run with more 4-bit experts on a larger machine. At 104 GiB it is
  19 + 87 at 32K.
- macOS has to allow the wired GPU memory. The default `recommendedMaxWorkingSetSize` is about 96 GiB on a 128 GB Mac;
  `sudo sysctl iogpu.wired_limit_mb=...` raises it. The runtime should read the limit at boot and size the floating
  pool from it.

## Contents

| File | What | Size |
|---|---|---|
| `layers/L{L}/tp{0..7}.safetensors`, L = 3..44 | Expert planes of layer L (1.5-bit base and 4-bit residual), split for tensor parallel 8 | 3.92 GB per layer, 164.8 GB total |
| `layers/L{L}/manifest.json` | Layout of layer L, the fixed set, per-file sha256 | |
| `nonexpert-0000{1..4}-of-00004.safetensors` | Everything that is not a routed expert: attention (Kimi KDA linear attention, plus DSA/MLA every 4th layer), mHC hyper-connections, dense MLP (layers 0-2), shared experts, router gates and `e_score_correction_bias`, norms, embeddings, lm_head, and the MTP layer 45 apart from its routed experts. Copied byte for byte from zai-org/GLM-5.3-Flash (FP8 where the source is FP8, bf16/fp32 elsewhere). | 15.5 GB |
| `vision_tower.safetensors` | The native vision tower and merger (`model.visual.*`), bf16, byte for byte | 1.1 GB |
| `mtp_experts-0000{1,2}-of-00002.safetensors` | The 288 routed experts of the MTP layer 45, FP8, byte for byte | 7.2 GB |
| `model.safetensors.index.json` | Index of all the above (3,532 tensors) | |
| `nonexpert_manifest.json`, `nonexpert_tensor_sha256.json` | Per-file and per-tensor sha256 | |
| `fixed_set.json` | The 19 fixed experts per layer, with their scores | |
| `serving/predictor/` | Floating-set predictor for 63 floating experts (default) | |
| `serving/predictor_nf48/` | The same predictor trained for 48 floating experts, for smaller budgets | |
| `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json` | From zai-org/GLM-5.3-Flash. `config.json` adds a `quantization_config.nestquant` block; all other keys are unchanged. | |

### MTP layer

The MTP layer is kept as the source FP8, as in the GLM-5.3 NestQuant releases. Its routed experts (7.25B
parameters, 7.2 GB) are in separate files. They do not fit in 96 GiB next to the floating pool, so the Mac setup runs
without MTP and does not load `mtp_experts-*`. Machines with more memory can load them for speculative decoding.

## Method

- **Rotation:** random signs + Hadamard-128 on both sides of each weight matrix.
- **1.5-bit base:** a pattern-rate bitshift trellis code. Step p of each 16-step window uses
  1 + ((0xAAAA >> (p % 16)) & 1) bits, so 24 bits per 16 weights. It has a per-tile sign and is fitted with LDLQ.
- **4-bit residual:** a second trellis code on the rotated residual, fitted jointly with the base. It is 2.5 bits per
  weight on gate/up and 2.8125 on down (2 bits plus a 0xFBDE pattern, residual code 10; the serving kernel has to
  support it). Measured rates: 1.52 bpw at level 2 and 4.14 bpw at level 4, including scales and the low-rank plane.
- **Calibration:**
  - 4.0M text tokens.
  - 1,200 images (radiology, web screenshots, natural images, OCR), run through Flash's own vision tower.
  - Per-expert Hessians blend 75% text and 25% vision.
- **Fixed set:** the 19 experts per layer with the highest boundary-weighted REAP score (routing weight times expert
  output norm, summed over tokens).
  - Tokens just before the end of reasoning and the end of each turn get more weight: 50 for the last token, 20 for 2-4
    tokens before, 5 for 5-16 and 2 for 17-32. This keeps the experts that decide when to stop at 4 bit.
  - Text and vision scores are blended 75/25.
- The weights of Flash's experts are close to iid Gaussian after rotation (kurtosis 2.996-3.004 on 60 matrices), as in
  GLM-5.3. The 1.5-bit base error is about 2x the 2-bit error.

### Floating-set predictor

`serving/predictor/` is the joint predictor (jF) from the GLM-5.3 NestQuant releases, retrained on Flash routing
traces (80% decode, 20% prefill, FP8 model). Every 16 tokens it scores all 288 experts per layer and keeps the top 63
as floating, with hysteresis 0.7 on experts already loaded.

Share of routed expert salience (routing weight squared times input norm) served at 4 bit, on held-out test
sequences:

| Floating set | Decode | Prefill |
|---|---|---|
| Static (fixed set + default floating set, never updated) | 35.2% | 52.3% |
| v2 GBDT predictor | 76.7% | 78.0% |
| **jF, 63 floating (default)** | **77.8%** | **79.0%** |
| jF, 48 floating (`predictor_nf48`) | 72.0% | |

Seeding the decode set from the prefill routing raises the first 16-token block from 37.5% to 72.1%. The GPU serving
path picks the same top-63 sets as the reference (672/672 on the parity check).

## Quality

### Expert output error

Relative L2 error of each expert's output on held-out calibration rows, routed rows only, for the worst fixed and
worst ordinary expert in each layer's spot check. The comparison is EXL3 fitted on the same Hessians at the same rates.

| | Relative error | vs EXL3 |
|---|---|---|
| Level 2 (1.5 bit) | about 44% | -1% to +6% in most layers; +8% to +10% in L6 and L31; +15% to +17% in L3 |
| Level 4 (4.14 bit) | about 7.5% | -1.5% to -6.5% in every layer except L3 (-0.8% / +1.1%) |

The early-layer gap at 1.5 bit is a property of Flash's early experts. A 2-bit encode shows the same gap against
EXL3 at 2 bit.

### KL divergence against the BF16 teacher

Full-vocabulary KL(BF16 ‖ quant) on confirmation windows 0000-0003 of
[brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits)
(teacher: `zai-org/GLM-5.3-Flash-BF16`; 4 × 2,048 tokens, 8,188 positions; general, legal, code-agentic and reasoning
text). fp8 KV cache: the MLA latent is stored as e4m3 with four power-of-two scales per token (vLLM's `fp8_ds_mla`
layout), and the query latent is quantised to e4m3 at attention time. Two numbers matter on the Mac:

- **Prefill.** Layer-major prefill streams every expert of the next layer at 4 bit, so a prefilled prompt sees all
  experts at 4 bit.
- **Decode.** Each token sees the 19 fixed and the jF-predicted floating experts at 4 bit and the rest at 1.5 bit.
  Every window is scored as a cold-start decode from its first token, with the default floating set and swaps every
  16 tokens. This is pessimistic: in real use the prompt is prefilled first and seeds the floating set.

| Setup | 4-bit experts per layer | KLD | Top-1 agreement with BF16 | Perplexity |
|---|---|---|---|---|
| FP8 source | all | 0.0193 | 95.7% | 2.771 |
| **Prefill (layer-major, all experts at 4 bit)** | **288** | **0.0248** | **94.9%** | **2.774** |
| **Decode, 96 GiB, 32K (19 + 63 jF)** | **82** | **0.0797** | **90.9%** | **2.861** |
| Decode, 96 GiB, 128K (19 + 59 jF) | 78 | 0.0843 | 90.4% | 2.871 |
| 19 + 63, static floating set | 82 | 0.195 | 85.3% | 3.141 |
| All experts at 1.5 bit | 0 | 0.301 | 81.6% | 3.493 |

BF16 perplexity on these windows is 2.773. Per window (FP8 / prefill / decode 32K):
general 0.019 / 0.018 / 0.113, legal 0.020 / 0.043 / 0.113, code-agentic 0.013 / 0.015 / 0.044, reasoning
0.025 / 0.023 / 0.049. Differences below about 0.005 are within run-to-run numeric noise.

### KL divergence against the FP8 source

Full-vocabulary KL divergence against the FP8 source model, run end to end over 146 held-out windows (106,348
positions, 85% of them decode). Floating sets come from the jF predictor in serving order.

| Setup | 4-bit experts per layer | KLD, all | KLD, decode | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|
| FP8 source | all | 0 | 0 | 100% | 3.080 |
| All experts at 4 bit | 288 | 0.084 | 0.071 | 94.8% | 3.079 |
| **This repo, 96 GiB, 32K (19 + 63 jF)** | **82** | **0.172** | **0.127** | **91.9%** | **3.081** |
| This repo, 96 GiB, 128K (19 + 59 jF) | 78 | 0.176 | 0.131 | 91.9% | 3.078 |
| 19 + 63, static floating set | 82 | 0.210 | 0.167 | 90.4% | 3.086 |
| All experts at 1.5 bit | 0 | 0.308 | 0.245 | 86.4% | 3.110 |

- The predictor recovers 61% of the KLD gap between all-1.5-bit and all-4-bit; the static set recovers 44%.
- Flash is sensitive to small numeric changes. Re-running the FP8 model with a different batch layout gives KL 0.058
  against its own stored logits, because bf16 rounding flips near-tied router choices. KLD values below about 0.06
  are within that noise.
- These windows are chat traces, and 10% of positions are system, user or tool-output text, which Flash predicts
  poorly (FP8 perplexity about 93 there). Those positions carry most of the KLD. Split by context:

| Positions | Share | All 4 bit | 19 + 63 jF | 19 + 59 jF | Static 19 + 63 | All 1.5 bit |
|---|---|---|---|---|---|---|
| Assistant turns | 65% | 0.007 | 0.021 | 0.022 | 0.036 | 0.086 |
| Plain text (no chat roles) | 26% | 0.031 | 0.076 | 0.077 | 0.119 | 0.211 |
| System, user, tool output | 10% | 0.75 | 1.46 | 1.49 | 1.64 | 2.08 |

## Verification

- All 3,532 source tensors outside the routed experts of layers 3-44 are here exactly once. Each one's dtype, shape and
  sha256 match the source (`nonexpert_tensor_sha256.json`). None of the 72,576 routed-expert tensors of layers 3-44
  are included.
- Every layer was converted to safetensors, round-tripped, decoded at both levels and compared with the encoder before
  upload, then spot-checked against EXL3 (above). The LFS sha256 of every file on the Hub matches its manifest.
- The KLD run decodes the expert planes with the reference decoder, which is bit-exact against the encoder. Running
  the FP8 arm with the teacher's exact batch shapes reproduces the stored logits to KL 1e-6.
