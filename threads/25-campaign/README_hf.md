---
license: mit
base_model: zai-org/GLM-5.3
tags:
- quantization
- moe
- glm
---

# GLM-5.3 NestQuant 2-4 bit

GLM-5.3 with its routed experts quantized to NestQuant, a nested 2/4-bit format. Every expert has a 2-bit base and an optional 4-bit residual plane. At serving time an expert can be switched from 2 bit to 4 bit by loading its residual plane on top of the base. The base bytes do not change.

**Status: encoding in progress.** Layers are uploaded as they finish. This repo cannot be loaded with stock vLLM or transformers yet; a serving kernel and loader will be published separately.

## Contents

| Part | Format | Size |
|---|---|---|
| Routed experts, layers 3-77 | NestQuant: 2-bit base (2.014 bpw) + 4-bit residual (4.126 bpw total), plus a small low-rank correction on some experts | ~5 GB per layer |
| Attention, dense MLP (layers 0-2), shared experts, router, norms, embeddings, lm_head | FP8 / bf16, copied byte for byte from zai-org/GLM-5.3 | 21 GB |
| MTP layer 78 | FP8, copied byte for byte | 10 GB |
| Vision tower + projector | copied from jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid | 0.9 GB |

Files:
- `layers/L{L}/tp{0..7}.safetensors`: expert planes for layer L, split for tensor parallel 8, with `layers/L{L}/manifest.json` describing the layout and the default 4-bit set.
- `nonexpert-*.safetensors` and `model.safetensors.index.json`: everything that is not a routed expert, plus the vision weights.
- `config.json`: vision-language config (text config in `text_config`). `config.text.json` is the text-only config.

## Method

- Rotation: random signs + Hadamard-128 on both sides of each weight matrix.
- 2-bit base: bitshift trellis code (K=2, L=16) with per-tile sign, fitted with LDLQ against a blend of the 2-bit and 4-bit targets.
- 4-bit residual: a second trellis code on the rotated residual (2 bits per weight on gate/up, 2.3125 on down), fitted jointly with the base.
- Low-rank correction: experts whose input has a few very large activation channels get a rank 1-4 fp16 correction (about +0.014 bpw on average).
- Calibration: 15.4M tokens of text plus 1,200 images (radiology, web screenshots, natural images, OCR). Per-expert Hessians blend 75% text and 25% vision.

## Quality

Relative output error of individual experts against FP8, compared with EXL3 on the same calibration data (48 experts across layers 3-77; negative is better):

| Level | Mean vs EXL3 | Worst expert |
|---|---|---|
| 4 bit | -5.3% | +2.9% |
| 2 bit | +2.0% | +16% |

End-to-end model evaluations will be added when encoding is complete.

## Default 4-bit set

`manifest.json` lists 26 experts per layer that are kept at 4 bit by default. They were chosen by usage on the calibration data, weighted towards tokens just before the end of reasoning and the end of each turn. The rest can be upgraded to 4 bit at runtime.
