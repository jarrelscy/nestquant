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

**Status: encoding complete (all 75 expert layers, 19,200 experts).** This repo cannot be loaded with stock vLLM or transformers yet; a serving kernel and loader will be published separately.

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

Relative output error of individual experts against FP8, compared with EXL3 on the same calibration data (150 experts, two per layer across layers 3-77: one from the default 4-bit set and one other; negative is better):

| Level | Mean vs EXL3 | Worst expert |
|---|---|---|
| 4 bit | -5.1% | +6.6% |
| 2 bit | +1.5% | +18% |

At 2 bit, layers 3-6 are the weakest (mean +10% vs EXL3); layers 7-77 average +1.0%. End-to-end model evaluations are in progress and will be added here.

## Default 4-bit set

`manifest.json` lists 26 experts per layer that are kept at 4 bit by default. They were chosen by usage on the calibration data (boundary-weighted REAP), weighted towards tokens just before the end of reasoning and the end of each turn, blended 75% text and 25% vision. The rest can be upgraded to 4 bit at runtime.

## Serving layout

`serving/tp4/` holds the same experts pre-packed for the NestQuant streaming server at tensor parallel 4, so the server does not have to repack anything at startup. The files, per rank r (0-3) and layer L (3-77):

- `rank{r}/L{L}.bin`: the 256 level-4 records of layer L. Each record is 2,560,000 bytes (a multiple of 4 KiB) and holds, in order, gate|up P4, gate|up block words, down P4, down block words and the rank-4 fp16 low-rank U4 plane. Every segment starts on a 256-byte boundary. Record format `nq-p4rec-v1`.
- `res/rank{r}/L{L}.pt`: the resident planes of the layer (2-bit base, sign variant, scales, low-rank V/U2). Format `nq-res-v1`.
- `layers/L{L}.json`: the per-layer block. It has the size and sha256 of every file, the record layout, the default 4-bit set, the floating default and routing counts, and a `layer_hash`.
- `rank{r}.json`: the index. It gives rec_bytes, segment offsets, L0=3 and NE=256, plus the file, offset, bytes and sha256 of each layer.
- `manifest.json`: formats, the layout, the per-layer hashes, the default allocation (`default_allocation`, 26 experts per layer), `floating_default`, and the per-layer `n_routed` counts.
- `COMPLETE`: every layer and its hash. It is written last. The release is complete only when this file exists.

The server reads one record file per rank, with the record of (L, E) at ((L - 3) * 256 + E) * rec_bytes. `serving/nq_assemble.py` copies the per-layer blocks into `serving/tp4/rank{r}.bin` at those offsets. It is a pure byte copy with sha256 checks (a few minutes on an SSD; `--move` deletes each block after copying it). After that, `serving/tp4/` can be used directly as the server's record directory. When layers are re-fitted, only their blocks and index entries change. The record format name changes if the layout ever changes.
