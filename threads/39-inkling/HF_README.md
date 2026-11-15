---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Small
tags:
- nestquant
- moe
- quantized
---

# Inkling-2-Small NestQuant 1.75/4 bit

NestQuant encode of the routed experts of [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B total, 17B active, 48 layers, 384 experts, top-8). Target: 2x RTX PRO 6000 (2 x 96 GB).

Each expert holds two nested levels in one encode:

| Level | Bits per weight | Size, all 18,432 experts |
|---|---|---|
| L2 (base) | 1.76 | 129 GB |
| L4 (base + residual) | 4.13 (gate/up 4.04, down 4.35) | 303 GB |

At serve time every expert is resident at L2, and a hot set of experts per layer is upgraded to L4.
On 2x RTX PRO 6000 the memory left after the L2 copy and the BF16 non-expert weights fits about 64 hot experts per layer.

## Contents

- `layers/L{0..47}/tp{0,1}.safetensors`: the encoded experts, split in 2 tensor-parallel shards
  (the down projection's 2304 intermediate channels / 8 = 288 is not a multiple of 128, so TP4/TP8 shards are not possible).
- `layers/L{L}/manifest.json`: format, per-projection layout, rates and sha256 of each shard.
- `COMPLETE`: written last; lists every layer file with its sha256.
- `serving/predictor/`: floating hot-set predictor (weights, inference module, training code). See its README.

Only the routed experts are here. Take the attention, embedding, vision encoder and MTP weights from the original repo.
Container format `nestquant-v1-tp-safetensors`: tensor keys `E{e}.{proj}.{plane}`, structure in the safetensors metadata `tree`.

## Calibration

- Source weights: the native checkpoint (NVFP4 experts on layers 0-39, MXFP8 on 40-47).
- Text: a 7.5M-token mix of chat, code, reasoning and agent transcripts, in Inkling's chat template.
- Images: 1,200 images (radiology, web screenshots, COCO, synthetic documents), 640 px, 256 image tokens each.
- Each expert's error weighting is 75% text and 25% image.

## Quality

KLD against the native model (top-64 log-probs plus tail mass), teacher-forced on held-out data.
"Static hot 64" = the 64 most-routed experts per layer at L4, the rest at L2 (fits 2x RTX PRO 6000).
"Floating hot 64" = 64 experts per layer at L4, chosen per request every 16 tokens by `serving/predictor/`.

| Setting | Text KLD (119K tokens) | Text top-1 | Image-caption KLD (14K tokens) | Caption top-1 |
|---|---|---|---|---|
| All experts L2 | 0.0361 | 94.7% | 0.0779 | 90.9% |
| Static hot 64 | 0.0179 | 96.7% | 0.0663 | 91.1% |
| Floating hot 64 (`serving/predictor`) | 0.0114 | 97.6% | 0.0365 | 93.2% |
| Floating hot 64 (salience EMA, no network) | 0.0114 | 97.5% | 0.0398 | 93.2% |
| All experts L4 | 0.0060 | 98.5% | 0.0227 | 95.8% |

The floating sets use the same memory as the static set. Without the 474 tool-JSON tokens noted below, text KLD is
0.0074 for the predictor and 0.0079 for the EMA heuristic. With all 384 experts hot, the floating eval path
reproduces the all-L4 result exactly.

The image calibration costs nothing on text and roughly halves the image-caption KLD. A text-only calibrated build scores
0.0179 text / 0.1223 caption KLD at hot 64.

About 0.005 of each text value comes from a small set of tool-definition JSON tokens that flip under any change to
layer 0, including 1% random noise. Every quantization of this model shares that floor.

## Limits

- The predictor is trained on calibration text, not on Inkling's own generations.
- The floating KLD is teacher-forced: the predictor reads the routing of the same quantized model, block by block.
- No serving integration is published yet.
