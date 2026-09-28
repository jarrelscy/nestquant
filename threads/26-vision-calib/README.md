# T26: vision calibration for the GLM-5.3 expert Hessians

This thread adds vision-token statistics to the calibration. The text Hessians come from T19's capture, and the
encoder blends each expert's Hessian as H = 0.75·H_text/tr + 0.25·H_vision/tr, the v3 recipe. The image tokens come
from the grafted GLM-5.2V vision tower and its original projector. The hidden states come from the same FP8
GLM-5.3 reference forward that T19 uses.

## Pipeline
| step | script | output |
|---|---|---|
| dataset (CPU) | `build_calib_mm.py` (glm52 `tools/capture53/build_calib_mm.py`, paths changed, + held-out) | `/tmp/nestquant/calib-mm` |
| mm corpus (CPU) | `mm_corpus.py` | `/tmp/nestquant/19-capture-mm/corpus/c2048_mm` (T21 group layout + `rowkind.npy`, `images.json`) |
| vision features (GPU, ~1 GB, 3 min) | `vis_feats.py` | `corpus/c2048_mm/feats.bf16` [1400, 256, 6144] |
| stage 1 + stage 2 | `launch_full.sh <gpu1> <gpu2...>` → `capture_mm_fwd.py` (T24 `capture_fwd_fast.py` + splice) and `capture_mm_stats.py` (T19 `capture_stats.py` restricted to real rows) | `/tmp/nestquant/19-capture-mm/stats/L{3..77}` (nestquant-19-stats-v2) |
| blend | `nq26_blend.BlendCapture(text_cap, vision_cap, 0.25)` or `nq19_load_mm.patch` (`Capture(..., mm_root=...)`) | encoder H / G, fixed-set score |

The dataset has 4 domains × 300 calib samples plus 4 × 50 held-out samples. The domains are ROCOv2 radiology,
WebSight screenshots, COCO-Caption2017 natural images and SynthDoG-en OCR. All images are 448×448, which gives 256
vision tokens each. The seed, streams and filters are the same as in the earlier campaign. The held-out samples are
the same streams continued past sample 300, with ids `<domain>_0300..0349`.

Each sample is one attention segment, in the grafted model's GLM-native chat template:
`[gMASK]<sop><|user|><|begin_of_image|>256×<|image|><|end_of_image|>{prompt}<|assistant|><think></think>{caption}`.
The prompt is "Read the text in the image." for OCR and "Describe the image." for the other domains. Samples are
first-fit packed into 2048-token windows. The padding at the end of a window is dropped before any statistic is
computed. The environment variable `NQ26_ROWS` chooses which rows go into the statistics: `valid` (the default,
image and caption rows, as in v3) or `image` (image rows only).

## Restore after a /tmp wipe
```bash
T=/home/coder/git/nestquant/threads/26-vision-calib
$T/fb_backup26.sh restore calib-mm                       # dataset (1.7 GB)
$T/fb_backup26.sh restore 19-capture-mm                  # corpus + feats (4.4 GB) + small stats files, no grams
# The raw grams (*.f32, ~43 GB/layer, 3.3 TB) are NOT on flashblade (budget). Rebuild them deterministically
# (~1 h on 4 slots). This needs the GLM-5.3 FP8 source in /tmp/nestquant/src/glm53-fp8 and the vision files in
# /tmp/nestquant/vision-graft.
rm -rf /tmp/nestquant/19-capture-mm/stats /tmp/nestquant/19-capture-mm/stats0 /tmp/nestquant/19-capture-mm/shards /tmp/nestquant/19-capture-mm/bnd_rows
$T/launch_full.sh <gpu> <gpu> <gpu> <gpu>
```
If the corpus is also gone, rebuild it with `run.sh mm_corpus.py`, then `CUDA_VISIBLE_DEVICES=<g> run.sh vis_feats.py`.

## Result (full run, 2026-09-29 01:23-01:56 Melbourne)
- It took 33 min. Stage 1 ran on 1 GPU (6.9 GB CUDA, 39.4 GiB RSS peak) and stage 2 on 3 workers. 1400 images were spliced, and there are 75 layers of stats: 3.1 TB on /tmp,
  with 17.9 GB of small files on flashblade. `/tmp/nestquant/19-capture-mm/DONE` summarises the run.
- `coverage.py` → `coverage.json` counts routed vision rows per expert. Across all layers, 214 experts have n_v < 128
  (0-15 per layer, most in L10-L22) and 2021 have < 1000. Only L15 E50 has no vision rows at all. The median is ~5.3k.
- The vision REAP salience is mostly different from text: Spearman against the text sal averages 0.18, ranging -0.08 to 0.52.
- `final_check.py` checks that the blend is finite with mean diag 1, that w_eff is 0 for L15 E50, and that the fixed-set score sums to 1.
