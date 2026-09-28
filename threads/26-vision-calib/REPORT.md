# T26 report: vision calibration for the GLM-5.3 expert Hessians (complete)

**Outcome.** The production encoder can now use vision Hessians. The recipe is H = 0.75·H_text/tr + 0.25·H_vision/tr
per expert, taken from `/tmp/nestquant/19-capture-mm/stats/L{3..77}`, which is in T19's stats-v2 format and
includes the vision salience.

## What was built
- **Dataset.** `/tmp/nestquant/calib-mm` holds 4 domains × 300 calib samples, built with seed 42 and the same
  recipe as glm52: ROCOv2, WebSight, COCO-2017 val and SynthDoG-en. There is also a disjoint held-out split of
  4 × 50 samples. All images are 448×448, which gives 256 vision tokens each. This is backed up to flashblade.
- **Capture.** Image tokens are the GLM-5.2V tower's output passed through the original projector. They are
  spliced into T24's `capture_fwd_fast` FP8 GLM-5.3 forward. T19's `capture_stats` then runs on the real rows only
  (309.6k image rows and 67.7k caption rows; padding is dropped).
- **Run.** The full run took 33 min on 01:23-01:56 Melbourne, 2026-09-29. Stage 1 used 1 GPU, peaking at 6.9 GB
  VRAM and 39.4 GiB RSS; stage 2 used 3 workers. All 1400 images were spliced and all 75 layers have finite stats.
- **Blend.** `nq26_blend.BlendCapture(text, vision, 0.25, n_min=128)` produces the blend.
  - An expert with n_v routed vision rows gets vision weight w_e = 0.25·min(1, n_v/128).
  - `fixed_set_score` computes 0.75/0.25 on salience column 4.
  - T12's `nq_layer` uses the blend through `--stats-mm`.
  - `nq19_load_mm.patch` is an optional loader patch for T19.

## Findings
- **Coverage is good.** Only L15 E50 has no vision rows; it gets the pure text H. 214 experts have n_v < 128 and so
  get a reduced vision weight. 2021 experts have n_v < 1000. The median is ~5.3k rows per expert. The full
  breakdown is in `coverage.json`.
- **Vision routes to different experts than text.** The Spearman correlation between vision and text salience
  averages 0.18 (range -0.08 to 0.52). It is close to 0 in the middle layers.
- **The blend changes H a lot.** T12 measured a relative Frobenius difference of 0.2-0.56 between blended and
  text-only H, so the vision term reshapes H rather than slightly adjusting it. Before the production encode I
  recommend a one-layer held-out text-KLD A/B: text-only vs blended.

## Durability
- **Backed up to flashblade:** calib-mm (1.75 GB) and 19-capture-mm small files + corpus + features (17.9 GB).
  Both were checked with a dry-run sync, which showed nothing left to upload.
- **Not backed up:** the gram files (`*.f32`, 3.1 TB), because of the flashblade budget. They rebuild
  deterministically with `launch_full.sh` in ~35 min.
- The restore procedure is in the README.

Commits: 7a443fa, 904a0ce, and this report.
