# Thread 06: SwiGLU output-aware objective (written by the lead from the agent's final message)

Files: run2.py (grid → grid_{glm,mimo}_{holdout,full}.json), xcheck.py (harness cross-check → xcheck.json), env.sh (with copied lib/libcudart.so.12 for exllamav3). Decoded weights in /tmp/nestquant/06-expert-objective/.

## Setup
- Upstream `quantize_exl3` with `fit_matched_exl3` arguments; each arm changes only H, the output-side metric G, the target and sigma_reg. The base arm reproduces the brief exactly (GLM 40.01/37.41 at 2b, 10.46/9.74 at 4b).
- The matched EXL3 baselines use plain one-sided LDLQ (no YAQA). Downstream-weighted arms use the installed `ldlq_2hess` two-sided rounding.
- Captures: GLM `glm53_matched_context_pilot_v1_capture/layer_16`; MiMo `native_id_control_v1_capture/layer_55` (control) and `ood_controlled_v1_capture/layer_55` (OOD).
- Selection on a seeded 20% holdout of training_sample rows, then refit on all training rows. Cross-checked with harness.evaluate() to ~0.01 pp.
- Notation: c = SiLU′(g)·u for gate, SiLU(g) for up; O = Σ p²·c·cᵀ; G = diag(O)^β.

## Results (relative expert-output L2, %)

**GLM L16 E36** (forced / routed 139 rows / OOD forced; holdout = prob²-weighted held-out training rows)

| Objective | 2-bit | holdout | 4-bit | holdout |
|---|---|---:|---|---:|
| EXL3 matched | 40.01 / 37.41 / 42.57 | 24.66 | 10.46 / 9.74 / 11.15 | 6.37 |
| Sequential down (GPTQ target) | 41.41 / 38.64 / 44.27 | 25.11 | 10.79 / 10.08 / 11.55 | 6.48 |
| Full two-sided G on gate/up | 40.06 / 37.30 / 42.61 | 24.51 | 10.47 / 9.73 / 11.18 | 6.35 |
| Uniform rows | 37.18 / 35.92 / 40.13 | 24.82 | 9.70 / 9.34 / 10.48 | 6.35 |
| prob¹, damping 0.3 | 35.99 / 34.51 / 38.26 | 23.46 | 9.35 / 8.93 / 9.95 | 6.00 |
| **prob¹, damping 0.3, G β=0.5** | **35.90 / 34.33 / 38.21** | 23.41 | **9.34 / 8.90 / 9.95** | 5.99 |
| NVFP4 | | | 12.78 / 12.29 / 12.71 | |

**MiMo L55 E70** (control forced / control routed 31 rows, noisy / OOD forced; holdout ~6.5k mostly-routed training rows is the reliable routed estimate)

| Objective | 2-bit | holdout | 4-bit | holdout |
|---|---|---:|---|---:|
| EXL3 matched | 39.47 / 15.99 / 45.77 | 19.77 | 10.43 / 4.00 / 12.05 | 5.02 |
| Sequential down (own sample-only baseline 20.84 / 5.32) | 43.15 / 16.52 / 49.04 | 21.05 | 11.39 / 4.29 / 13.06 | 5.40 |
| Two-sided, G = I | – | 19.99 | – | 5.07 |
| G = WdᵀWd | – | 21.94 | – | 5.59 |
| Full two-sided G | 40.10 / 14.73 / 46.33 | 18.33 | 10.74 / 3.77 / 12.59 | 4.65 |
| G β=1 | 40.23 / 14.61 / 46.67 | 18.33 | 10.82 / 3.73 / 12.67 | 4.66 |
| **G β=0.5** | **39.19 / 14.93 / 45.11** | 18.55 | **10.42 / 3.85 / 12.11** | 4.72 |

## Findings
1. Sequential down correction: reject. Training error −10–12%, held-out/eval +1–4% on both models, both bit widths, every ridge. Down's Hessian from quantized activations alone is neutral.
2. Downstream-weighted gate/up works through the output Hadamard, zero inference cost. All gain comes from per-channel activation energy (WdᵀWd hurts; G = I gains nothing). It scales with the energy spread: AM/GM 1.6 (gate) / 2.1 (up) on MiMo, 1.03 on GLM E36, hence ~0.1 pp on GLM. β=1 specialises to routed channels (MiMo held-out routed −7%, forced/OOD +2–5%); β=0.5 keeps ~6% routed gain with no forced/OOD loss. β=0.5 was picked after seeing β=1 on eval captures; confirm on another expert.
3. Row weighting and damping are the biggest GLM lever (calibration fix). prob² weighting leaves ~990 effective rows for a 6144-dim input (training 14% vs eval 37–40%). prob¹ + damping 0.3 cuts ~10% at both widths. MiMo (~200k effective rows) stays best at prob² + 0.03. The GLM EXL3 baseline is held back by this; matched comparisons must give EXL3 the same choice.
4. The holdout ranking is identical at 2 and 4 bit in every grid, so one objective serves base and refinement.

## Recommendation
- Gate/up: tr(E·H·Eᵀ·G), H = Σ w·xxᵀ, G = diag(Σ w·c²)^0.5, rounded with `ldlq_2hess`.
- Down: plain teacher-activation Hessian.
- Row weights/damping per expert from effective sample size: prob¹ + ~0.3 when prob² ESS is ≤ a few thousand rows, prob² + 0.03 when large.
- Versus matched EXL3: GLM 2b routed 37.4 → 34.3, 4b routed 9.74 → 8.90 (OOD 11.15 → 9.95, 27% below NVFP4). MiMo ~6% lower routed at both widths, forced/OOD unchanged.

## Lead's note on fairness
Most of the GLM gain (prob¹ + damping) is available to EXL3 as well. The objective-only gain is the β=0.5 G row vs the prob¹/0.3 row: ~0.2 pp at 2b, ~0.03 pp at 4b on GLM; on MiMo, held-out 19.77 → 18.55 (2b) with EXL3's own calibration settings.

## Addendum (agent's restated final message under the narrowed scope)
- Under the new targets, the recommended objective beats matched EXL3 on GLM at 2 and 4 bit and on MiMo at 2 bit (MiMo held-out 18.55 vs 19.77), with the same EXL3 format and inference cost.
- The agent notes it did not run a separately labelled "EXL3 refit at p¹ + sigma_reg 0.3". The "p¹ weights, damping 0.3" arm is the upstream EXL3 quantizer with only H changed, so the lead treats it as that fair baseline (GLM 34.51 / 8.93 routed).
- Code files: eo.py (helpers), run.py (first pass → results_glm.json), run2.py (main grid), xcheck.py, env.sh.
