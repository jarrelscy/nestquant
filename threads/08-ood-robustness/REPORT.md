# Thread 08: OOD robustness at 4 bit (written by the lead from the agent's final message)

Files: sweep.py, select.py, selection_{2,4}.json, sweep_final_4.json, sweep_final_4_all.json, sweep_final_2.json, subspace.json, baselines.json, table4.md, table2.md, env.sh (exllamav3 needs cuda12 runtime on LD_LIBRARY_PATH). Scratch: /tmp/nestquant/08-ood-robustness/.

## Method
- Upstream exllamav3 `quantize_exl3` fed a custom H. Refit with saved stats reproduces matched EXL3 to <5e-4 relative weight difference.
- Selection on a seeded 80/20 split of the training sample: router-weighted, unweighted and a "tail" subset (held-out rows with most energy in the Hessian's low-eigenvalue directions, a training-only OOD stand-in). Rule fixed in select.py before any capture scoring: best mean of the three, no worsening of router-weighted error on any expert.
- Final: refit on 100% of the training sample, scored once on `glm53_matched_context_pilot_v1_capture/layer_16.pt` (12 control, 8 ood documents). "ID" = forced on control tokens, "all" = forced on all 5,120 tokens.

## 1. Why calibrated quantizers lose on OOD
- p²-weighted effective rows ~900 (E36) to ~1,900 for a 6,144-dim input, so the Hessian's low-eigenvalue directions are mostly sampling noise.
- LDLQ puts the error there: EXL3-4 error per unit weight energy ~0 in the top 1% of directions, ~0.012 in the bottom 50%. NVFP4 is flat at 0.004–0.008.

| E36 inputs | energy outside 99% subspace | EXL3-4 error from bottom-50% directions |
|---|---:|---:|
| training rows | 1% | 25% |
| capture ID | 14% | 64% |
| capture OOD | 22% | 72% |

| E36 | EXL3-4 | NVFP4 |
|---|---:|---:|
| training rows | 4.0 | 10.4 |
| held-out training rows | 7.0 | — |
| capture ID | 10.0 | 12.8 |
| capture OOD | 11.2 | 12.7 |

OOD is the far end of a continuum: finite-sample Hessian overfitting. Fixing it helps ID and OOD together.

## 2. GLM capture, 4 bit (mean over E36/E92/E165)

| method | all | ID | OOD | routed |
|---|---:|---:|---:|---:|
| EXL3-4 matched | 10.52 | 9.98 | 11.49 | 8.98 |
| NVFP4 | — | 12.58 | 12.88 | — |
| σ=0.3 | 9.68 | 9.28 | 10.41 | 8.29 |
| σ 0.5 gate/up, 1.0 down | 9.48 | 9.14 | 10.12 | 8.21 |
| p¹ + σ=0.3 (thread 06) | 9.34 | — | 10.14 | 8.17 |
| p¹ + 0.5/1.0 | 9.22 | — | 9.93 | **8.15** |
| **selected: 75% uniform-token mix + 0.5/1.0** | **9.16** | **8.77** | **9.87** | 8.22 |

Per expert (all / routed / OOD), selected vs EXL3-4: E36 9.17/8.92/9.69 vs 10.45/9.73/11.15; E92 9.24/8.46/10.12 vs 10.69/9.28/11.92; E165 9.07/7.29/9.81 vs 10.43/7.92/11.42.

| OOD domain (mean of 3) | EXL3-4 | NVFP4 | selected |
|---|---:|---:|---:|
| fasta | 11.89 | 13.19 | 10.14 |
| encoded_bytes | 11.62 | 13.06 | 9.96 |
| smt_bitvectors | 11.48 | 12.86 | 9.86 |
| scientific_telemetry | 11.16 | 12.63 | 9.67 |

Training-row error rises ~1.45x (overfitting removed); bytes and decode unchanged; OOD/ID ratio 1.152 → 1.125. Cross-check: p¹ + σ=0.3 on E36 gives 9.35/8.89/9.94 vs thread 06's 9.35/8.93/9.95.

## 2-bit check

| method | ID | OOD | routed (E36 / E92 / E165) |
|---|---:|---:|---|
| EXL3-2 artifact | 38.30 | 43.84 | 37.33 / 35.83 / 30.25 |
| σ=0.3 | 35.73 | 39.91 | — |
| **selected** | **33.88** | **38.02** | 34.39 / 33.01 / 28.63 |

The same setting was selected independently at 2 bit.

## Didn't help
Less damping (0.01, 0.003); OAS shrinkage (picks σ ~0.02–0.2, <1% gain); balanced or minimax reweighting of token groups; replacing H with the broad prior entirely (hurts routed); a layer-wide prior behaves like the uniform-token prior (gain is token diversity). YAQA output-side Hessian not tested; it uses the same tokens, so combine it with this shrinkage.

## Verdict: adopt
H = (1−α)·H_routed/tr + α·H_uniform/tr, α ≈ 0.5–0.75, plus σ·mean(diag)·I with σ ≈ 0.5 gate/up, 1.0 down. Anything in α 0.5–0.75, σ 0.3–1 is within ~1% of best; p¹ or p² for the routed part both work. The 4-bit L16 bar becomes ~9.2 all / 9.9 OOD / 8.2 routed.

Caveats: L16 pilot only (~18k rows); with the 15M-token calibration the best σ should shrink, so re-select on a held-out split. Routed cohorts are small (139/287/120). OOD documents are synthetic structured text.
