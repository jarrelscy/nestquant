# Thread 09: post-quantization continuous tuning (written by the lead from the agent's final message)

Files: summary_table.md (full per-run table), results_refit/ (damping sweeps). All errors are relative expert-output L2 in %, p²-weighted, same arithmetic as evaluate(); decode reproduces receipts exactly (GLM 40.01/37.41 at 2b, 10.46/9.74 at 4b forced/routed; MiMo 15.99 routed at 2b). Tuned candidates were not separately run through thread 05's evaluate().

**Break-even:** on GLM's matched RD curve (2/3/4 bit: 40.01/20.54/10.46), one bpw of codes cuts error ~66%. A side parameter must beat 0.7% for 0.0104 bpw, ~22% for 0.333 bpw.

## Gain per parameter type (routed, base → tuned)

| parameter | +bpw | GLM E36 2-bit | GLM E36 4-bit | MiMo 2-bit | MiMo %/bpw |
|---|---|---|---|---|---|
| fine-tune su/sv scales | 0 | +0.08% | −0.01% | −0.78% | free |
| per-projection output bias | 0.0043 | −0.06% | −0.07% | **−1.07%** | ~250 |
| rotated-domain row/col scales | 0.0104 | +0.08% | +0.02% | −0.86% | ~83 |
| 33-knot LUT / free value table | 0 / 0.0012 | +0.04 / +0.07% | −0.03% | +0.05 / −0.36% | – |
| 16×16 tile scales (8-bit) | 0.031 | +0.02% | −0.02% | −1.48% | ~47, below break-even |
| scales + bias + LUT | 0.015 | +0.01% | −0.05% | −1.24% | ~84 |
| combined + tile | 0.046 | −0.04% | −0.05% | −1.75% | ~38, below |
| low-rank r8 | 0.083 | −0.36% | −0.33% | −2.20% | ~26, below |
| low-rank r32 | 0.333 | −1.36% | −1.36% | −4.05% | ~12, below |

Low-rank GLM gain is almost all from SVD init. Forced/OOD change ±0.3% for cheap parameters; MiMo 4-bit within ±0.8%. Sign vectors can't be tuned post hoc. Discrete polish skipped.

## Overfitting
- The codes overfit. EXL3 refit on 90% of the training sample: train 15.5% vs clean held-out 28.0% at 2b, 3.9% vs 7.2% at 4b. A split of the same calibration rows shows 18.0% and hides this.
- Post-tuning has nothing to fit on GLM (~1.8k routed rows): ≤0.33% on clean held-out, 0% on captures. MiMo gains grow with data (bias −1.0% at ~4k rows; r32 −1.9% at 1.6k → −4.05% at 29k).

## Damping (0 bits)
Clean held-out picks sigma_reg 0.3 for all 9 GLM pilot experts (L16/49/66 × E36/92/165) at 2 and 4 bit.

| GLM L16 E36 | forced | routed | OOD forced |
|---|---|---|---|
| EXL3-2, 0.03 | 40.01 | 37.41 | 42.57 |
| EXL3-2, 0.3 | 37.05 | 34.97 | 38.84 |
| EXL3-4, 0.03 | 10.46 | 9.74 | 11.15 |
| EXL3-4, 0.3 | 9.64 | 9.11 | 10.10 |
| NVFP4 | 12.78 | 12.29 | – |

Across 9 experts: routed −6.0–9.1% (2b), −6.5–8.8% (4b); forced −6.6–9.5%; OOD −7.4–10.5%. The in-sample held-out gets worse with damping (17.97 → 19.30), so selecting on it rejects damping. MiMo (408k rows): routed 15.99 → 16.47, forced/OOD improve. Post-tuning on sigma 0.3 codes adds ~0.

## Verdict
- Reject as format features: low-rank, tile scales, rotated row/col scales, LUT/value table.
- Optional encoder-side: fine-tune sign×scale magnitudes against the SwiGLU output with held-out early stopping (free); per-projection output bias (0.0043 bpw, ~−1% routed when data is rich, i.e. MiMo 2-bit).
- Highest priority: a damping knob tuned on codes refit without the held-out rows.
- Fair comparison: GLM EXL3 baselines at the same damping.

Lead's note: thread 08's mixed Hessian (75% uniform + σ 0.5/1.0) supersedes plain sigma 0.3 as the GLM calibration.
