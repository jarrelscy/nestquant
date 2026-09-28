# Thread 02: feedback conflict in progressive 2→4-bit codes (H2) (written by the lead from the agent's final message)

## Question
How much of the nested 4-bit loss comes from fitting the base to its own 2-bit LDLQ feedback, and which joint fitting rule recovers it? Target: both endpoints within ~2% of native.

## Method
- GLM L16 E36, real FP8 teacher (/tmp/nestquant/glm53-fp8-experts), real Hessians, random orthogonal rotations both sides, per-row RMS normalisation, block UDU, LDLQ in 16-column blocks.
- Main code: nested 2+2 mul1 (native EXL3 mul1 K2 base + one K2 mul1 residual plane with per-block scale; thread 03's code). Mechanism study on a nested scalar code (4 Lloyd base levels, free 16-level decoder reinterpreting base bits).
- Testbed reproduces references: thread-08 H nat2 34.41 vs EXL3-2 34.42, nat4 8.88 vs 8.89; p²/σ0.03 seq L4 12.14 vs thread 03's 12.10, no-feedback base 10.01 vs 9.99.
- Rules (separate E2/E4 feedback states unless noted): seq (native LDLQ-2 base, residual to 4-bit target); blend λ (base target (1−λ)t2 + λt4); gam γ (base target w − γ·E2·M; γ=0 = no-feedback base); blend+gam; joint/CD (scalar only); MatGPTQ shared (one error state); innov (diagnostic: quantise T2−Q2, decode W4 = Q2 + Δ·M⁻¹; exact but needs a dense per-expert triangular transform, not deployable).
- Proxy tr(EHEᵀ); selection on the thread-06 20% holdout (seed 606, p²); final numbers via harness.evaluate() on the matched capture. EXL3 anchors refit under each H with quantize_exl3_like (thread-08 H E36: 9.17/8.89/9.71 vs thread 08's 9.17/8.92/9.69).

## Results (nested 2+2 mul1, GLM L16 E36, relative expert-output L2 %)

### Thread-08 H

| rule | L2 routed | L4 routed | L2 forced | L4 forced | L4 OOD | holdout L2 / L4 |
|---|---:|---:|---:|---:|---:|---|
| **EXL3-2 / EXL3-4 (same H)** | **34.42** | **8.89** | 35.36 | 9.17 | 9.71 | – |
| NVFP4 | – | 12.29 | – | 12.78 | 12.71 | – |
| nat2 / nat4 | 34.41 | 8.88 | 35.38 | 9.16 | 9.70 | 24.00 / 6.09 |
| seq (λ=0) | 34.41 | 9.54 | 35.38 | 9.86 | 10.43 | 24.00 / 6.59 |
| blend 0.1 | 34.55 | 9.48 | 35.38 | 9.80 | 10.36 | 24.35 / 6.55 |
| blend 0.2 | 34.60 | 9.45 | 35.42 | 9.72 | 10.28 | 24.50 / 6.51 |
| **blend 0.3** | **34.59** | **9.35** | 35.48 | 9.67 | 10.22 | 24.98 / 6.47 |
| blend 0.5 | 35.04 | 9.28 | 35.75 | 9.57 | 10.13 | 26.09 / 6.40 |
| blend 0.7 | 35.86 | 9.19 | 36.51 | 9.48 | 10.03 | 28.23 / 6.34 |
| gam 0.75 | 34.72 | 9.42 | 35.44 | 9.71 | 10.27 | 24.70 / 6.48 |
| gam 0.5 | 35.01 | 9.25 | 35.76 | 9.57 | 10.11 | 26.10 / 6.43 |
| gam 0 | 39.62 | 9.16 | 40.47 | 9.44 | 9.99 | 36.05 / 6.31 |
| blend 0.3 + gam 0.75 | 34.91 | 9.28 | 35.71 | 9.58 | 10.14 | 25.86 / 6.40 |
| blend 0.5 + gam 0.75 | 35.36 | 9.20 | 36.14 | 9.51 | 10.06 | 27.26 / 6.34 |
| innov (not deployable) | 34.41 | 9.14 | 35.38 | 9.47 | 10.03 | 24.00 / 6.35 |

### L2 / L4 routed across calibrations

| rule | p²/σ0.03 | p¹/σ0.3 | p¹/σ1.0 | thread-08 H |
|---|---|---|---|---|
| EXL3 anchors | 37.43 / 9.75 | 34.51 / 8.92 | – | 34.42 / 8.89 |
| seq | 37.35 / 12.14 | 34.39 / 9.90 | 34.38 / 9.42 | 34.41 / 9.54 |
| blend 0.3 | 35.67 / 11.21 | 34.43 / 9.56 | 34.74 / 9.28 | 34.59 / 9.35 |
| blend 0.5 | 35.27 / 10.72 | 34.71 / 9.38 | 35.12 / 9.19 | 35.04 / 9.28 |
| gam 0 | 39.62 / 10.01 | 39.62 / 9.16 | – | 39.62 / 9.16 |
| innov | 37.35 / 10.21 | 34.39 / 9.23 | – | 34.41 / 9.14 |

## Findings
1. H2 confirmed; better calibration shrinks it. With H = I nesting is nearly free (scalar 1.00 / 1.01–1.04x native). seq level-4 penalty: proxy 1.9–2.5x (trellis) and 4.4x (scalar) at p²/σ0.03; output +25% at p²/σ0.03, +7.2% under thread-08 H. ~2.7 points of the +7.2% is the code-structure floor (no-feedback base and exact innov both land at 9.14–9.16, matching thread 03's 0.24 dB); the pure feedback conflict is ~4%.
2. blend, gam and combinations lie on one L2/L4 frontier under good calibration. Per-block adaptive λ and refinement-aware alternation reproduce it with no gain; scalar joint+CD is dominated.
3. Target not met. Closest deployable point blend 0.3: L2 +0.5%, L4 +5.1%. Even exact innov is +2.8% at L4. Every rule beats NVFP4 at 4 bit by 24–26%.
4. MatGPTQ-style shared feedback is dominated (scalar λ=μ=0.5: 2.1x proxy at 2 bit, 18x at 4 bit). Separate E2/E4 states are right (agrees with thread 11).
5. Holdout and capture disagree at level 2 (blend 0.3: holdout +4%, capture +0.5%); they agree on level-4 ranking.
6. Decode-side innov approximations fail: M⁻¹ is neither low-rank nor block-diagonal (50% energy needs rank 709 on gate, 273 on down); block-diagonal, low-rank, another expert's M, diag+low-rank H are 15–80x worse on proxy.
7. The code is additive (f4 = f2 + δ·g), matching the T2R2 kernel.

## Verdict: partial adopt
- GLM: nested 2+2 mul1, blend λ = 0.3, separate feedback states: L2 34.59, L4 9.35 routed vs EXL3 34.42 / 8.89, NVFP4 12.29.
- If L2 must equal EXL3-2 exactly: seq (34.41 / 9.54).
- A nested 4.0-bpw level 4 cannot beat EXL3-4 on GLM. Options: accept +3–5% (still ~24% better than NVFP4), or store a separate native 4-bit code.
- MiMo: seq/nat2 (native base); H2 work unnecessary. MiMo proxy conflict was small (seq 1.4–2.0x on real H with Gaussian weights); no MiMo output check (source was gone).

## Files
fb.py (scalar), fbt.py (trellis), gam2.py + eval2.py (calibration frontier), eval_out.py, damp_sweep.py, earlier sweeps; common.py and gam2.py now point at /tmp/nestquant/glm53-fp8-experts. results/: output_calib_frontier{,_ext,_mix}.json, trellis_calib_frontier*_glm*.json, trellis_damp_sweep_glm.json, frontier_*.json, trellis_*.json. Scratch /tmp/nestquant/02-feedback-conflict/.
