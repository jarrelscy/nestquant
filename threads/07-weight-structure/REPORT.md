# Thread 07: weight structure (written by the lead from the agent's final message)

## Verdict
No codec-exploitable structure beyond i.i.d. Gaussian after Hadamard rotation, on GLM-5.3 or MiMo (GLM-5.2 finding holds). All gains are measured as data minus an i.i.d. Gaussian control through the same code. The one real opportunity is a spec choice for MiMo: use the MXFP4 source verbatim as the 4-bit endpoint.

Scripts/JSON: ws.py, run_per_expert.py, cross.py, fp4.py, fp4_refine.py, trellis_entropy.py, *.json. CPU only, ≤16 threads, peak ~3.2 GB.

## Data
GLM-5.3 L16/L49 from verified shards in /tmp/orbit-duet-glm53-fp8; L66 from orbit-duet/runs/source_glm53/layer_66; experts 36/92/165. MiMo: only raw MXFP4 L55 E70 (runs/source_mimo). Random-sign block-128 Hadamard (input side or both sides). Gaussian reverse water-filling at 2/4 bit + unit-Gaussian Lloyd-Max check.

## 1. Post-Hadamard marginal and scales
| | GLM-5.3 (9) | MiMo E70 | iid control |
|---|---|---|---|
| excess kurtosis before/after | 0.01–0.10 / ±0.003 | 0.14–0.18 / ≤0.007 | 0 |
| P(\|z\|>4σ) before | 7e-5–1.8e-4 | 1.6e-4–4.7e-4 | 6.3e-5 |
| 2b scalar SNR tensor/row/16-block | 9.30/9.30/9.74 | 9.29–9.30/9.30/9.74 | 9.30/9.30/9.74 |
| 4b scalar SNR | 20.18/20.19/20.66 | 20.18/20.18–20.19/20.66 | 20.19/20.19/20.66 |
| alloc gain rows/cols/16-blocks | ≤.003/≤.003/.277 | ≤.004/≤.004/.280 | .001/.002/.277 |
Per-row scale ≤0.01 dB beyond control (EXL3 already stores row/col fp16 scales). Per-16 scale gain equals the Gaussian control (normalisation artefact) and would cost ~0.5 bpw. Unrotated GLM is already near Gaussian (0.02–0.11 dB below rotated; MiMo ≤0.13).

## 2. Structure before rotation
- Outliers: top 1% columns hold 1.1–1.6% energy; column allocation ≤0.03 dB before, 0.002 after rotation.
- Low rank: top-32 SVs hold GLM 4–8% (chance ~4%), MiMo 7.5–10% (3.9%). Net with 8-bit factors ≤0.005 bpw (rank 1), negative for rank ≥2.
- Gate vs up: GLM L16 conditioning up on gate saves 0.028–0.038 bits/up weight (41–43% of up energy in gate row space vs 33% chance); L49/L66 ≤0.008; MiMo 0.017. Row-space overlap survives a shared rotation; paired-row correlation vanishes with row rotation (ρ² ~5e-4).
- Cross-expert (GLM L16/L49, leave-one-out over 66 experts): mean expert useless (cos ≤0.009, ≤0.0004 dB); best-matching rows at chance (0.048 vs 0.050), no upcycling signature. Layer-shared KLT basis gains 0.07–0.16 dB (0.01–0.03 bits), needs per-coefficient rates; 0.002 dB on Gaussian or Hadamard basis. MiMo not testable (one expert).

## 3. MiMo MXFP4 grid
- Lossless: nibble entropy 3.82–3.83 bits; context saves ≤0.05; block scales carry 0.68–0.84 bits each (0.02–0.03 bpw vs 0.25). Lossless 3.84–3.85 bpw vs 4.25 raw, needs variable-length decode.
- Nested FP4 prefixes: 2-bit nested 8.3–8.4 dB, best split 9.0–9.1 (trellis ~11.6); 3-bit nested 13.8–13.9, best 15.1–15.2 (trellis ~17.6 est.). ~3.2/3.7 dB loss. Reject.
- Trellis base refined to exact FP4: 2-bit base + ideal entropy-coded refinement = +1.81 bits (~3.84 total, no cheaper than lossless source); 2-bit + 2 fixed bits 94.5% exact but 18.0 dB (< native 4-bit trellis ~23.5); LDLQ-style base needs 2.06 bits (matches orbit-duet's 2.0–2.5); 3-bit base + 1 bit 93.5% exact, 23.9 dB (≈ native 4-bit). All need decode in the unrotated domain (inverse Hadamard of the base, ~7 adds/weight), breaking the ops budget. Reject for the progressive path.
- Verbatim MXFP4 as 4-bit endpoint: 4.25 bpw, zero error vs reference (EXL3-4 at 4.00%), LUT decode. Upgrade streams 4.25 bpw instead of +2; base unused at 4 bit.

## 4. EXL3 symbol entropy
30 tensors in 20 matched .bin (18 GLM, 2 MiMo): 2-bit 2.0000 bits/symbol with or without context; 4-bit 4.0000 (≥3.9983 given previous two); 16-bit words 15.97–15.99 (estimator bias). Entropy coding saves ≤0.002 bpw.

## Ranking
| # | structure | gain | survives Hadamard | verdict |
|---|---|---|---|---|
| 1 | MiMo 4-bit = verbatim MXFP4 | 4.00% → 0 at 4.25 bpw | n/a | adopt if spec allows |
| 2 | MiMo 3-bit trellis + 1-bit FP4 refinement | ≈ native 4-bit | no | reject |
| 3 | layer-shared KLT | 0.01–0.03 bits | yes | reject |
| 4 | up conditioned on gate | ≤0.04 bits/up weight (L16) | row space only | reject |
| 5 | low rank + residual | ≤0.005 bpw | yes | reject |
| 6 | per-row/col scales | ≤0.01 dB (EXL3 has them) | no | nothing to do |
| 7 | per-16 scales | 0 dB beyond Gaussian, ~0.5 bpw cost | no | reject |
| 8 | entropy-coding trellis symbols | ≤0.002 bpw | n/a | reject |
| 9 | mean-expert / delta coding | 0 | n/a | reject |

## Recommendation
Treat rotated weights as iid Gaussian; spend effort on rate allocation, feedback, code strength and kernels. Open decision: may MiMo's 4-bit view be the MXFP4 source (4.25 bpw, ignoring the base)?

Caveats: weight-domain only (Hessian-weighted structure is thread 01). MiMo = one expert. 3/4-bit trellis SNRs are estimates.
