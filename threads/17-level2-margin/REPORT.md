# Thread 17: GLM 2-bit margin over EXL3-2 (written by the lead from the agent's final message)

## Verdict
Target (≥3% on all 9 experts at equal bpw) not reached. Adopt per-tile sign + two-sided G (β=0.5) + blend λ=0.3 for the level-2 encoder: +1 bit/tile (+0.0039 bpw), 0 decode ops (sign folds into mul1's final hfma constants), level-4 format unchanged. sg4 (3 bits/tile) is break-even after paying its rate.

## Setup
Same H for all arms (thread-08 mix, σ 0.5/0.5/1.0); anchors harness EXL3-2 (2.0104 bpw) and EXL3-4. "nat" reimplementation matches EXL3-2 within ±0.01% on 9 experts; routed seed noise ±0.25%. Net = raw minus measured rate slope (~1.08% rel L2 per 0.01 bpw; 1 bit/tile ≈ 0.42%).

| candidate | bpw | E36 L2 r/f/o | 9-expert L2 Δ vs EXL3-2 (r/f/o) | L4 Δ vs EXL3-4 (r/f/o) | net L2 |
|---|---|---|---|---|---|
| nat | 2.0104 | 34.42 / 35.36 / 37.38 | 0 / 0 / 0 | +7.0 / +7.0 / +6.9 | 0 |
| G β=0.5 | 2.0104 | 34.32 routed | E36 −0.3 | – | −0.3 |
| blend λ=0.3 | 2.0104 | 34.53 / 35.36 / 37.11 | E36 +0.3 / 0 / −0.7 | E36 9.31 | ≈0 |
| block CD after LDLQ | 2.0104 | capture ±0 | rejected | – | 0 |
| seqW down (+sign+G) | 2.0143 | 34.07 / 35.03 / 37.20 | −1.06 / −0.94 / −0.64 | breaks L4 (E36 10.5) | −0.6 / −0.5 / −0.2 |
| per-tile sign | 2.0143 | 34.17 / 35.17 / 37.27 | −0.72 / −0.56 / −0.54 | – | −0.3 / −0.1 / −0.1 |
| sg4 | 2.0221 | 34.05 / 35.01 / 37.09 | −0.82 / −1.10 / −1.08 | – | ≈ +0.3 |
| **sign+G+b0.3** | 2.0143 | 34.26 / 35.14 / 36.90 | **−1.10 / −1.30 / −1.82** | +3.75 / +3.87 / +3.89 | **−0.7 / −0.9 / −1.4** |
| sg4+G+b0.3 | 2.0221 | 34.13 / 34.88 / 36.65 | −1.39 / −1.88 / −2.47 | +3.21 / +3.18 / +3.16 | −0.1 / −0.6 / −1.2 |

Per expert, sg4+G+b0.3 at L2: routed +0.4% (L16E165) to −2.8%, forced −1.4 to −2.4%, OOD −1.9 to −3.0%; mean abs 35.10 / 37.55 / 39.45 vs EXL3-2 35.61 / 38.27 / 40.45.

## Why capped
After rotation and damping, weighting inside each 16-weight block is flat (AM/GM 1.0001-1.002). Only headroom is the trellis's 0.39 dB gap to the RD bound; 3% needs ~0.26 dB and per-tile side info buys back about its own bit cost. Diagnostic: an H built from the capture itself (overfit, not deployable) drops routed L2 from 34 to 9-16, so calibration quality dominates coding.

## Rejected
Global su/sv reseed (within noise), g-scale multiplier, better ring closure (best-of-4 starts ~0.2% L2 for 4x encode).

## Open
seqW down helps L2 0.4-0.5% but needs its own level-4 target. Agent suggested down-heavy allocation; thread 16 found it worse on 9/9, so not pursued.

## Files
core17.py (quantizer), t_*.py, t_confirm9.py, summarize9.py, results/*.json (confirm9.json, confirm9_summary.txt). Logs /tmp/nestquant/17-level2-margin/.
