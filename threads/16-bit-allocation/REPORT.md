# Thread 16: bit allocation at fixed average bpw (written by the lead from the agent's final message)

## Verdict
Reject benefit-driven allocation within a projection for GLM at 2.0 and 4.0 bpw: no gain. Keep uniform rates per (layer, expert, projection, plane), no rate map. If a fractional per-projection K forces mixing, use the positional two-class rule (higher rate on the last-processed 128-column chunks in LDL order), no metadata, ~1% better than naive placement on 9/9. Never skip refinement inside the 4-bit tier.

## Setup
Nested 2+2 mul1, blend λ=0.3, δ per 16x128, format rotation (random signs + Hadamard-128 both sides), thread-08 H, same-H EXL3 anchors. Pass 1 uniform fit records exact block-LDL cost per unit for rates {0, 1.5, 2, 2.5, 3}; greedy benefit-per-bit on each unit's convex hull, per TP8 shard (constant bytes); pass 2 refits. Reproduces thread 02 on E36 (34.63 / 9.36 vs 34.59 / 9.35).

Bit accounting: EXL3-2/4 2.0104/4.0104; uniform 2+2 2.0104/4.0182; greedy +0.0011 for rate map.

## 9-expert means (L16, L49, L66 x E36, E92, E165), relative expert-output L2 %

| scheme | routed | forced | OOD-forced | OOD-routed |
|---|---|---|---|---|
| EXL3-2 | 35.75 | 38.27 | 40.46 | 39.99 |
| uniform @2 | 35.64 | 38.16 | 40.05 | 39.37 |
| greedy P4 @2 | 35.67 | 38.16 | 40.05 | 39.31 |
| forced ±0.5 base @2 | 39.44 | 42.13 | 44.41 | 43.64 |
| EXL3-4 | 9.20 | 10.01 | 10.62 | 10.38 |
| uniform @4 | 9.70 | 10.53 | 11.16 | 10.93 |
| greedy P4 @4 | 9.68 | 10.54 | 11.17 | 10.94 |
| forced ±0.5 P4 @4 | 10.64 | 11.59 | 12.33 | 12.04 |
| g/u 1.75 + down 2.5, greedy | 9.88 | 10.96 | 11.79 | 11.44 |
| g/u 1.75 + down 2.5, positional | 9.87 | 10.96 | 11.78 | 11.44 |
| g/u 1.75 + down 2.5, alternate | 9.97 | 11.08 | 11.88 | 11.51 |

Per-expert routed (L16 E36/E92/E165, L49 ..., L66 ...):
- EXL3-2: 34.42 33.06 28.42 39.35 37.19 30.69 40.35 35.81 42.44
- uniform @2: 34.63 33.38 29.23 39.44 36.87 31.14 39.24 35.49 41.36
- EXL3-4: 8.89 8.45 7.26 10.27 9.47 7.82 10.37 9.24 11.02
- uniform @4: 9.36 8.87 7.68 10.85 10.05 8.17 10.98 9.84 11.48

E36 screen: skipping P4 on 1/6 of units gives L4 15.24 (vs 9.36); act-order gives no change.

## Why nothing to allocate
Row-group unit-cost CV ~0.003; columns rise smoothly in LDL order (0.8-1.7x mean, CV 0.15-0.20 gate/up, 0.05-0.07 down). Ideal continuous allocation gain ≤0.054-0.072 dB at L4, ≤0.045 dB at L2 on gate/up, ≤0.011 dB on down. One 0.5-bit step changes unit cost ~2x, so swaps rarely pay.

## Findings
1. No gain from within-projection reallocation at 2.0 or 4.0.
2. Skipping refinement is catastrophic; a cheaper base does not fund refinement (±0.5 on base +10% at L2).
3. Positional rule matches greedy (97% of units) and beats alternate placement 9/9.
4. Down-heavy split (g/u 1.75, down 2.5) is worse than uniform on all 9 experts: +1.8% routed, +4% forced, +5.6% OOD.
5. Coarser δ (per 16 cols x 256-row shard) may save ~0.004 bpw; only indicative.

## Files
alloc_lib.py, run_alloc.py, compose*.py, skip_only.py, anchors.py, evalw.py, aggregate.py, gaincal.py, env.sh (GPU 1); results/e36_screen.json, eval_L*E*.json, alloc_L*E*.json, compose_L*E*.json, nine_expert_summary.json. Weights/curves /tmp/nestquant/16-bit-allocation/L*E*/.
