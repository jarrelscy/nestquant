# Thread 01: rate allocation over block-LDL innovations (written by the lead from the agent's final message)

**Lead's caveat:** this thread treated `native_id_control_v1_capture` and `ood_controlled_v1_capture` as GLM. Both are MiMo captures. The "native forced" GLM columns below are therefore invalid. The routed held-out GLM numbers use the matched capture (139 routed rows) and stand. MiMo does have frozen captures (those two files), which were not used here.

## Verdict
H1 mostly rejected. On GLM, allocation gives nothing on held-out data and hurts when allocated from in-sample pilot Hessians. It pays only on MiMo down combined with a TP-compatible act-order. Validation used LDLQ + scalar Lloyd-Max (absolute errors worse than EXL3: 48.8 vs 37.4 routed on GLM E36), so only deltas matter.

## Method
EXL3 transform (damp 0.025·mean diag, signs, 128 Hadamard, block LDL 16x16, last block first). Per-column scale + per-block fp16 scale. Ladders: whole bits 0–8, ~half-bit, fine (1–64 levels); greedy by marginal return per bit. Variants: feedback-aware weighting, act-order, TP8-compatible act-order for down, fixed bytes per TP8 shard, row and 2D allocation. Metrics: undamped proxy; relative expert-output error on 8192 training rows (routed p² and plain); matched GLM capture for held-out routed. 4-fold CV: L fitted on 3/4 of training rows, innovations measured on the held-out 1/4.

## Ideal gains (in-sample Hessians)
Clamps never bind, so gain = AM/GM, identical at 2/3/4 bit. Values: AM/GM dB gate_up/down; whole-bit; half-bit.
| expert | AM/GM | whole-bit | half-bit |
|---|---|---|---|
| GLM L16 E36 | 0.63/0.36 | 0.21/0.06 | 0.54/0.26 |
| GLM L16 E92 | 0.72/0.21 | 0.28/0.00 | 0.63/0.11 |
| GLM L16 E165 | 0.62/0.40 | 0.20/0.08 | 0.52/0.30 |
| GLM L49 E36 | 0.49/0.31 | 0.12/0.04 | 0.39/0.19 |
| GLM L49 E92 | 0.58/0.34 | 0.19/0.01 | 0.48/0.23 |
| GLM L49 E165 | 0.60/0.47 | 0.18/0.06 | 0.50/0.37 |
| GLM L66 E36 | 0.41/0.28 | 0.07/0.02 | 0.30/0.17 |
| GLM L66 E92 | 0.54/0.35 | 0.13/0.03 | 0.44/0.26 |
| GLM L66 E165 | 0.43/0.29 | 0.09/0.03 | 0.32/0.19 |
| MiMo L55 E70 | 0.11/0.41 | 0.01/0.07 | 0.03/0.32 |

GLM spread is a small-sample artifact (smooth positional trend, first blocks ~2.3x mean, last ~0.5x; ~18k rows for 6144 dims). Out of sample: GLM gate_up in-sample 0.40–0.72 → CV 0.02–0.08 → matched ≤0.011 dB; down 0.21–0.47 → 0.01–0.04 → ≤0.031. Held-out/in-sample innovation ratio 2.6–4.4x, i.e. LDLQ on pilot GLM Hessians is heavily overfit. MiMo down spread is real (0.47 dB CV vs 0.41 in-sample, profile correlation 0.99); MiMo ratio 1.55x/1.07x.

## Real quantizer, GLM L16 E36, damping 0.025 (routed %)
| scheme | 2b train | 2b held-out | 4b train | 4b held-out |
|---|---|---|---|---|
| uniform | 21.30 | 48.82 | 5.83 | 14.22 |
| whole-bit, in-sample D | 20.04 | 51.73 | 5.61 | 15.05 |
| fine, in-sample D | 19.43 | 50.72 | 5.38 | 14.75 |
| fine, feedback-aware | 19.40 | 50.97 | 5.38 | 14.80 |
| fine, CV D | 20.63 | 48.94 | 5.64 | 14.26 |
| fine + act-order | 19.51 | 50.67 | 5.37 | 14.81 |
In-sample gains ~1.1 dB (gate/up), 0.55 dB (down); held-out all worse or neutral.

## Side finding: damping (uniform rates, GLM E36 held-out routed)
| damping | 2b | 4b |
|---|---|---|
| 0.025 | 48.82 | 14.22 |
| 0.1 | 46.69 | 13.58 |
| 0.3 | 45.51 | 13.16 |
| 1.0 | 45.26 | 13.03 |
~0.7 dB at both 2 and 4 bit. (The agent's "native forced" columns used a MiMo capture and are omitted.) At high damping allocation is within ±0.3 pp of uniform. Orbit-duet EXL3 refits used sigma_reg 0.03 (thread 05), so a fair comparison needs EXL3 refit with the same tuned damping.

## MiMo L55 E70 (training-sample rows; routed %, down proxy dB)
| scheme | 2b | 3b | 4b |
|---|---|---|---|
| uniform | 26.43 (−14.37) | 14.00 (−19.96) | 7.28 (−25.64) |
| fine, no shard constraint | 25.43 (−14.90) | 13.45 (−20.49) | 7.02 (−26.17) |
| fine, fixed TP8 shard budget | 25.56 (−14.81) | 13.49 (−20.42) | 7.03 (−26.12) |
| TP act-order alone | 28.07 (−13.37) | 14.97 (−18.88) | 7.80 (−24.54) |
| act-order + whole-bit ±1 split | 24.46 (−15.74) | 13.13 (−21.07) | 6.90 (−26.58) |
| act-order + fine | 24.21 (−15.78) | 12.96 (−21.16) | 6.80 (−26.76) |
| 256 Hadamard on down, uniform | 26.00 (−14.67) | – | 7.15 (−25.97) |
TP-compatible act-order: sort intermediate channels by energy, deal round-robin to 8 shards, sort within shard; folds into gate/up row order at no cost. Sorted spread 1.93 dB in-sample, 2.22 dB CV (corr 0.999). MiMo gate/up allocation ~0.12 dB. GLM: act-order does nothing for down; allocator picks uniform.

## Row / 2D allocation
Row AM/GM dB (gate/up): GLM E36 per row 0.16/0.14, per 16 rows 0.016, per 128 rows 0.002; MiMo 1.95/3.27, 0.20/0.40, 0.03/0.05. Greedy row allocation stays uniform; 2D ≈ block allocation (±0.03 pp). Adds nothing.

## Nested decodable scheme
Greedy is nested by construction. MiMo down + TP act-order: in every 256-channel shard (16 blocks), 8 high-energy blocks at R̄+1, 8 low at R̄−1: 1/3 → 2/4 → 3/5 bits at levels 2/3/4. Every plane is a constant +1 bpw per shard, no rate metadata (learned profile would cost 0.00012 bpw). Meets thread 10's constant-bytes constraint.

## Gains
GLM held-out: 0 dB (−0.2 to −0.5 from in-sample D). MiMo gate/up ~0.1 dB (~0.02 bit). MiMo down act-order + ±1: proxy +1.37/+1.11/+0.94 dB at 2/3/4 bit; expert output −0.67/−0.56/−0.46 dB (≈0.12/0.10/0.08 bpw at expert level).

## Recommendations
1. Down: TP-compatible act-order + 128 Hadamard + fixed ±1 split, per layer, enabled only when CV shows a gain (neutral on GLM). Kernel: two rate classes per level across 16-row slabs.
2. Gate/up: uniform rates.
3. Never allocate from in-sample D on pilot-size statistics.
4. Raise LDLQ damping on GLM pilot statistics to ~0.3–1.0, tuned on held-out training rows; retune with full-corpus statistics.

## Caveats
Scalar not trellis. MiMo outputs on training rows. GLM L66 weights missing (shards downloading); L66 E165 down from an unverified shard. GLM held-out routed rests on 139 rows.

## Addendum: real trellis validation (thread 05 harness, mul1, sigma_reg 0.03, scored with evaluate())
gate/up uniform at base K. down: act-order-shard permutation, then per TP8 shard the 8 low-energy blocks get K−d and the 8 high-energy blocks K+d (bpw-neutral).

**MiMo L55 E70** (proxy + routed on training-sample rows; no MiMo eval capture used):

| Base K | uniform proxy / routed | ±1 proxy / routed | proxy gain | routed gain |
|---|---|---|---|---|
| 2 | 0.01984 / 0.1892 | 0.01702 / 0.1793 | −0.67 dB | −0.47 dB |
| 3 | 0.00488 / 0.0945 | 0.00418 / 0.0897 | −0.67 dB | −0.45 dB |
| 4 | 0.00124 / 0.0479 | 0.00106 / 0.0455 | −0.68 dB | −0.45 dB |

About 0.11 bit on the down proxy and ~0.075 bpw on expert output. ±0.5 gets nearly all of it (proxy 0.01707 at K2, 0.00426 at K3). Permutation without the split is worse (+1.1 dB proxy). sigma_reg 0.3 gives the same gains (−0.82 / −0.56 dB at K2).

**GLM** (L16/L49/L66 E36, matched eval capture): ±1 is badly worse (down proxy +2.3 to +3.0 dB, routed +0.9 to +1.4 dB; L16 E36 K2 0.3741 → 0.4408). ±0.5 is worse (+0.14 to +0.33 dB routed). Reversed split worse, permutation alone neutral. With small per-block energy spread the Jensen penalty dominates.

**Harness limit:** K=4.5 fails in exllamav3 1.5.1 (`quantize_tiles_frac: no instance for (KA, MASK) = (4, 43690)`). Mixed-integer per-block K (1/3, 3/5) and half-integer K at 1.5/2.5/3.5 work.

## Final recommendation
- Uniform by default: gate/up everywhere, and GLM down.
- Per-layer gate for down: enable act-order-shard + ±1 split (1/3 → 2/4 → 3/5 at levels 2/3/4; nested, every shard averages base K, zero metadata) only where CV AM/GM of act-order-shard innovations is large (MiMo-like, ~2 dB). GLM fails this gate.
- Never allocate from in-sample D on the pilot stats.
- Damping 0.3–1.0 on pilot stats; EXL3 must be refit at the same damping for a fair comparison.
- Net value: ~0.075 bpw on MiMo-like experts, zero on GLM.

Caveats: MiMo output numbers are in-sample (profile is CV-stable). GLM held-out is 139 rows for L16 E36. L66 E165 down came from an unverified shard. The L16 E36 trellis run crashed at K4 ±0.5 before saving JSON; K2/K3 are only in /tmp/nestquant/01-rate-allocation/tv_0.03.log.

Files: trellis_validate.py, trellis_validate_mimo.py, results/trellis_validate_*.json, results/trellis_validate_mimo_{0.03,0.3}.json.
