# T35 Stage 1 design note: pattern-rate base + ~4-bit residual (GLM-5.3, 2x DGX Spark)

Status: design only. Stage 0 result and verdict are in section 0.

## 0. Stage 0 result (report_partb.json, kvq, 4 bf16conf windows, jF_all k0 hm0.7 sync)

Sanity check: e20_77 (b2.0, s=1, H77) reproduces T34 jF77_hm07 = 0.02607 exactly.

| b | H | GiB | sal pooled | slot4 | KLD | d vs b2.0@6 (paired SE) |
|---|---|---|---|---|---|---|
| 2.0 | 6 | 175.1 | .322 | .164 | 0.05592 | 0 |
| 1.75 | 32 | 174.8 | .599 | .401 | 0.04493 | -0.01099 (0.00392), -19.7% |
| 1.5 | 54 | 175.3 | .712 | .527 | 0.04508 | -0.01084 (0.00383), -19.4% |
| 1.25 | 71 | 174.7 | .773 | .607 | 0.04582 | -0.01010 (0.00560), -18.1% |
| 1.0 | 86 | 174.9 | .815 | .668 | 0.05376 | -0.00216 (0.00251), -3.9% |
| 1.5 | 77 | 195.1 | .791 | .632 | 0.03626 | -35.2% (more memory) |

Verdict:
- GO (moderate) for a base in 1.5-1.75. Every window improves (-13% to -28%). With n=4 the result is 2.8 paired SE.
- The emulation assumes the base error lies along the nq2 error direction, scaled by sqrt(r(b)/r(2)).
- Best b is about 1.75: it ties 1.5 on KLD and has fewer decode bytes and less churn.

## 1. Budget and rates

- Routed-expert budget: 175 GiB. 84.38 GiB per bpw over all 19200 experts; 0.3296 GiB per bpw per hot slot across all layers (sizing.py).
- Current overhead on top of the trellis rate (scales, U/V, headers): about 0.026 bpw average (gate/up ~0.035, down ~0.040).
- Hot level target stays at about 4.0-4.13 total bpw, so that the L4 plane (and the jF predictor trained on L4 deltas) keeps its quality.
- Required residual rate at base b (total ~4.13, same overhead):
  - b = 2.0 (today): residual gate/up 2.0 + down 2.3125.
  - b = 1.75: gate/up 2.25, down 2.5625 (patterns (2,0x8888) / (2,0xAAAA)+).
  - b = 1.5: gate/up 2.5, down 2.8125. Constraint 2*Kgu + Kd ~ 7.78.
  - b = 1.0: gate/up 3.0, down 3.3. This needs residual K=3 plus pattern (not in the kernel today).
- Whether base+residual at 1.5 + 2.5 matches today's 2.0 + 2.0 L4 error is not known. The nested fold is not rate-additive in the usual way, so measure the L4 floor on the 9-expert pilot first (T14 method) before committing.

## 2. Encoder changes (threads/12-reference-encoder/nq_encode.py)

- `base_quant` calls `viterbi(..., 2)`, hard-wired to K=2. Replace it with the pattern Viterbi (`PV.patq` / CUDA ext `nq_fracvit`), with (KA, MASK) chosen per unit: 1.5 = (1,0xAAAA), 1.75 = (1,0xEEEE), 1.25 = (1,0x8888).
- Stream packing: `D.pack_stream(symbols(sk, 2), 2)` assumes 2 bits every step. It needs the variable-bits-per-step packing already used for the residual (`RKB` BITS = 4*(16*KA + popc(MASK)) per 64-weight lane record).
- `prep` uses ks=(2, res_K) and a per-K gsr gain. Add the base pattern as a key, and calibrate a base gain per pattern. Stage 0's gain grid (rate_mse.py) is a starting point: rel-MSE 1.0 0.272, 1.25 0.194, 1.5 0.137, 1.75 0.097, 2.0 0.068 on iid N(0,1). That tracks 4^(2-b) within 1%.
- Residual K: `D.res_K_units` already takes uniform/pos/mask rules. Add residual patterns 2.25/2.5 (K=2 + mask) and 2.5625/2.8125 for down.
- The fold `Q4 = fold(S(sb), S(sr), Mb, N)` hashes trellis states. The mul1 value is a hash of the 16-bit state, so the fold does not depend on base K. No change is expected. But the base state walk changes with a pattern (1-bit steps shift fewer bits into the state), so re-verify fold decode bit-exactness in the reference decoder.
- The manifest records base (KA, MASK) per unit, and the validator (T28) checks it.

## 3. Kernel and decoder (threads/13-moe-layer-kernel/nqmoe.cu)

- Base today: one uint4 per lane record (64 weights x 2 bits = 128 bits), hard-wired. The residual already has templated rates `RK<RC>` (RC5 = 1.5, RC1 = 1.75, step_off, funnel_greedy).
- Plan: template the base through the same `RK<>` machinery and reuse the p4 sub-array layout (uint4 x n4 | uint2 | uint | ushort) for the base plane.
  - 1.5: 96 bits per lane record = uint2 + uint. Aligned.
  - 1.75: 112 bits = uint2 + uint + ushort. Not 32-bit aligned, so it needs the split-array layout (as p4 does).
  - 1.25: 80 bits = uint2 + ushort.
- Decoder (`nq_decode`, reference and CUDA) must accept the base pattern, both for L2-only and L4 = fold(base, residual).
- GB10 = sm_121 (DGX Spark). Build for sm_121 on CUDA 12.9+/13 using the mma.sync path. No tcgen05/TMEM on GB10, so don't use the sm_100 paths. Check the shared-memory limits (99 KB per block) for the ring buffers.
- Memory system: 128 GB unified LPDDR5x at ~273 GB/s per Spark. Decode is bandwidth-bound. Bytes per routed slot ~ base + slot4*(4.14 - base):
  - b2 @ H6: ~2.41 bpw per slot read (KLD-window slot4).
  - b1.75 @ H32: ~2.72; b1.5 @ H54: ~2.90.
  - So a lower base with more hot experts costs more bytes per token. Include that in the final tradeoff, not just KLD.
- Cold residual planes (L4 data for non-hot experts) sit on NVMe. Predictor churn turns into SSD reads. On the KLD windows, churn is 1.62 slots/layer/token at b1.75@32 and 2.23 at b1.5@54 (Part A sm120tf: 1.17 / 1.56). Size this against the Spark's NVMe bandwidth before choosing H.

## 4. Stage 1 pilot

- 9-expert pilot (3 layers x 3 experts, early/mid/late, include a high-salience expert). For each base b in {1.5, 1.75} with its matched residual:
  1. Encode with the pattern-rate base.
  2. Check the reference decode bit-exactly.
  3. Measure L2 and L4 rel-MSE vs FP8 against today's (2.0, 2.0/2.3125).
  4. Compare with Stage 0's emulated s(b) (1.417 for 1.5, 1.191 for 1.75). Stage 0 assumed that the base error scales as sqrt(r(b)/r(2)) with the same direction as nq2. The real encoder error has a different direction, so check whether its KLD effect matches.
- Then one layer end-to-end through the harness (threads/18-e2e-eval), then the full model.

## 5. Fine-tune (after the encoder): user rules

These are user rules for all later stages.

- **Loss only on model-decoded tokens.** Training loss uses only the positions the model itself decoded:
  - run-1 fp8dec (/tmp/nestquant/33-search/ceiling/fp8dec_run1b);
  - the sm120tf teacher-forced dump.
- Prompt/prefill positions and human text (calib, wikitext, etc.) may be forwarded as context but get **zero loss**.
- This applies to every loss term:
  - the logit KL;
  - the salience-weighted router KL against FP8 routing;
  - the layer-local propagated pass. Build its rows from decode positions only.
- **Held-out tasks.** Hold out whole tasks (not windows) for eval. Report held-out decode KLD first, before any other metric.
- **Trainable parameters.** Continuous parameters only: su/sv, delta, low-rank U/V, plus the router gates. Codes stay fixed, so the layout is unchanged (use `layout_check` from threads/27-pv-tune/nq27_tune.py).
- **Train in the serve config.** jF hot set at 4 bits and everything else at the base, with the same predictor, hm and refresh as serving.
- **Loss.** Logit KL plus a per-layer salience-weighted router KL vs FP8 routing. Don't match the exact top-8.
- **Harness.** Extend threads/27-pv-tune's tuner (nq27_tune.py: parameters and layout_check; nq27_band.py: band prep/tune/eval). T27 found that per-expert fits don't carry to full-model KLD because upstream drift and routing dominate (T18 routing attribution). So the fine-tune has to be multi-layer or sequential with propagated inputs, and it has to include the router term.
- **Privacy.** Traces, hidden states and teacher windows stay on the box, except the internal flashblade backup. Tuned artifacts get no uploads without user approval.
