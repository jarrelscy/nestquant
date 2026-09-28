# NestQuant v0 design (lead, 2026-09-28 19:20 AEST)

Status: frozen enough to build. Open items are marked OPEN; the owning thread decides them.

## Targets (user)
- GLM 5.3: beat EXL3 at 2 and 4 bit, and NVFP4 at 4 bit, all against the FP8 reference, including OOD. EXL3 gets the same calibration fix (fair anchors below).
- MiMo 2.6: beat EXL3 at 2 bit only.
- Speed: beat EXL3 at batch 1–4 at 2 and 4 bit on A100.

## Level 2: FROZEN (user accepted 1-2% margin)
Per-tile sign (1 bit/256-weight tile, folded into mul1 hfma constants) + two-sided G β=0.5 on gate/up + blend λ=0.3. 9-expert: −1.1/−1.3/−1.8% vs EXL3-2 routed/forced/OOD at 2.0143 bpw. Remaining work is level 4 only.

## Fair anchors (GLM, relative expert-output L2 %, routed)
- EXL3 with thread 08's H: 2 bit E36 34.39 / E92 33.01 / E165 28.63; 4 bit mean over E36/92/165: all 9.16, OOD 9.87, routed 8.22 (E36 8.92 / E92 8.46 / E165 7.29).
- NVFP4 4 bit: E36 12.29, E92 11.60, E165 10.65 routed.
- MiMo L55 E70 EXL3-2: 15.99 routed on the control capture, 19.77 on held-out training rows.

## Format
1. Rotation: EXL3 random signs (su/sv) + Hadamard-128 on both sides. Down's input rotation stays inside a 256-wide TP8 shard.
2. Level 2 (base): EXL3 mul1 bitshift trellis, K=2, L=16, 256-weight tiles, tail-biting rings of at least 128 weights (kernel constraint from thread 04; thread 03 measured 0.06 dB at 128).
3. Level 4: base + δ_b · g(P4). g is a second K=2 L=16 mul1 trellis coding the rotated residual; δ_b is one fp16 scale per 16x16 block folded into the final HFMA2. Decode is additive (combined-window reinterpretation rejected by thread 03).
4. Level 3: dropped (user 2026-09-28: only 2 and 4 bit needed). Mixing 2b/4b blocks inside the 4-bit tier is allowed as allocation (thread 16).
5. Planes: base, P3/P4 stored as separate contiguous chunks per (layer, expert, plane, TP shard), constant size per (layer, plane, shard), 64 KiB aligned. Level-specific metadata (δ, block masks) travels with its plane.
6. MiMo only: act-order-shard permutation of down's input channels + ±1 per-block split (1/3 → 2/4 → 3/5), enabled per layer when CV AM/GM of act-order-shard innovations is large (thread 01).

## Fitting
1. Hessian (GLM, small calibration): H = 0.25·H_routed/tr + 0.75·H_uniform/tr, then + σ·mean(diag)·I, σ = 0.5 gate/up, 1.0 down (thread 08). MiMo (large calibration): router-p² H, σ = 0.03. Rule: choose by effective sample size on a held-out split whose rows are left out of the refit (thread 09).
2. Gate/up: two-sided rounding (`ldlq_2hess`) with G = diag(Σ w·c²)^0.5 (thread 06). Down: plain H.
3. Base: fitted with blended feedback between the 2-bit and 4-bit targets (dual per-precision feedback, thread 11). GLM λ = 0.3 (thread 02: E36 L2 34.59 / L4 9.35 vs EXL3 34.42 / 8.89, NVFP4 12.29). MiMo uses seq (λ = 0, native 2-bit base).
4. Residual P4: LDLQ with its own feedback against the 4-bit target. Production (thread 12) fits base and P4 in one joint pass (inner 0); the base bytes are shared by both levels of the artifact but depend on the residual rate.
5. Optional encoder-side: tune su/sv magnitudes against the SwiGLU output with held-out early stopping; MiMo 2 bit may add a per-projection output bias (0.0043 bpw) (thread 09).
6. Boundary weighting: NOT used for calibration (user decision 2026-09-29). Production H = thread-08 recipe with boundary weight 1. Thread 12 A/B (6 experts L16/49/66, old-corpus chunk 0, p4126): flat 50x on the 32 tokens before </think> / <|im_end|> gives geomean nq/L4 end d<=32 forced -3.2%, but think d<=32 +2.4%, val all forced +3.0%, val routed +7.0%, matched routed +6.8%. The increment is 37-43% of tr(H) and is dominated by end + context rows. ESS shrink (k 30-300) changes nothing; a 20% trace cap gives end -2.5% for all +0.7% / routed +2.2%. nq vs EXL3-4 on the same H stays about -4% in every arm. The --bnd/--bnd-k/--bnd-cap code stays in threads/12-reference-encoder (nq_bnd.py), off by default. Boundary weights (REAP 50/20/5/2) only choose the fixed always-4-bit set (fixed_set.json).

## Level-4 gap (thread 02)
A nested 4.0 bpw level 4 sits ~2.8% above native at best (code-structure floor), so it cannot beat EXL3-4 on GLM at exactly 4.0 bpw. User requirement is one dynamic 2-4 bit artifact, so a separate native 4-bit code is ruled out. Thread 14 (E36): a pattern-rate residual (EXL3-style frac trellis, where step i shifts in KA + bit((-i) mod 16) of MASK bits; bit(i mod 16) is wrong) with per-projection K of 1.875 on gate/up and 2.25 on down moves L4 routed/forced/OOD from 9.35/9.67/10.22 to 9.14/9.53/10.12 at the same bytes (EXL3-4: 8.89/9.17/9.71). Interleaved half-bit blocks lose to uniform because rate-mixing costs a Jensen penalty. Update (T14, E36 on T12's frozen stack): g/u 1.9375 (KA1, MASK 0xFFFE) + down 2.3125 (KA2, 0x9248) at 4.0846 bpw gives 8.801/9.107/9.637, below EXL3-4 8.897/9.172/9.699; E36 only. On all 9 experts the 1.875/2.25 split at 4.0221 bpw gives routed/forced/OOD geomean 0.985/0.9985/1.0045 vs T12's uniform residual and still trails EXL3-4 by 3.7/4.7/5.2%; it helps L16 but not L49/L66, so the split must be chosen per expert (threads/14-level4-floor/REPORT.md). T12's frozen base is +3.2% vs EXL3-2 on 16:165. Gap closers under test in thread 12: 3-bit residual on top-benefit blocks at +0.0625/+0.125/+0.25 bpw, per-tile seed/codebook choice, per-block δ.
ADOPTED production Level 4 (thread 12, 9 experts, lead 2026-09-28): pattern residual gate/up K2 (uniform), down 2.3125 (KA2, kernel MASK 0x9248 = EXL3 step mask 0x2492) = **4.1263 bpw**, encoded in ONE joint pass (base and P4 fitted together, inner 0; no rate-canonical K2-reference pass). Vs EXL3-4 mean/worst %: routed −3.17/−1.43, OOD forced −2.33/−1.75, OOD routed −2.32/−0.83 (every expert below EXL3-4 on every split). The same pattern with the 2-pass canonical base + inner 2 gives −3.36/−0.22, −2.35/−1.59, −1.86/+1.67. 4.0846 (g/u 1.9375) fails the rule (OOD forced mean +0.76–0.84, worst OOD routed +2.0–2.3). Positional 4.15625 gives −2.42/+0.03, −2.84/−2.24, −2.21/+0.98. Encode is 23–50 s/expert on a shared A100 with the CUDA pattern Viterbi (threads/12-reference-encoder/csrc/nq_fracvit.cu = exllamav3 quantize_tiles_frac_kernel<KA, step mask>). The kernel needs residual K codes 2 and 2.3125. L2 under single pass: routed −0.3 mean / +4.0 worst vs EXL3-2 (16:165); per-expert λ fallback A/B pending (user decision). Entry point: threads/12-reference-encoder/nq_layer.py (manifest: default_allocation = T19 fixed_set.json, top 26 per layer by boundary-weighted REAP 50/20/5/2/1).

## Kernel
Update (thread 15, REPORT.md + ref15_spec.py): adopted decoder is the int-fold RM_P with greedy funnels, V1 residual, per-projection fractional residual K, 4-lane rings (256 weights): 4b 38.8-43.8 µs B1-B4 (7.15 ops/weight), 2b 26.4-30.7 µs (3.91). Q4 = fp16(A'(1024+F)+C), F = (Mb·S(sb)+N·S(sr)+128)>>8, δ = N/Mb; encoders must emit this exactly. Per-unit K3 residual units need a kernel branch (not built). MoE layer kernel (thread 13, RM_P port done): full I=2048 2b 1.70-1.87x and 4b 1.05-1.16x vs EXL3 at B1-B4; TP8 shard 2b 2.04-2.35x and 4b 1.38-1.67x vs EXL3's best KSPLIT. Bit-exact vs ref15_spec. The kernel must be compiled with every residual K code the checkpoint uses (NQ_RK_CODES). No dense prefill kernel yet.
Adopted (thread 04 follow-up, REPORT2.md): nqk2.cu A4 additive decode, rings shared by 2 or 4 lanes (128/256 weights), one fp16 δ per 16x128 block folded into 2 HFMA2, fused-B 2 launches. 4b 42.5-46.8 µs B1-B4 vs EXL3 bare 53.9-74.9; 2b 28.7-31.3 vs 60.2-72.5. Level 3 dropped.
Custom tensor-core GEMV (mma.m16n8k16), each lane decodes straight into fragments, split planes, per-expert level and pointer tables read at CUDA-graph replay, one launch for mixed levels. Budget: 10 or fewer ops/weight at 4 bit, 8 or fewer at 2 bit. Fuse Hadamard/SwiGLU into the GEMVs.

## Streaming
Thread 10: static 4-bit set by benefit per byte; recency upgrades at 1–4 streams with one-step lag; next-layer top-12–16 prefetch under TP; downgrade instantly under KV pressure; per-step byte cap at link rate.

## Sources (after the 2026-09-28 /tmp wipe)
- GLM FP8: /tmp/nestquant/glm53-fp8-experts (9 experts, per-expert files) now; full model downloading to /tmp/nestquant/src/glm53-fp8 (ETA ~19:55 AEST).
- MiMo MXFP4: /tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source, being re-downloaded by another agent (read only, ETA ~19:30 AEST).
