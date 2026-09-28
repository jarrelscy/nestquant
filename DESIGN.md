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
4. Residual P4: LDLQ with its own feedback against the 4-bit target, given the frozen base.
5. Optional encoder-side: tune su/sv magnitudes against the SwiGLU output with held-out early stopping; MiMo 2 bit may add a per-projection output bias (0.0043 bpw) (thread 09).

## Level-4 gap (thread 02)
A nested 4.0 bpw level 4 sits ~2.8% above native at best (code-structure floor), so it cannot beat EXL3-4 on GLM at exactly 4.0 bpw. User requirement is one dynamic 2-4 bit artifact, so a separate native 4-bit code is ruled out. Thread 14 (E36): a pattern-rate residual (EXL3-style frac trellis, where step i shifts in KA + bit(i mod 16) of MASK bits) with per-projection K of 1.875 on gate/up and 2.25 on down moves L4 routed/forced/OOD from 9.35/9.67/10.22 to 9.14/9.53/10.12 at the same bytes (EXL3-4: 8.89/9.17/9.71). Interleaved half-bit blocks lose to uniform because rate-mixing costs a Jensen penalty. 9-expert confirmation is running and should stack with extra P4 rate. Gap closers under test in thread 12: 3-bit residual on top-benefit blocks at +0.0625/+0.125/+0.25 bpw, per-tile seed/codebook choice, per-block δ.

## Kernel
Update (thread 15, REPORT.md + ref15_spec.py): adopted decoder is the int-fold RM_P with greedy funnels, V1 residual, per-projection fractional residual K, 4-lane rings (256 weights): 4b 38.8-43.8 µs B1-B4 (7.15 ops/weight), 2b 26.4-30.7 µs (3.91). Q4 = fp16(A'(1024+F)+C), F = (Mb·S(sb)+N·S(sr)+128)>>8, δ = N/Mb; encoders must emit this exactly. Per-unit K3 residual units need a kernel branch (not built). MoE layer kernel (thread 13): TP8 shard 1.36-2.90x faster than EXL3 exl3_moe_coop at B1-B4; being ported to RM_P.
Adopted (thread 04 follow-up, REPORT2.md): nqk2.cu A4 additive decode, rings shared by 2 or 4 lanes (128/256 weights), one fp16 δ per 16x128 block folded into 2 HFMA2, fused-B 2 launches. 4b 42.5-46.8 µs B1-B4 vs EXL3 bare 53.9-74.9; 2b 28.7-31.3 vs 60.2-72.5. Level 3 dropped.
Custom tensor-core GEMV (mma.m16n8k16), each lane decodes straight into fragments, split planes, per-expert level and pointer tables read at CUDA-graph replay, one launch for mixed levels. Budget: 10 or fewer ops/weight at 4 bit, 8 or fewer at 2 bit. Fuse Hadamard/SwiGLU into the GEMVs.

## Streaming
Thread 10: static 4-bit set by benefit per byte; recency upgrades at 1–4 streams with one-step lag; next-layer top-12–16 prefetch under TP; downgrade instantly under KV pressure; per-step byte cap at link rate.

## Sources (after the 2026-09-28 /tmp wipe)
- GLM FP8: /tmp/nestquant/glm53-fp8-experts (9 experts, per-expert files) now; full model downloading to /tmp/nestquant/src/glm53-fp8 (ETA ~19:55 AEST).
- MiMo MXFP4: /tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source, being re-downloaded by another agent (read only, ETA ~19:30 AEST).
