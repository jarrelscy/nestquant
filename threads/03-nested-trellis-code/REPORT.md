# Thread 03: nested (successively refinable) trellis code, 2 → 3 → 4 bpw

## Question
Which successively refinable code on i.i.d. Gaussian data (a stand-in for rotated expert weights) gets closest to the native EXL3 bitshift trellis at 2, 3 and 4 bpw while keeping decode cheap? Target: level 2 equal to native 2-bit, level 4 within 0.1–0.2 dB of native 4-bit, and 4-bit decode ops no higher than EXL3-4.

## Method
- Source: N(0,1), 256-weight tail-biting tiles (512–2048 tiles, 131k–524k weights). "dB vs RD" is measured against D(R)=2^-2R.
- Native baseline: the exllamav3 1.5.1 CUDA Viterbi (`quantize_tiles`), L=16, with a scale search. It uses the default mul1 codebook, and mcg and 3inst were also measured. My own torch Viterbi (`trellis.py`: generic L/k/V, per-position context, two-pass tail-biting with wrap warm-up) matches the kernel to within 0.1% at K=2 (0.06828 vs 0.06820).
- Refinement stages are Viterbi-fitted on the frozen lower level's residual, with a residual scale δ per stage (per 16-row block in the real pipeline).
- The joint product-state trellis (Jafarkhani–Tarokh style) is in `joint.py`: level-2 value f2(b16) plus level-4 value f2+δ·g(b16‖r8). It is a joint Viterbi over 2^20 states with 16 branches and cost αD2+βD4.
- Decode ops come from real SASS (`exp5*.py` (ring length, J-style), `sass/decode.cu`, nvcc 13 sm_80, cuobjdump). Each count is ALU+LDS instructions per weight for 32–64 weights per thread, including the activation HFMA2 and address math. Global/shared loads of the packed words are excluded.
- Real-data check: the thread-05 harness, full EXL3 pipeline (rotation, LDLQ, scale refit), GLM L16 E36, evaluated on the matched capture. For the nested fit, the level-2 base tiles are recorded block by block and replayed at level 3/4 with the same g_scale, and only the residual is quantized under that level's LDLQ feedback. Level 2 is therefore bit-identical to EXL3-2.

## i.i.d. Gaussian results
MSE; dB above the RD bound; Δ = dB worse than native mul1 at the same rate.

| code | L2 | L3 | L4 | 4-bit decode ops/w (SASS) | LUT / smem | fit cost |
|---|---|---|---|---|---|---|
| RD bound | .06250 | .01563 | .003906 | | | |
| Lloyd-Max scalar | .1175 (+2.74) | .03454 (+3.45) | .009497 (+3.86) | ~2 | 16 halfs | trivial |
| **native EXL3 mul1 L16** | .06843 (+0.39) | .01740 (+0.47) | .004453 (+0.57) | **5.56** (2-bit: 5.81) | none | 0.15 s/Mw (kernel) |
| native mcg / 3inst | .06862 / .06903 | .01744 / .01780 | .004541 / .004624 | | | |
| A. mul1 K2 + mul1 K2 residual (L16) | = native | – | .004709 (+0.81, Δ0.24) | 11.31 (2.03x) | none | 2×0.15 s/Mw |
| A'. same, residual L=20 | = native | – | .004678 (+0.78, Δ0.21) | ≈11.3 | none | 190 s/Mw (torch) |
| A-ctx. residual hash XOR base-window hash | = native | – | .004725 (Δ0.25) | +1 | none | |
| A-LUT. learned direct LUT residual L=12/10/8 | = native | – | .004986 / .005274 / .005535 (Δ0.49/0.73/0.95) | ~8 | 8 KB / 2 KB / 512 B | Lloyd 8 it |
| B. mul1 + K1 + K1 (3 planes) | = native | .018649 (Δ0.30) | .005161 (Δ0.64) | 16.7 (3.0x) | none | 3×kernel |
| B-ctx. K1 stages with context-hashed code | = native | .018362 (Δ0.23) | .004934 (Δ0.45) | ~17.5 | none | torch 20 s/Mw |
| C. HYB V=2 Q9 base + HYB V=2 Q9 residual | .06916 (+0.44, Δ0.05) | – | .004799 (+0.89, Δ0.32) | **7.95 (1.43x)**; 2-bit **4.05 (0.70x)** | 2 KB per stage (512×half2) | torch ~3 s/Mw + Lloyd |
| D. mul1 base + HYB residual | = native | – | .004776 (Δ0.30) | 9.78 | 2 KB | |
| V=2 direct-LUT residual L12 | = native | – | .005129 (Δ0.61) | ~7 | 16 KB | |
| E. joint product trellis, 8-bit ref window (8 tiles, noisy) | α=1,β=16: .0732 (+0.69) | – | .00493 (+1.01) | ~10 | 512 B | 5400 s/Mw |
| E, β only | .0943 (+1.79) | – | .00452 (+0.63) | | | |
| F. alternating αD2+βD4 base refit (β/α = 1–4) | .06843 | – | .004707 (no change) | | | |

Findings:
1. On i.i.d. Gaussian each greedy trellis stage costs about one trellis loss (0.39–0.44 dB). Level 4 over two stages loses 0.81 dB, against 0.57 dB for native 4-bit, so the floor is Δ≈0.24 dB (0.21 dB with an L=20 refinement). This is much better than the ~2 dB the literature reports for small-state multistage TCQ, but still just outside the 0.1–0.2 dB target.
2. The base residual has kurtosis 3.03, autocorrelation below 0.002 at lags 1–8, and a conditional mean of about 0. Its variance moves by at most ±11% across q2 buckets. Context-conditioned refinement (hashing in base bits, context scales) has nothing to exploit, and the joint and alternating objectives are stuck at the greedy fixed point. This is Equitz–Cover in practice: for a Gaussian source the greedy cascade is essentially the right structure.
3. The JT-style joint product trellis reaches native-4 quality (.00452) only by giving up level 2 (+1.4 dB). Its tradeoff curve lies entirely above A. Rejected.
4. The third level is expensive. Splitting the 2-bit refinement into two 1-bit planes (B) costs another 0.2–0.4 dB at level 4 (Δ0.45 with context hashing, which helps K=1 codes by about 0.07 dB).
5. Decode: every nested code needs two independent code evaluations at level 4. No candidate reaches ≤ EXL3-4 ops. The best is HYB+HYB at 1.43x ALU, plus random-index LDS (bank conflicts). Its HYB base is 0.05 dB worse than native 2-bit, but its 2-bit decode is 0.70x of EXL3-2. mul1+mul1 costs exactly 2x: the δ scale folds into the second HFMA2, so there is no extra add. HYB detail: the index must come from the middle bits of w·w+w. The top bits gave .0778 at L2.

## Real pipeline (GLM L16 E36, router-weighted relative output L2 %, routed / forced; lower is better)

| method | all routed | all forced | ood routed |
|---|---|---|---|
| EXL3-2 (= nested L2) | 37.41 | 40.01 | 42.76 |
| EXL3-3 | 19.16 | 20.54 | 22.10 |
| EXL3-4 | **9.74** | 10.46 | 11.18 |
| NVFP4 (brief) | 12.29 | 12.78 | |
| nested 2+2 L4, base = EXL3-2 (LDLQ) | 12.10 | 12.98 | 13.79 |
| nested 2+1 L3 | 23.17 | 24.84 | 26.37 |
| nested 2+1+1 L4 | 14.29 | 15.30 | 16.20 |
| control: 2+2 refit per tile under L4 feedback (not nested) | 10.00 | 10.72 | 11.53 |
| base without LDLQ feedback, L2 | 38.88 | 40.07 | 40.24 |
| **nested 2+2 L4 on the no-LDLQ base** | **9.99** | 10.72 | 11.50 |

Proxies agree: EXL3-4 .00070/.00072/.00116, nested on the LDLQ base .00133/.00138/.00193, nested on the no-LDLQ base .00077/.00080/.00127.

The real-data gap splits into two parts:
- Code structure: 9.74 → 10.00 (+2.7% relative, ≈0.23 dB). This matches the i.i.d. Δ0.24.
- Feedback conflict: 10.0 → 12.1. It comes entirely from freezing a base that was fitted with 2-bit LDLQ feedback. The residual std grows from 0.26 to 0.31.

Fitting the base without error feedback removes the conflict (nested L4 9.99, below NVFP4 12.29). It costs +1.5 pp at L2 routed, +0.06 pp forced, and is better on OOD routed (40.2 vs 42.8). This is H2 confirmed at full strength (for thread 02).

## Verdict and recommendation
- **Adopt (4-bit endpoint): base = native EXL3 mul1 K=2 L16, plus one K=2 L16 mul1 residual plane with a per-block δ, fitted greedily with the base frozen.** Level 2 decodes with the unchanged EXL3 kernel. On i.i.d. data level 4 is 0.24 dB off native 4-bit. On GLM it is 10.0 vs 9.74 routed, provided the base is not fitted with full 2-bit LDLQ feedback. That setting is thread 02's lever: blend or reduce the base's feedback, and one no-feedback point is already at 9.99/38.9.
- The code structure (K2 residual, L=16 or L=20) is not the bottleneck. L=20 buys 0.03 dB at 1000x the fitting cost. Learned LUTs, context hashing and joint/JT product trellises do not help on Gaussian residuals.
- **Level 3:** a 1-bit plane (K1 with a context-hashed code) gives a good L3 (Δ0.23 dB i.i.d.). If the 4-bit level must then contain it, L4 drops to Δ0.45 i.i.d. and 14.3% on GLM (LDLQ base). Better options:
  - (a) store P3 (1 bpw) and P4 (2 bpw) as separate planes. L4 reads base+P4 only and L3 reads base+P3 only, so VRAM is exact at each level but host/disk holds 5 bpw and the 3→4 upgrade fetches 2 bpw.
  - (b) make level 3 a per-block rate mix of the 2- and 4-bit planes, driven by the LDL-importance allocation (H1). On i.i.d. data a uniform half/half mix is poor (MSE (D2+D4)/2).
  Flag for thread 10: both (a) and (b) keep every plane inside a 256-tile shard with constant bytes per (layer, plane, shard). Option (a) violates "each level reads a prefix of the planes" only in that L4 skips P3.
- **Speed:** no nested code meets ≤ EXL3-4 decode ops. mul1+mul1 is 2.03x EXL3-4 ALU ops per weight. HYB(V=2, 512×half2 LUT)+HYB is 1.43x at 4 bit, and its 2-bit level is 0.70x of EXL3-2 at a 0.05 dB MSE cost; it is the only candidate with a decode-speed argument at both rates. Whether ALU ops or bandwidth/latency dominate at B1–4 has to be settled by the kernel thread (H4).

## Follow-up for thread 04 (kernel constraints)
**(a) Tail-biting ring length** (native mul1 L16, i.i.d.; my two-pass wrap-warm-up tail-biting, which is within 0.1% of the kernel at 256):

| ring length | K=2 MSE (dB vs RD, Δ vs 256) | K=4 MSE (dB vs RD, Δ vs 256) |
|---|---|---|
| 256 (EXL3 tile) | .06833 (+0.39) | .004456 (+0.57) |
| 128 | .06924 (+0.44, Δ0.06) | .004484 (+0.60, Δ0.03) |
| 64 | .07306 (+0.68, Δ0.29) | .004616 (+0.73, Δ0.15) |
| 32 | .09300 (+1.73, Δ1.34) | .005482 (+1.47, Δ0.90) |

At 64 weights per lane the base loses 0.29 dB, about as much as the whole nested-structure loss. 32 is unusable. Use rings of at least 128 weights; 128 costs 0.03–0.06 dB. My heuristic could be slightly pessimistic for very short rings, but the trend is structural: an 8-step state memory against a short ring.

**(b) J-style combined-window decode** (one mul1 hash of 8b base | 4b P3 | 4b P4, no additive base term):
- Base frozen at native 2-bit (so L2 = native): L4 MSE is .0868, worse than L2 alone (.0682). J3 (8b base | 8b P3) gives .295. The hash does not cluster around the level-2 value, so the refinement bits only select among unrelated values.
- Fitted jointly with the product-state Viterbi, αD2+βD4, 8 tiles:

| α | β | D2 | D4 |
|---|---|---|---|
| 1 | 1 | .0825 (+1.2 dB) | .0591 (+11.8 dB) |
| 1 | 4 | .135 | .0348 |
| 0 | 1 | 2.16 | .00444 (= native 4) |

- Verdict: **J3/J4 as specified cannot be a progressive code.** Each level only works if the other level is sacrificed. A progressive decode needs the level-k value to contain the lower level's value additively: f4 = f2(base16) + δ·g(P window), with g optionally also hashing base bits (no quality gain). The fast J kernels therefore time an invalid code. The valid options are T2R2 (mul1+mul1, 13 ops, 58.8 µs B1 / 62.6 B4) and T2H (mul1 + HYB refinement, 10 ops, 53.9/57.2). i.i.d. quality: T2R2 Δ0.24 dB, T2H Δ0.30 dB vs native 4-bit. To get back toward J4 speed, look at a cheaper g (for example HYB with bank-replicated LUT, or a mul1 variant whose FMA absorbs the base add; the add costs nothing because δ folds into the second HFMA2).

## Files
`trellis.py` (Viterbi), `exp1.py`/`exp2_joint.py`/`exp3.py`/`exp4*.py` (i.i.d. experiments, `*.log`/`*.json`), `joint.py` (product-state JT-style trellis), `nested_harness.py`/`nested_controls.py` (GLM L16 E36, JSON results), `exp5*.py` (ring length, J-style), `sass/decode.cu` + `sass/count.sh` + `sass/sass_counts.json` (instruction counts), `run.sh` (GPU 2 env incl. libcudart.so.12 path).
