# Thread 04 follow-up: additive progressive decode kernel (written by the lead from the agent's final message)

Kernel nqk2.cu / nq2.py / build2.py; checks check3.py, check_fused.py; tuning tune2.py, tune_fused.py; bench bench_chain2.py; results tune2.json, tune_fused.json, chain2.json (GPU 3), chain2_gpu7.json (final, idle GPU 7). Builds in /tmp/nestquant/04-decode-kernel/build2.

## Design
- Tensor-core mma batch GEMV, 16-row strip x 128-k chunk, 64 weights per lane, split planes: base uint4 (2b), P3 uint2 (1b residual), P4 uint4 (2b residual).
- Tail-biting ring shared by G lanes (G=2: 128 weights, G=4: 256), neighbour's first word via one __shfl_sync per plane per chunk. No cost vs 64-weight rings.
- Additive decode with δ folded: base = HFMA2(h_b, A, B·(1+δ)); out = HFMA2(h_r, δ·A, base). One fp16 δ per 16x128 block.
- Fused-B: gate+up does input WHT-128 in prologue; down's prologue recomputes SwiGLU with Hadamards from the gate/up accumulator; down zeroes the ping-pong accumulator and does the output Hadamard. 2 launches.
- Correct vs dense decode, numpy ring reference and torch chain (2e-4 rel).

## Full chain, median µs, GPU 7, random weights

| Level | Decoder | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| 4b | EXL3 adapter | 67.3 | 78.3 | 88.9 | 89.7 |
| 4b | EXL3 bare kernels | 53.9 | 64.8 | 73.8 | 74.9 |
| 4b | old T2R2 | 58.8 | 60.9 | 62.1 | 62.6 |
| 4b | A4 G=2 unfused | 46.8 | 48.5 | 49.8 | 50.6 |
| 4b | A4 G=2 fused-A | 45.0 | 46.3 | 47.7 | 48.2 |
| 4b | **A4 G=2 fused-B** | **42.5** | **45.2** | **45.7** | **46.8** |
| 4b | A4 G=4 fused-B | 42.4 | 44.9 | 45.7 | 47.0 |
| 4b | T4 single-stream (not progressive) | 29.9 | 31.9 | 33.9 | 35.2 |
| 3b | A3 base+P3 fused-B | 42.3 | 44.0 | 44.9 | 46.0 |
| 3b | MIX 50% of 128x128 blocks at 4b | 40.2 | 42.3 | 42.7 | 43.7 |
| 2b | EXL3 adapter | 73.5 | 91.7 | 86.5 | 86.2 |
| 2b | EXL3 bare kernels | 60.2 | 77.8 | 72.3 | 72.5 |
| 2b | **B2 G=2 fused-A** | **28.7** | **29.5** | **31.1** | **31.3** |
| 2b | B2 G=2 fused-B | 28.8 | 30.4 | 31.4 | 32.9 |

Marginal ops/weight: B2 5.4, T4 5.1, A4 10.3 (2 IMAD hash, 2 IDP, 3.5 SHF/LOP, 1.1 PRMT, 1.0 HFMA2), A3 10.5, old T2R2 ~13.

## Findings
- Fusion saves 3-4.5 µs, less than the 7.5 µs launch count suggested (serialised last-block tail, prologue work).
- Level 3 option (a) costs as much as level 4 (issue-bound, two hashes). Option (b) mix is fastest but only at 128x128 thread-block granularity; per-strip mixing costs as much as A4.
- Ring length 128 vs 256 is free for speed.
- Remaining floor ~10 ops/weight = two mul1 hashes. Next cut needs a codebook change (e.g. residual byte-sum folded into the base dp4a with power-of-two δ ratio, ~1.6 ops/weight saved).

## Verdict
Adopt A4 additive, split planes, G=2 or 4, fused-B: 21-37% faster than EXL3 bare kernels at 4b B1-B4, 37-49% faster than the adapter. 2b: B2 fused-A, 2.1-2.5x faster than EXL3 bare kernels. Level 3 = option (b) at 128x128 block granularity. Reject T2R2 and per-strip mixing. Still to do: single graph-replayed launch with per-expert level/pointer tables.
