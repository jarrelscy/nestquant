# Thread 04: decode kernel speed (written by the lead from the agent's final message)

Code and results in this directory: nqk.cu, build.py, nq.py (kernel + bindings); bench_exl3.py, bench_chain.py, syn.py, tune.py, timing.py; exl3_bench.json, chain.json, chain_j.json, syn.json, tune.json. Scratch/logs: /tmp/nestquant/04-decode-kernel/.

**Lead's correction (from thread 03):** J3/J4 is not a valid progressive code. With a frozen native base, J4 level 4 (.0868 MSE) is worse than level 2 alone; fitted jointly, level 4 reaches native only when level 2 collapses. The decode must be additive, f4 = f2(base) + δ·g(planes), so the valid kernels are T2R2 (58.8/62.6 µs B1/B4) and T2H (53.9/57.2). Per-lane tail-biting rings also cost quality: 0.29/0.15 dB at 64 weights and 1.34/0.90 dB at 32 (2b/4b), vs 0.06/0.03 dB at 128. Use rings of ≥128 weights. The J timings still bound what a single-hash decode can reach.

## Question
Which decode structure and kernel design lets a progressive 2/4-bit format beat EXL3 latency for one expert (gate+up [2048x6144], SwiGLU, down [6144x2048]) on A100 at batch 1–4, and what is the largest decode ops/weight budget that still wins?

## Method
- EXL3 baseline: installed exllamav3 1.5.1 (mul1) on the GLM L16 E36 2b/4b files via `orbit_duet/exl3_adapter.py`. "Adapter" = full forward (3 GEMVs + ~8 torch glue kernels). "Kernels" = the 3 GEMVs + silu only.
- Custom kernel `nqk.cu`: tensor-core mma.m16n8k16 (fp16 in, fp32 acc, batch in N). Each lane decodes 64 weights per (16-row strip, 128-col chunk) straight into fragments, using its own tail-biting trellis (no warp shuffles). Split-K with fp32 atomics, 2-deep register pipeline.
- Full chain, 5 launches: input Hadamard → fused gate+up (N=4096) → Hadamard/SwiGLU/Hadamard → down → output Hadamard.
- Decoders: UNI (uniform, speed ceiling), T (mul1), H/H2B (QTIP HYB V=2, plain and bank-replicated LUT), T2R1/T2R2 (base trellis + 1/2 additive refinement trellises), T2H (base + HYB refinement), J3/J4 (combined-window mul1: higher level re-reads base bits together with its own plane bits), SYN (mul1 + R synthetic ops/weight).
- Timing: one CUDA graph per candidate, 6 expert copies, cold L2 per replay, shuffled order in one process, 40–150 blocks, medians. GPU 3 shared with an intermittent ~44 GB process, so absolute numbers ±5–10%; comparisons are paired.
- All decoders match a dense decode (rel err ~3e-7). Random weights (latency-neutral).

## EXL3 breakdown

| | 4b | 2b |
|---|---|---|
| Adapter, B1 | 60–67 µs | 63–73 µs |
| Kernels, B1 | 48–54 µs | 51–60 µs |
| Adapter, B4 | ~88 µs | ~86 µs |
| Kernels, B4 | ~74 µs | ~72 µs |

~13 µs is torch glue. Effective throughput ~390 GB/s (4b) and ~185 GB/s (2b) against ~1.5 TB/s practical, so EXL3 is limited by latency and instruction issue; 2b is no faster than 4b. Bandwidth floor ~12 µs (4b), ~6 µs (2b), plus ~2.5 µs per launch.

## Full chain latency (median µs per expert, cold L2)

| Decoder | ~ops/weight | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| **2b** | | | | | |
| EXL3 adapter | – | 72.9 | 91.7 | 85.9 | 86.7 |
| EXL3 kernels | – | 60.1 | 77.8 | 71.5 | 72.6 |
| UNI2 | 1.6 | 26.4 | 27.9 | 28.9 | 29.7 |
| T2 mul1 | 4.1 | 33.0 | 34.5 | 35.2 | 36.2 |
| H2 HYB | 2.9 + bank conflicts | 35.9 | 38.7 | 40.4 | 43.1 |
| H2B | 3.4 | 37.0 | 39.5 | 41.3 | 43.7 |
| **3b** | | | | | |
| T2R1 | 10.2 | 48.0 | 49.9 | 50.6 | 51.4 |
| J3 | ~7 (est.) | 42.1 | 44.6 | 46.2 | 46.8 |
| **4b** | | | | | |
| EXL3 adapter | – | 66.7 | 78.4 | 89.3 | 88.7 |
| EXL3 kernels | – | 53.5 | 65.0 | 75.5 | 74.1 |
| UNI4 | 1.75 | 30.4 | 32.5 | 33.8 | 34.7 |
| T4 mul1 (not progressive) | 4.6 | 36.4 | 38.0 | 39.2 | 39.9 |
| H4 HYB | 2.9 + bank conflicts | 38.7 | 41.9 | 44.2 | 47.0 |
| T2R2 split | 13 | 58.8 | 61.1 | 62.0 | 62.6 |
| T2R2 interleaved | 13 | 58.3 | 59.6 | 60.6 | 61.1 |
| T2H split | 10 | 53.9 | 55.4 | 56.4 | 57.2 |
| T2H interleaved | 10 | 56.3 | 58.0 | 58.2 | 59.0 |
| **J4 split** | ~9 (est.) | **49.6** | **51.2** | **52.1** | **52.6** |
| J4 interleaved | ~9 (est.) | 47.5 | 49.0 | 51.3 | 51.7 |

Ops/weight from cuobjdump SASS diffs against UNI4 (= 1.75); ncu is not permitted on this box. The 3 Hadamard/SwiGLU launches cost 7.5–8.5 µs; fusion can recover this.

## Decode budget sweep (mul1 + R extra ops/weight, chain µs)

| | R0 | R2 | R4 | R8 | R12 | R16 | EXL3 adapter |
|---|---|---|---|---|---|---|---|
| B1, 4b | 36.3 | 43.6 | 49.3 | 59.0 | 68.5 | 83.5 | 67.0 |
| B4, 4b | 40.1 | 47.2 | 51.8 | 61.2 | 70.3 | 82.6 | 89.0 |
| B1, 2b | 33.0 | 41.2 | 46.5 | 56.0 | 66.5 | 77.4 | 73.1 |
| B4, 2b | 36.1 | 44.3 | 49.7 | 59.3 | 69.6 | 80.5 | 86.8 |

~2.4–3 µs per extra op/weight. Break-even total ops/weight: 4b B1 ~15–16 vs adapter, ~11–12 vs bare kernels; 4b B4 ~25; 2b B1 ~18; 2b B4 ~22. **Design budget: ≤10 ops/weight at 4b, ≤8 at 2b** (≥20% margin at B1).

## Plane layout (for thread 10)
- Split planes (base uint4 = 2 bit, P3 uint2, P4 uint2 per strip/chunk/lane) cost 0–4% vs interleaved (J4 +4% B1, +1.6% B4; T2H split was faster).
- Tiling, launch shape and output layout are identical at 2/3/4 bit; the level is a decoder switch, so a per-expert level/pointer table read at graph replay is a direct extension (not implemented).
- Gate/up tile by 16-row output strips, down by 128-col input chunks; both divide the 256-wide TP8 shards.

## Verdict
- **Adopt:** J3/J4 combined-window mul1 with split planes in a custom tensor-core kernel. 2b base 33–36 µs (2.2–2.4x faster than EXL3 adapter), 3b 42–47 µs, 4b 49.6–52.6 µs (vs EXL3 adapter 66.7–88.7, bare kernels 53.5–74.1).
- **Reject:** HYB V=2 on A100 (bank conflicts, slower than mul1); additive refinement at 4b (T2R2 loses to EXL3 bare kernels at B1, 58.8 vs 53.5; T2H roughly ties).
- **Needs more:** quality of the per-lane 32–64-weight tail-biting trellis and of J-decoding (base bits reused in the 4-bit window) vs EXL3's 256-weight tile trellis. Fusion.

## Recommendation
1. Progressive format decodes one mul1 hash per weight per level from a combined window, split planes, ≤10 ops/weight at 4b and ≤8 at 2b.
2. Speed work by payoff: the custom tensor-core kernel (1.8–2.2x alone); fuse Hadamard/SwiGLU into the GEMVs with arrival counters (5–8 µs); single graph-replayed launch driven by per-expert level/pointer tables.
3. Pick bits for quality and memory. Decode is issue-bound, so 2b is only ~3–4 µs faster than 4b.
