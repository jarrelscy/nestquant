# Mac performance measurements

## KL divergence on generated traces

The [10 November measurements](results/2026-11-10-kld-traces.json) score the reference runtime with teacher-forced KL divergence. Layer-major prefill measured **0.0342** against the BF16 teacher, under the 0.05 goal. Streamed decode fell from 0.155 to 0.066 as the hot allocation rose from 2 to 46 four-bit experts per layer. The 23, 35 and 46 expert arms measured 0.080, 0.072 and 0.066, inside or below the 0.07 to 0.09 decode goal, but with only part of the base resident. One full-resident-base arm completed after a restart, with a 1 GiB refinement cache: zero synchronous base reads, 2.3 to 3.9 decode tokens/s and an 82.75 GiB peak. Larger refinement caches with a resident base did not fit.

| Arm | Four-bit experts per layer | Mean KLD | 95% interval | Top-1 agreement | Decode tokens/s | Peak footprint |
|---|---:|---:|---:|---:|---:|---:|
| Prefill against BF16, 8,188 positions | 288 | 0.0342 | 0.027 to 0.043 | 94.1% | not applicable | 65.6 GiB |
| Decode, all experts at 4 bits | 288 | 0.0061 | 0.005 to 0.008 | 98.5% | 0.30 to 0.37 | 59.9 GiB |
| Decode, hot fraction 2/288 | 2 | 0.1548 | 0.120 to 0.194 | 90.3% | 1.36 to 2.16 | 60.0 GiB |
| Decode, asynchronous 1% | 2.9 | 0.1460 | 0.110 to 0.189 | 90.3% | 1.55 to 2.16 | 61.7 GiB |
| Decode, asynchronous 2% | 5.8 | 0.1260 | 0.094 to 0.161 | 90.2% | 1.54 to 2.19 | 61.7 GiB |
| Decode, asynchronous 4% | 11.5 | 0.1017 | 0.077 to 0.129 | 92.2% | 1.39 to 2.06 | 61.7 GiB |
| Decode, 8 GiB cache, 20 GiB base cache | 23.0 | 0.0802 | 0.059 to 0.103 | 93.2% | 0.86 to 1.11 | 53.2 GiB |
| Decode, 12 GiB cache, 16 GiB base cache | 34.6 | 0.0716 | 0.051 to 0.095 | 93.2% | 0.58 to 0.82 | 53.0 GiB |
| Decode, 16 GiB cache, 12 GiB base cache | 46.1 | 0.0656 | 0.049 to 0.085 | 93.1% | 0.38 to 0.73 | 52.9 GiB |
| Decode, complete resident base, 1 GiB cache | 2.9 | 0.1477 | 0.111 to 0.189 | 90.4% | 2.28 to 3.90 | 82.8 GiB |

Prefill windows 0000 to 0003 measured 0.0275, 0.0487, 0.0272 and 0.0333. Each is 0.006 to 0.012 above the model card's value for the same window. The cause is not identified.

Decode arms score the first 160 generated tokens of eight greedy GLM-5.3-Flash Q4_K traces, 1,280 positions per arm. Their reference is the all-4-bit layer-major prefill of the same trace, a proxy for the BF16 model. The all-4-bit decode arm is the noise floor of that comparison. Its 0.0061 shows that decode and prefill agree when precision is equal.

The decode mean depends on a small tail. The median position is at or below 0.005 in every arm, and the 95th percentile is 0.48 to 0.75 in the mixed arms. Code and arithmetic traces measured 0.02 to 0.08 at 4%. Prose and planning traces measured 0.14 to 0.16. The 4% interval spans 0.1, so that arm neither passes nor fails a 0.1 goal. From 23 experts per layer the interval is mostly or wholly under 0.1. The gain flattens: doubling the allocation from 23 to 46 experts per layer improved the mean by 0.015, less than either interval's half-width.

The three largest allocations traded base cache for refinement cache to stay inside a 53 to 61 GiB admission. They made 10,000 to 21,000 synchronous base reads per trace and decoded at 0.4 to 1.1 tokens/s. Asynchronous readiness also fell behind: at 16 GiB, 4,953 to 6,445 refinements per trace were not ready against 5,516 to 6,878 loaded, and the executed 4-bit share was 38% to 49% against 49% to 57% desired. These arms measure quality against hot allocation. They do not demonstrate a resident base, and a resident base would change both the speed and the readiness. Half of the 16 GiB arm ran while an unrelated download wrote to the same SSD.

The other decode arms used `--base-cache-gib 32`. Each trace made 3,400 to 5,900 synchronous base reads, which explains the low decode rates. Those reads also give the asynchronous loader more time than a resident base would. In the 4% arm, 205 to 364 refinements per trace were not ready and executed at 1.5 bits, against 881 to 1,302 loaded. The 2/288 arm still executed 7% to 12% of routed expert calls at 4 bits, so it is not an all-1.5-bit measurement.

### Full-base admission on a working desktop

The full-resident-base arm with a 4 GiB refinement cache stopped during base loading at layer 29. The guard acted correctly. Free plus file-backed memory was 75.5 GiB. After the 16 GiB host reserve, the supervisor admitted 59.8 GiB. The reference runtime needs about 82 GiB for that arm: 55.46 GiB of base, a 16.8 GiB backbone, the 4 GiB cache and about 5 GiB of working memory.

The operating system reported 86% free memory before that launch. That figure counts inactive anonymous pages, which are not reclaimable without compression or swap. The desktop held about 51 GiB in anonymous, compressed and wired pages. An earlier estimate of 18 GiB came from the largest process sizes and was wrong.

| Refinement cache | Estimated peak | Free plus file-backed memory needed at launch |
|---:|---:|---:|
| 4 GiB | 82 GiB | 98 GiB |
| 8 GiB | 86 GiB | 102 GiB |
| 16 GiB | 94 GiB | 110 GiB |

### Full-base residency after a restart

After a restart with most applications closed, free plus file-backed memory reached 97 to 100 GiB. The arm with the complete base and a 1 GiB refinement cache then completed all eight traces. It made no synchronous base read. Decode ran at 2.28 to 3.90 tokens/s, against 1.55 to 2.16 for the same hot allocation with a 32 GiB base cache. Its KLD of 0.1477 matches that bounded arm's 0.1460, as expected for the same 2.9 experts per layer. Between 28 and 75 refinements per trace were not ready, against 248 to 393 loaded.

Three attempts were stopped first. They corrected the memory model above:

| Attempt | Progress | Stop |
|---|---|---|
| 2 GiB cache | Base layer 39 of 44 | Supervisor budget, 74.3 GiB against 74.1 admitted |
| 1 GiB cache | Base layer 43 of 44 | Process guard at warning pressure, 79.6 GiB, 21.5 GiB still reclaimable |
| 1 GiB cache | Decoding the first trace | Supervisor budget, 82.29 GiB against 82.08 admitted |

- The fixed cost outside the expert caches is about 25.7 GiB, not 16.8 GiB. The backbone alone measures 16.78 GiB of tensors and a 20.87 GiB process footprint, and releasing the MPS cache recovers nothing. The estimates in the table above are therefore about 4 GiB too low.
- `kld_harness.py --guard-above-default-cap` now lets the process guard tolerate warning pressure, matching an external watchdog that stops at critical pressure. The default guard is unchanged.
- The base is not wired. The compressor held 27.7 GiB during the third attempt, so idle base planes were compressed. A resident base that the system compresses costs decompression time on reuse.
- Host headroom fell to 14.6 GiB during the completed run, under the 16 GiB reserve and above the 8 GiB stop line. Larger refinement caches were not attempted.

Full-base residency with streamed refinements is therefore demonstrated only at the smallest hot allocation, and only on a nearly empty desktop. The decode goal and the resident base were not met in one run. No benchmark relaxed the reserve or stopped unrelated applications.

## Split-loop model trial remains experimental

The [four-run model comparison](results/2026-11-06-native-split-model.json) measured a 4.38 tokens/s split-loop median, versus 3.81 for inline controls. That is a 15.2% gain on one fixed-input replay. Both split runs were faster than both controls. The predeclared screening rule passed, but the two controls differed by 0.54 tokens/s. Confidence is moderate in generalizing this result. The 15-30 tokens/s target remains unmet.

| Trial order | Decoder | Decode tokens/s | Peak footprint |
|---|---|---:|---:|
| 1 | inline | 4.076 | 30.27 GB |
| 2 | split | 4.602 | 30.26 GB |
| 3 | split | 4.164 | 30.25 GB |
| 4 | inline | 3.534 | 30.26 GB |

Each fresh process replayed the same 128 inputs with 64 base rows, eight-token chunks, staged required reads, and a 3% refinement allocation. The predictor retained `mix=0.75`, `hm=4`, `G=1`, and a 64-token half-life. Diagnostics stayed disabled. Every replay retained normal pressure, added no swap-outs, and restored Qwen under the unchanged 40 GiB guard.

Ready refinements covered 19.79-19.94% of expert calls and 38.18-38.65% of routing salience. The split runs matched 126 and 127 of the first control's 129 greedy predictions. The second control matched 128. Async refinement availability can change effective precision between runs. Fixed inputs and salience coverage do not establish answer accuracy.

A subsequent HumanEval/3 answer passed the restricted Docker grader at 4.05 decode tokens/s. It generated 149 tokens after 127.91 seconds to the first token. Peak footprint was 30.30 GB and pressure stayed normal. However, system-wide swap-outs increased by 1 MiB, failing the predeclared zero-swap acceptance rule. The counter does not identify the responsible process. Further model trials stopped. The other 11 coding cases were not run, and inline remains the generation default.

All 95 native runtime tests and 18 service/isolation tests passed. Two Max-account Fable/high reviewers checked the kernel, benchmark, runtime wiring, and trial plan. Required source-equivalence and distinct-name checks passed before GPU trials. These native trials retain the 4096-token validation bound. They do not validate the separate daily llama.cpp service's 512K context.

## Split loops reduce mixed component time

The optional split-loop decoder passed the [component criteria](results/2026-11-06-native-split-loops.json) in all 48 comparisons. At eight rows, mixed precision improved by about 16% and all-cold mixed execution by about 20%. These percentages compare medians across six layer/projection medians. The generation default remains the inline decoder.

| Mode | Inline, eight rows | Split, eight rows | Inline, 32 rows | Split, 32 rows |
|---|---:|---:|---:|---:|
| base | 0.385 ms | 0.381 ms | 1.153 ms | 1.152 ms |
| refined | 0.612 ms | 0.613 ms | 1.990 ms | 1.975 ms |
| mixed | 0.621 ms | 0.523 ms | 2.037 ms | 1.690 ms |
| mixed-cold | 0.507 ms | 0.404 ms | 1.608 ms | 1.225 ms |

Each of 24 scenarios used five warm-ups and 25 shuffled samples for each variant. All 8,400 timed outputs matched the inline reference exactly. The separate unbanked reference stayed below 8.83e-9 relative root-mean-square error. That reference shares the inline decoder arithmetic. It checks banking independently, but does not independently validate the decoder.

The predeclared timing gate required mixed gains above the larger interquartile range at both eight and 32 rows. Base and refined regressions could not exceed that range. All comparisons passed. Peak physical footprint was 1.42 GB under the 4 GiB guard. Pressure stayed normal, free memory stayed at least 69%, and no swap-outs were added.

Two Fable/high reviews cleared the component experiment after source and kernel-name checks. The default generated source matched the preceding version across 24 parameter combinations. All 48 variant kernel names were distinct. Five expanded decoder tests passed across both residual patterns, both sign layouts, four threadgroup sizes, and all supported load/lookup variants. Confidence is high in the measured component gain. Its compiler cause and model throughput effect remain unverified.

## Distinct expert row scaling

The [row-scaling probe](results/2026-11-06-native-row-scaling.json) passed every raw output comparison across layers 3 and 44, three projections, and four precision modes. The full eight-row outputs also matched their 32-row prefixes exactly. The production kernel remained unchanged.

The table reports medians across the six layer/projection medians for each mode. Each underlying median contains 25 shuffled samples after five warm-ups.

| Mode | One row | Eight rows | 32 rows |
|---|---:|---:|---:|
| base | 0.166 ms | 0.383 ms | 1.155 ms |
| refined | 0.198 ms | 0.607 ms | 1.985 ms |
| mixed | 0.226 ms | 0.629 ms | 2.039 ms |
| mixed-cold | 0.189 ms | 0.512 ms | 1.607 ms |

Each row selects a distinct expert from the same 32-expert bank. Increasing rows increases both work and the weight footprint. Prefix routes also change cache reuse. These measurements do not isolate arithmetic from memory cost or establish a model throughput ceiling. The one-row mixed case is fully refined; larger mixed counts are half refined. Confidence is high in the component measurements and low in projected model gains.

Peak physical footprint was 1.40 GB, versus the 4 GiB guard. Sampled resident memory peaked at 0.90 GB. Pressure remained normal, reported free memory stayed at least 67%, and no system swap-outs were added. The supervisor restored Qwen after the probe. The separate daily GLM service was then restored with its existing 512K configuration. Two independent Fable/high static reviews found no blockers before execution.

## Cooperative ring result

The [cooperative ring comparison](results/2026-11-05-native-decoder-shuffle.json) retained exact outputs across all 48 variant/scenario pairs. It regressed performance. The original byte decoder remains the generation default.

| Layer and mode | Original | Original with lookup | Cooperative ring | Cooperative ring with lookup |
|---|---:|---:|---:|---:|
| 3, base | 1.063 ms | 1.034 ms | 1.720 ms | 1.542 ms |
| 3, mixed | 1.861 ms | 1.810 ms | 2.961 ms | 2.499 ms |
| 44, base | 1.050 ms | 1.020 ms | 1.690 ms | 1.534 ms |
| 44, mixed | 1.823 ms | 1.786 ms | 2.948 ms | 2.486 ms |

The probe peaked at 733.60 MiB, retained normal pressure, added no swap-outs, and restored Qwen. All 93 runtime tests passed. Max-account Fable/high confirmed the alignment correction and reviewed the cooperative plan. Its requested layout guard is implemented. The review did not predict the observed regression reliably. Confidence is high in these component measurements and low in attributing them to a particular compiler instruction sequence without disassembly. No full-model trial used these variants.

The remaining model-level goal is open. The next investigation checks serial-versus-MTP differences in the already downloaded EXL3 alternative, which previously reached the target range. That earlier result is not current safety evidence for the desktop workload. NestQuant remains available with its original decoder.

## Integer decoder variants

The [four-variant component probe](results/2026-11-05-native-decoder-variants.json) retained exact outputs in all 32 variant/scenario pairs. Wider packed reads regressed. The integer sum lookup improved component medians by about 3%. Neither variant is selected by generation.

| Layer and mode | Original | Packed reads | Integer lookup | Packed reads plus lookup |
|---|---:|---:|---:|---:|
| 3, base | 1.057 ms | 1.858 ms | 1.028 ms | 1.837 ms |
| 3, refined | 1.718 ms | 3.492 ms | 1.662 ms | 3.422 ms |
| 3, mixed | 1.870 ms | 3.011 ms | 1.816 ms | 2.986 ms |
| 3, mixed-cold | 1.465 ms | 2.244 ms | 1.414 ms | 2.230 ms |
| 44, base | 1.056 ms | 1.857 ms | 1.023 ms | 1.835 ms |
| 44, refined | 1.738 ms | 3.508 ms | 1.673 ms | 3.441 ms |
| 44, mixed | 1.818 ms | 2.981 ms | 1.770 ms | 2.954 ms |
| 44, mixed-cold | 1.470 ms | 2.236 ms | 1.421 ms | 2.236 ms |

The probe peaked at 733.94 MiB, retained normal pressure, added no swap-outs, and restored Qwen. Confidence is high in these component observations and low in any projected model gain. Fable/high identified an alignment hazard in the earlier uint-pointer draft. The measured commit already replaced it with byte-aligned packed vectors and offset tests. The follow-up Max-account Fable/high review confirmed that the alignment fix resolves the finding. It also checked the cooperative ring plan and requested a construction-time layout guard. That guard is implemented and tested. These are static reviews, not independent benchmark execution.

The next optional variant loads each compressed ring once across a SIMD group. Uniform lane shuffles extract subsequent states from those loaded words. The 90-byte residual tail includes the first two ring bytes in its final word. Independent bit references, all threadgroup sizes, both sign layouts, and mixed GPU outputs passed in the 93-test suite. This remains a component experiment until real weights and measured timing pass.

## Projection cost probe

The [projection probe](results/2026-11-05-native-projection-stages.json) compared full projections, compiled projections, and captured raw Metal matvec calls. All raw and full outputs remained exact. Compiled outputs also matched exactly in all 24 sampled scenarios. Each variant used five warm-ups and 25 samples in shuffled order.

| Layer and mode | Full projection medians, summed | Raw matvec medians, summed | Compiled projection medians, summed |
|---|---:|---:|---:|
| 3, base | 1.491 ms | 1.147 ms | 1.392 ms |
| 3, mixed | 2.274 ms | 1.891 ms | 2.155 ms |
| 44, base | 1.498 ms | 1.138 ms | 1.402 ms |
| 44, mixed | 2.311 ms | 1.892 ms | 2.172 ms |

These are three separately evaluated projections. Raw calls use already rotated inputs. The measurements suggest prioritizing the integer decoder, with moderate confidence. They do not establish a model throughput ceiling or isolate additive GPU intervals. The probe peaked at 724.03 MiB, retained normal pressure, added no swap-outs, and restored Qwen.

The next optional component comparison tests packed four-byte reads and a 65,536-entry integer lookup. Both retain the original byte decoder as the default. All 92 runtime tests passed, including every lookup entry, circular bit references, byte-offset views, and mixed projection equality. Apple specifies byte alignment for `packed_uchar4` in [Metal Shading Language, table 2.4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf). The wider read checks both word ends before loading and falls back near ring boundaries. The model path does not select either variant yet.

The lossless archive experiment reduced base storage from 59,553,531,072 to 58,326,697,152 bytes, saving 1.143 GiB. All 42 layers passed full byte reconstruction, and 24 real GPU comparisons matched exactly. Conversion and GPU verification peaked at 3.26 and 3.37 GiB, retained normal pressure, added no swap-outs, and restored Qwen. The subsequent full-base probe stopped at pressure level 2 on layer 42. Its child recorded a 57.68 GiB lifetime peak, versus 56.54 GiB sampled by the outer watchdog. Another 2.60 GiB of base remained unloaded. No refinements or generation were included. Qwen recovered with zero added swap-outs. Confidence is high in this failure and low in treating its stopping point as a fixed capacity. See [the lossless archive evidence](results/2026-11-05-native-lossless-base.json).

The compact-backbone full-base probe stopped at pressure level 2 on layer 44. It peaked at 61.43 GiB, retained at least 22% reported free memory, added no system swap-outs, and restored Qwen. The final layer loaded, but the final page sweeps did not complete. The existing 84/82 GiB guards and 56 GiB residency request were unchanged. Confidence is high that this configuration did not fit safely in the observed desktop state. This does not prove that every compact layout is infeasible. See [the stopped probe](results/2026-11-05-native-compact-fullbase-stop.json).

Owner: Andru. Measured: 2026-11-04 to 2026-11-05.

Native mixed execution passed 10 of 12 coding cases at a median 3.55 decode tokens/s with a 32 GiB base limit. A separate 40 GiB coding case passed at 4.62 tokens/s. A fixed-token replay reached 3.94 tokens/s, compared with 1.27 tokens/s for the reference path. These are different workloads and configurations. Interactive latency remains high, and the 15–30 tokens/s target remains unmet.

Hardware: Apple M5 Max, 40 GPU cores, 128 GiB unified memory, macOS 27.2. Checkpoint: GLM-5.3-Flash NestQuant 1.5/4, revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`.

## Bounded paged residency

A four-run zero/32/32/zero GiB comparison used source `6c36385`, 64 base rows, eight-token chunks, 3% hot allocation, and staged required reads. The outer watchdog stayed at 40 GiB. All runs retained normal pressure and added no system swap-outs. Qwen recovered after each trial.

| Requested residency | Decode tokens/s | First token | Required base read, copy, and lock wait |
|---|---:|---:|---:|
| Zero, first control | 4.065 | 45.58 s | 11.49 s |
| 32 GiB, first candidate | 4.594 | 44.38 s | 10.37 s |
| 32 GiB, second candidate | 4.705 | 43.59 s | 10.43 s |
| Zero, second control | 4.115 | 44.99 s | 11.25 s |

The candidate median was 4.65 tokens/s, versus 4.09 for controls, a 13.7% increase on this replay. Confidence is high in these measurements and moderate in generalizing the gain. Every peak stayed below 28.20 GiB. Reported free memory stayed at or above 46%. Both candidates retained approximately 27.8 GiB of resident pages after decode. Resident, wired, framework, and physical counters overlap.

The 3% allocation contained 363 slots. Candidate ready-call coverage was 19.54-19.70%, with 38.06-38.23% salience coverage. Wanted-call delivery fell to 95.23-96.10%, versus 97.41-97.48% in controls. One candidate matched the first control on 126/129 greedy predictions, and the other matched 129/129. Fixed inputs do not freeze refinement readiness or establish answer quality.

A subsequent autoregressive HumanEval/3 answer passed the pinned restricted grader at 4.22 decode tokens/s and 128.03 seconds to first token. It peaked at 28.22 GiB, retained normal pressure, added no swap-outs, and restored Qwen. This is one coding case. The owner waived Fable review before these trials. The 15-30 tokens/s target remains unmet. See [the results and protocol](results/2026-11-05-native-paged-residency.json).

## Base cache and native execution

A fixed 128-token replay matched the reference store's 5,641 observed decode base misses at 32 GiB. On the same access trace, a 40 GiB cache would have missed 4,817 times. The trace produced the same miss count at 44, 48, and 56 GiB. This run starts cold. It measures one prompt and holds the observed routes fixed. It does not estimate the memory pressure or routing changes of a live larger cache.

An offline audit reconciled both reference traces with their recorded 32 GiB miss and eviction counts. Their decode phases first accessed 4,817 and 4,844 experts that prefill had never used. At 40 GiB, every remaining simulated miss was one of those first accesses. Better eviction cannot remove these cold reads on these traces, with high confidence. Prior residency or accurate prefetch could move them earlier. This result does not establish the optimum for native per-layer banks or different prompts. See [the first-access counts](results/2026-11-05-cache-first-accesses.json).

| Fixed-token reference comparison | Synchronous base | Async base after route selection |
|---|---:|---:|
| Decode rate | 1.272 tokens/s | 1.268 tokens/s |
| Decode base misses | 5,641 | 5,677 |
| Decode base-load wall time | 48.75 s | 50.33 s |
| Base reads delivered by the worker | 0 | 5,275 |
| Time waiting for the base worker | 0 s | 40.56 s |
| Peak physical process footprint | 57.87 GiB | 59.69 GiB |
| Minimum reported free memory | 49% | 39% |

Both runs used 32 GiB of base cache, the existing asynchronous refinement scheduler, 2% refinement slots, and eight-token layer-major prefill. The input tokens matched. The greedy next token matched at 128 of 129 positions. Refinement readiness and routes can change across runs. Keep synchronous base reads as the reference default. The separate async base worker reserves 64 MiB. It stages required reads once the route is known. The caller waits for a running required read and performs the ordinary synchronous read if it cancels a queued read. The real-weight check matched all tested planes and observed 60.00 MiB of staging.

The synchronous replay spent 13.94 s at the PyTorch-to-host boundary, including upstream Metal synchronization, 0.10 s submitting inputs to MLX, 0.08 s copying evaluated MLX results to host, and 1.50 s returning the output to PyTorch. Those are wall-clock windows. They include work scheduled by the preceding stage where a call synchronizes a stream. The profiling option separates an explicit upstream drain from the host conversion. The existing PyTorch path reports reference causal-convolution and recurrent Kimi Delta Attention fallbacks.

A separate 32-token reference replay used explicit upstream synchronization at each crossing. Across 33.34 s of decode, it measured 3.65 s draining previous Metal work, 0.89 s copying PyTorch tensors to host, 0.025 s submitting to MLX, 0.021 s copying evaluated MLX output to host, and 0.37 s returning output to PyTorch. Required base reads took 18.03 s. The added synchronization changes scheduling, so these timings describe this profiling run.

### Memory feasibility on this desktop

The prepared base banks contain 55.46 GiB of packed planes, scales, correction metadata, and low-rank padding. The affine-8 backbone, embedding, and output head contain 9.79 GiB. The native 2% decode refinement bank reserves about 1.87 GiB. The measured short native contexts use about 141 MiB of recurrent cache arrays and less than 2 MiB of sparse attention cache arrays. These logical sizes exclude conversion scratch and framework overhead.

| Load experiment | Observed result |
|---|---|
| Direct shared Metal read, base alone | Pressure rose after 47.52 GiB of logical base, at a 47.57 GiB process footprint. |
| Stock MLX read, base alone | Pressure rose near 50 GiB of process footprint. |
| Stock MLX read with a 56 GiB process residency request | Layer 43 completed with 54.14 GiB of base. Pressure rose before the last layer finished. |
| File-backed MLX views, base alone | All banks mapped at 0.67 GiB process footprint. A GPU sweep of the mapped pages raised system pressure. |
| File-backed base with the prepared backbone | All banks mapped. Both the full forward and one-layer evaluation stopped under system pressure before the first answer. |
| Bounded native banks, 96 rows per layer | Two 64-token all-base requests completed at 2.58 tokens/s median and a 29.12 GiB sampled process peak. |

The supervisor suspended both active model downloads during these runs and retained the memory they already owned. In the direct-read trial, system compressor occupancy rose from 20.47 to 57.66 GiB as file-backed pages fell from 62.25 to 22.40 GiB. The guard tripped at pressure level 2 while the system reported 45% free memory. The complete base-plus-backbone path did not demonstrate safe residency under this desktop workload. Mapped tensors count towards MLX active bytes and may fault from storage. Process footprint, resident pages, and MLX accounting describe different views of the same allocations. The reference PyTorch driver counter also covers Metal allocations and must not be added to MLX active memory.

### Later full-base residency checks

After both model downloads exited, the complete 55.46 GiB base passed GPU page sweeps under the unchanged external guards. The stock MLX reader with a 56 GiB process residency request reached a 55.65 GiB peak footprint. macOS pressure remained normal, and reported free memory stayed at or above 29%. System swap-outs increased by 52.63 MiB during this run. This changes the earlier feasibility result: the complete packed base can fit by itself under the observed workload.

Touching each layer on the GPU before loading the next, with a 250 ms residency refresh, completed without additional system swap-outs. Its final resident-page count was 55.50 GiB, compared with 32.34 GiB in the preceding control. Both ultimately reported about 55.47 GiB wired. These runs changed loading behavior and occurred sequentially; they do not isolate refresh from page touching or background workload changes.

| Residency probe | Peak process footprint | Minimum reported free | Pressure | Result |
|---|---:|---:|---|---|
| Complete base, original loading mode | 55.65 GiB | 29% | Normal | Passed both GPU sweeps |
| Complete base, incremental GPU touches and refresh | 55.65 GiB | 29% | Normal | Passed both GPU sweeps |
| Same incremental probe from the tracked script | 55.65 GiB | 25% | Normal | Passed both GPU sweeps |
| Complete base plus affine-8 backbone, stock MLX reader | 64.38 GiB | 23% | Level 2 | Guard stopped at layer 43 |
| Direct persistent preload plus affine-8 backbone | 53.81 GiB | 26% | Level 2 | Guard stopped after layer 35 |
| Stock MLX reader plus backbone, 56 GiB residency request | 63.06 GiB | 22% | Level 2 | Guard stopped at layer 42 |
| Uncached MLX-owned reader plus backbone, 56 GiB residency request | 59.09 GiB | 23% | Level 2 | Guard stopped at layer 39 |

The combined probe's last normal-pressure layer contained 52.81 GiB of base and the 9.79 GiB backbone. Another 2.66 GiB of base remained at that point. The next layer triggered pressure before the final layer loaded. Refinements, predictor state, and generation caches were absent. A larger process cap would not solve this stop because system pressure triggered first.

The direct loader subsequently stopped after loading 43.56 GiB of base alongside the backbone. Another 11.90 GiB of base remained. The external supervisor detected pressure level 2, stopped the child, and restored Qwen. System swap-outs did not increase. Compressor occupancy increased from 16.09 to 30.02 GiB, while file-backed pages fell from 59.21 to 27.82 GiB. The process wired counter remained near 9.92 GiB, while the system wired counter increased with the imported base. These counters do not attribute wired pages in the same way. The run did not generate tokens and supplies no throughput result. Full-model residency under these guards remains unproven.

Reducing the stock loader's residency request from 68 to 56 GiB did not complete the combined probe. The inner guard stopped at pressure level 2 after layer 42, with 2.66 GiB of base still missing. The process peak was 63.06 GiB. Swap-outs did not increase, and Qwen recovered. A residency request is not a hard limit on every wired page reported by the process or system.

The uncached MLX-owned reader also stopped under pressure, after layer 39 with 6.62 GiB of base still missing. Its peak was 59.09 GiB. System swap-outs increased by 2.25 MiB across the run and cleanup. Qwen recovered. This variant uses an uncached file handle and retains MLX-owned arrays. A source-overwrite fixture verifies storage independence. MLX 0.32.4 evaluates each tensor serially for a supplied file object, confirmed by an instrumented 32 MiB fixture. This result does not support file-cache bypass alone as a solution to the combined residency limit.

A further uncached owned-array control omitted the explicit MLX residency request. It stopped at layer 32 under inner pressure level 2, with a 49.84 GiB peak and 43% minimum reported free memory. Another 15.87 GiB of base data remained unloaded. It added no system swap-outs and restored Qwen. Removing persistent wiring alone did not make the combined load pass. The source-default wiring limit is documented as zero in [MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html); this does not prevent the driver from making active GPU resources resident.

The direct loader passed real eight-expert comparisons at layers 3 and 44. Relative RMS error was at most 9.60e-9 against the per-expert reference. The check forbade further base reads during computation. It does not establish zero misses during full-model generation. All 36 runtime unit tests and six server boundary tests passed. See the [logical byte inventory, memory samples, guard receipts, and parity results](results/2026-11-05-full-base-residency.json).

A separate 64 MiB imported-buffer probe increased MLX active allocation by exactly 64 MiB. The [MLX allocator](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/allocator.cpp) also registers imported buffers in its residency sets. The probe's wired counter includes driver overhead and does not attribute every byte to the imported buffer. Logical bytes, MLX allocation, resident pages, and physical footprint remain overlapping measures.

The native tuned replay reserved 30.81 GiB of base banks and 1.87 GiB of decode refinement banks. Its largest one-layer prefill bank reserved 1.24 GiB. The async refinement worker reserved at most 49.53 MiB in this run. The affine-8 backbone uses 9.79 GiB, including 1.18 GiB of embeddings and 0.63 GiB of output weights. The CPU predictor retained 474.64 MiB of weights and buffers, 7.36 MiB of attention history, and 148 KiB of policy arrays. The model cache used 140.78 MiB for recurrent state and 1.80 MiB for sparse attention. These tensor and staging figures overlap with the framework and process measurements below.

| Native tuned replay memory | After model load | After prefill | After decode |
|---|---:|---:|---:|
| Physical process footprint | 10.68 GiB | 32.30 GiB | 45.73 GiB |
| Resident process pages | 10.85 GiB | 31.99 GiB | 45.40 GiB |
| MLX active allocation | 42.48 GiB | 43.86 GiB | 42.62 GiB |
| MLX allocator cache | <0.01 GiB | 0.20 GiB | 0.22 GiB |

The native banks reserve pageable storage before their rows are filled. In the reference replay, after decode, PyTorch MPS reported 16.94 GiB of tensor allocation and 52.94 GiB in its driver. MLX reported 33.85 GiB active and 0.47 GiB cached. The 57.87 GiB physical footprint measures the whole reference process. During the native tuned run the reported minimum system free memory was 37%. System-wide file-backed pages fell from 66.85 GiB before launch to 25.13 GiB after the child exited. Compressor occupancy rose from 19.56 to 22.56 GiB across those samples.

The bounded native prototype keeps hidden states, router weights, expert computation, and attention in MLX. It synchronizes and reads router IDs for disk-backed slot selection. The reported forward-to-route interval includes upstream model work. Packed arrays retain fixed storage. A slot may be overwritten only after the previous GPU output has evaluated. The mixed kernel selects base and refinement slots separately and preserves the reference output within 0.000002% relative RMS in the tested projections. Native mixed execution passed the exact reply and arithmetic smoke cases. The same-chunk layer-major check matched logits and all 145 cache arrays in the all-base runtime. See [run receipts](results/2026-11-05-native-memory-replay.json).

The native mixed runtime passed HumanEval/3 in a network-disabled grading container. It generated 149 tokens at 2.04 tokens/s after a 194.49 s prefill on a 135-token prompt. The run used 96 base rows per layer, 64 four-bit prefill rows, eight-token chunks, and 242 decode refinement slots. The peak process footprint was 33.43 GiB. Memory pressure stayed at level 1, with a minimum reported 48% free. The MLX peak was 30.94 GiB and the allocator held 253 MiB at the end. These measurements overlap the process footprint. The predictor retained 474.64 MiB of parameters and buffers and 11.95 MiB of attention history. The prompt and answer used 140.78 MiB of recurrent state and 3.24 MiB of sparse attention state.

The hot policy wanted 5,395 of 50,064 routed calls. It delivered 5,335, or 98.89% of wanted calls. Ready calls covered 10.66% of routed calls and 22.80% of squared routing salience. Across 134 overlapping calibration windows, jT assigned 25.11% of salience to the top budget, compared with 22.45% observed. These are results for one prompt. They measure scheduling and calibration, with one separate answer-quality check.

The chunking diagnosis locates the first BF16 discrepancy in layer-zero attention, before any routed expert. Its relative RMS difference was 0.0282% after attention and 0.7191% after the fourth layer. Cached decode still differed by 1.3086%, above the 1% check. The diagnosis compared ordinary Transformers with and without chunks. Use `verify_integration.py --reference chunked --diagnose-chunking` to reproduce it.

A precision control reproduced the 1.3086% failure with identical BF16 initialization. Upcasting the backbone after initialization reduced cached-decode error to 0.00427%, a 306.8-fold reduction. Prefill error fell from 0.67655% to 0.00336%. The real NestQuant expert path retained its float16 boundaries. This supports backbone rounding sensitivity as the dominant cause in this fixture, with high confidence. It does not prove trained-model accuracy or justify changing the original 1% threshold. The BF16 arm still exits nonzero and now preserves a failed JSON report. Both arms peaked below 7.10 GiB, retained normal pressure, recorded no additional swap-outs, and restored Qwen. See [the precision controls and complete receipts](results/2026-11-05-chunk-precision-control.json).

In a separate trained layer-zero component probe, median Torch/MPS attention took 1.225 ms. With synchronization around individual fallbacks, the convolution update took 0.213 ms and the recurrent Kimi Delta Attention function took 0.410 ms. Those component windows do not sum to a single ordinary forward. MLX BF16 took 0.756 ms at 0.884% relative RMS difference. MLX affine-8 took 0.587 ms at 1.824%. Affine-4 took 0.450 ms at 19.588% and was rejected. This probe uses 32 recurrent positions and measures one attention layer.

### Controlled preload pacing

Source `88f4064` adds optional pacing after each positional read, with pressure checks before and after every wait. Two independent Max-account Fable/high reviews cleared the control-first protocol after failure evidence was added. All 46 runtime unit tests and six server boundary tests passed. Each trial released idle Qwen and retained the 82 GiB inner guard, 84 GiB external guard, normal-pressure requirement, and 20% reported-free floor.

| Native full-base startup | Failure layer | Peak footprint | Base still unread | Read time | Pacing waits | Minimum reported free |
|---|---:|---:|---:|---:|---:|---:|
| No explicit wiring, unpaced | 25 | 39.37 GiB | 26.32 GiB | 6.92 s | 0 s | 44% |
| No explicit wiring, 512 MiB/s | 24 | 38.91 GiB | 26.65 GiB | 12.31 s | 47.18 s | 46% |
| 68 GiB residency request, 512 MiB/s | 37 | 55.13 GiB | 10.57 GiB | 12.63 s | 76.72 s | 23% |

Every trial stopped at pressure level 2 before prefill or generation. None reached either process cap. All restored Qwen, preserved the unrelated port forward, and added zero system swap-outs. The failed layer event retains its partial byte count and memory sample. The two inner-guard failures reported `MemoryError: macOS reports memory pressure; stop inference and reduce memory use`. The external supervisor also stopped the unpaced control under pressure.

Pacing engaged but did not make the unwired configuration pass, with high confidence for these trials. At failure, its process resident pages were 6.40 GiB versus a 38.91 GiB physical footprint. The wired trial retained 54.76 GiB of resident pages at a 55.13 GiB footprint. System compressor occupancy and process footprint overlap; the compressor cannot be attributed entirely to unrelated applications. Wiring changed residency materially, but neither combination established full-model feasibility. Sequential background activity limits causal attribution beyond this controlled setting.

The profiles use `--full-base --rows 288 --base-budget-gib 56 --chunk-size 8 --max-tokens 16`. Set `--full-base-wired-gib` and `--preload-mib-s` to the table's values. Keep the host watchdog active. The [pacing report](results/2026-11-05-preload-pacing.json) records complete arguments, per-layer samples, and guard outcomes. Startup `read_seconds` now measures only positional read calls. It excludes checks and pacing, so older whole-preload timings are not directly comparable. These failed trials supply no throughput or quality result.

### Matched native replay

Each run used the same 29 rendered prompt tokens and the same 128 forced decode inputs. Both native runs used the affine-8 backbone, four-bit prefill, eight-token chunks, 160 refinement scratch rows, and 242 decode refinement slots. The tuned run redistributed the same 6,720 base slots using the first half of the earlier reference trace. The two native reservations differed by about 1.3 MiB because packed correction sizes vary by layer.

| Measurement | PyTorch/MLX reference | Native, uniform rows | Native, tuned rows |
|---|---:|---:|---:|
| Startup | 7.28 s | 1.85 s | 2.11 s |
| Prefill to first token | 92.84 s | 55.86 s | 53.59 s |
| Decode throughput | 1.272 tokens/s | 2.792 tokens/s | 3.519 tokens/s |
| Base misses | 5,641 | 6,473 | 5,906 |
| Base read time | 48.75 s | 20.58 s | 17.58 s |
| Peak process footprint | 57.87 GiB | 45.68 GiB | 45.73 GiB |
| Minimum reported free memory | 49% | 39% | 37% |
| Greedy next-token agreement with reference | 129/129 | 129/129 | 128/129 |

The tuned allocation reduced native misses by 8.8%. This comparison holds the token sequence fixed, but it does not hold routing or operating-system storage state fixed. Refinement readiness matched on 17 of 128 decode steps between native runs. Treat the time difference as one measured outcome and the tuned plan as a calibration candidate. The evaluation still needs an independent trace. The reference uses a different backbone representation and reader, so this comparison does not isolate the benefit of the framework change or the banked kernel.

### Larger bounded caches

The larger-cache runs retain the same 29-token prompt, 128 forced tokens, 2% hot allocation, and eight-token chunks. The four-bit prefill scratch bank holds 160 rows. Each profile ran once. Background activity and asynchronous refinement readiness differ between runs.

| Native base layout | Logical base | Decode tokens/s | Base misses | Base read time | Peak process footprint | Minimum reported free |
|---|---:|---:|---:|---:|---:|---:|
| Uniform 160, 32 GiB limit | 30.81 GiB | 2.792 | 6,473 | 20.58 s | 45.68 GiB | 39% |
| Tuned 160, 32 GiB limit | 30.81 GiB | 3.519 | 5,906 | 17.58 s | 45.73 GiB | 37% |
| Uniform 192, 40 GiB limit | 36.98 GiB | 3.816 | 5,431 | 16.00 s | 51.34 GiB | 32% |
| Uniform 224, 44 GiB limit | 43.14 GiB | 3.941 | 5,001 | 15.09 s | 54.23 GiB | 27% |

Both larger profiles completed at normal pressure without additional system swap-outs and restored Qwen. The 44 GiB profile gained 3.3% over 40 GiB while reserving another 6.16 GiB of base tensors. The 40 GiB profile is a provisional option for further supervised tests, with moderate confidence. Longer contexts and broader quality checks remain untested. The default stays at 32 GiB.

Each larger profile matched 126 of 129 greedy next-token predictions from the tuned baseline. Their forced input tokens were identical. At 40 GiB, refinements covered 15.67% of expert calls and 30.95% of salience. The scheduler delivered 98.62% of wanted calls. These are coverage measurements, not answer-accuracy scores.

An autoregressive HumanEval/3 check at 40 GiB passed the network-disabled grader. It emitted the same 149 tokens as the earlier 96-row run. Decode increased from 2.04 to 4.62 tokens/s, and prefill fell from 194.49 to 130.66 seconds. This comparison also increased the prefill scratch bank from 64 to 160 rows. Its 51.92 GiB peak stayed at normal pressure with at least 33% reported free memory and no additional system swap-outs. One passing case does not establish general accuracy.

On that coding request, the 2% allocation delivered 10.60% refined calls and 22.75% salience coverage. Delivery of wanted calls was 98.22%. The predictor's top-budget forecasts assigned 24.99% salience versus 22.45% observed across overlapping windows. No predictor parameters were fitted on this request.

The later route timer accidentally included base selection and reads. Those windows must not be added to base-read time or compared with the earlier route-only timer. Complete decode rates and separate read times remain valid. Source `c56076c` restores the original timing boundary. See [the complete replay, coding, and isolation evidence](results/2026-11-05-native-cache-sweep.json).

### Prefill configuration sweep

Three runs at source `8a162b9` compared supported prefill settings without changing the runtime. Each used 192 base rows, a 40 GiB limit, 2% hot allocation, 29 prompt tokens, and 128 identical forced inputs. The 16-token, 240-row prefill check first matched logits, all 167 cache arrays, and committed predictor state between token-major and layer-major schedules. It compared equal chunk sizes, not 8-token against 16-token numerical equivalence.

| Prefill chunk / scratch rows | First-token latency | Decode tokens/s | Prefill refinement reads | Peak footprint | Minimum reported free |
|---|---:|---:|---:|---:|---:|
| 8 / 160 | 52.89 s | 3.095 | 31.86 GiB | 51.33 GiB | 31% |
| 8 / 240 | 54.67 s | 3.156 | 31.86 GiB | 51.41 GiB | 30% |
| 16 / 240 | 53.30 s | 3.547 | 31.89 GiB | 51.48 GiB | 31% |

The larger settings did not improve first-token latency on this short prompt. The scratch increase did not reduce refinement bytes. Decode always processes one committed token per step, so its observed speed difference cannot be attributed directly to larger prefill batches. Greedy next-token predictions matched the control at 125/129 and 124/129 positions, respectively. Asynchronous precision and background activity remain confounders.

All three replays completed at normal pressure and restored Qwen. The 8/160 control added 448 KiB of system swap-outs across the run and cleanup. The other two added none. The preceding parity run added 256 KiB while retaining normal pressure. The [sweep report](results/2026-11-05-native-prefill-sweep.json) contains arguments, predictor coverage, memory samples, and hashes of the complete local reports. These short replays do not establish longer-prompt memory safety or task accuracy.

A separate 135-token coding request with 192 base rows, 16-token chunks, and 240 scratch rows produced no saved answer. Its last durable sample reported pressure level 2 at a 51.23 GiB peak. The execution handle and child process were absent after interruption, while the receipt still said `running`. The exact exit reason and cleanup timeline were not captured. A live recovery check verified healthy Qwen and normal system pressure. The sweep report preserves this incomplete attempt separately. It supplies no throughput or quality score and prevents treating the short sweep as longer-prompt validation.

### Twelve-case coding subset

Source `8a162b9` passed 10 of 12 fixed HumanEval cases using native mixed execution. Each case ran in a fresh process with 160 base rows, a 32 GiB base limit, 16-token chunks, and 240 prefill rows. The hot allocation remained 2%, asynchronous refinements remained enabled, and predictive base reads remained disabled. Outputs had a 768-token ceiling and used the thinking-off template. These settings preserved the external 84 GiB guard, inner 82 GiB guard, normal-pressure requirement, and 20% reported-free floor.

| Measurement across twelve coding cases | Result |
|---|---:|
| Held-out executable checks | 10/12 passed |
| Median decode rate | 3.546 tokens/s |
| Decode range | 2.727–4.172 tokens/s |
| Median first-token latency | 115.807 s |
| First-token latency range | 80.221–147.136 s |
| Peak process footprint | 46.02 GiB |
| Minimum reported free memory | 34% |
| System swap-out increase, trials and cleanup | 2.125 MiB |
| Normal-pressure completion and Qwen restoration | 12/12 |

HumanEval/97 failed because the answer applied a final modulo after multiplication. HumanEval/158 reversed the maximum-selection rule and its tie-break. The generated answers completed normally, so these are answer failures rather than truncated outputs. The same twelve cases previously scored 11/12 with the separate EXL3 runtime, which failed only case 97. That comparison changes quantization and execution, so it does not identify the cause of the extra NestQuant failure.

Across 2,192 decode steps, the 242 allocated slots refined 86,879 of 736,512 routed calls (11.80%). They covered 25.23% of squared routing salience and delivered 98.85% of wanted calls. These are aggregate predictor and scheduler measurements, separate from the executable score. Twelve cases do not establish broad coding accuracy, with high confidence about that limit.

The [coding report](results/2026-11-05-native-coding-subset.json) includes every response, token sequence, command, phase counters, coverage report, and source-file hash. The grader used the pinned HumanEval dataset and a network-disabled, read-only container with a 256 MiB limit. Model startup, cold prefill, and decode are reported separately. These coding rates must not be substituted for the fixed-replay rates. System swap-outs also include other processes and cleanup, so they cannot be attributed entirely to inference.

### Native hot-allocation comparison

Four fixed-token replays alternated 2% and 3% hot allocations at source `52a54ba`. Each used 160 base rows, a 32 GiB base limit, 16-token chunks, 128 prefill rows, and the same 128 forced inputs. Predictor parameters remained `mix=0.75`, `hm=4`, `G=1`, and `half_life=64`. Predictive base reads remained disabled. Each profile ran twice under the same guards.

| Measurement | 2% allocation | 3% allocation |
|---|---:|---:|
| Allocated hot slots | 242 | 363 |
| Refinement reservation, including prefill and worker | 2.928 GiB | 3.864 GiB |
| Individual decode rates | 3.277, 3.484 tokens/s | 3.360, 3.448 tokens/s |
| Median decode rate | 3.381 tokens/s | 3.404 tokens/s |
| Median first-token latency | 52.207 s | 51.370 s |
| Actual refined-call coverage | 15.73% | 20.00% |
| Routing salience coverage | 31.06% | 38.79% |
| Delivery of wanted calls | 98.70% | 97.95% |
| Peak process footprint | 45.85 GiB | 46.80 GiB |
| Minimum reported free memory | 34% | 35% |
| Added system swap-outs | 0 | 0 |

The 3% profile increased refined-call and salience coverage in both runs without exceeding the existing 4 GiB refinement budget. Reducing prefill scratch capacity made that allocation possible. All four trials completed at normal pressure and restored Qwen. The roughly 0.7% median throughput difference is smaller than the variation between the controls. These two repetitions do not establish a speed gain, with high confidence about that limitation.

The [complete comparison](results/2026-11-05-native-hot-fractions.json) includes commands, prediction calibration, read counters, readiness digests, and evidence hashes. Forced inputs matched, while greedy predictions and asynchronous precision could differ.

The subsequent twelve-case coding check at source `e30f77e` also passed 10/12 with 3% hot allocation. It retained 160 base rows, 16-token chunks, and 128 prefill rows. Every case ended normally, retained normal pressure, and restored Qwen. The [three-percent coding report](results/2026-11-05-native-coding-subset-p3.json) includes responses, counters, exact commands, and hashes for each result, receipt, and log.

| Coding measurement | 2% / 240 prefill rows | 3% / 128 prefill rows |
|---|---:|---:|
| Executable checks passed | 10/12 | 10/12 |
| Median decode rate | 3.546 tokens/s | 3.467 tokens/s |
| Median first-token latency | 115.807 s | 133.543 s |
| Peak footprint | 46.02 GiB | 46.87 GiB |
| Actual refined-call coverage | 11.80% | 15.35% |
| Routing salience coverage | 25.23% | 31.81% |
| Delivery of wanted calls | 98.85% | 98.28% |
| Minimum reported free memory | 34% | 34% |
| Added system swap-outs | 2.125 MiB | 0 |

Both profiles failed cases 97 and 158. At 3%, case 158 selected the maximum unique-character count but still omitted the required lexicographic tie-break. Eight responses had identical token sequences across the profiles. The smaller prefill bank accompanied an increase in aggregate refinement reads from 850.58 to 1,087.59 GB. Different scratch capacity, generated trajectories, and desktop activity prevent attributing the timing difference solely to hot allocation.

Three percent improved measured coverage but did not improve this executable score. Confidence is high about the observed result and low about generalizing twelve cases to broad accuracy. The default remains unchanged.

### Refinement retention policy

Source `9060b5e` tested the existing retention parameter without changing runtime code. All runs used 3% hot allocation, 160 base rows, 16-token chunks, 128 prefill rows, and the same 128 forced inputs. The policy multiplies an existing pool member's score by `1 + hm`. The sequence was `hm=4`, `hm=1`, `hm=0`, and a second `hm=4` control. Other predictor settings and all memory guards remained unchanged.

| Measurement | `hm=4`, two controls | `hm=1`, one run | `hm=0`, one run |
|---|---:|---:|---:|
| Median decode rate | 3.412 tokens/s | 3.185 tokens/s | 2.976 tokens/s |
| Actual refined-call coverage | 20.03% | 23.87% | 24.65% |
| Routing salience coverage | 38.80% | 47.21% | 48.33% |
| Delivery of wanted calls | 97.98% | 95.69% | 87.16% |
| Median decode refinement reads | 7.92 GB | 22.90 GB | 38.78 GB |
| Minimum reported free memory | 35% | 34% | 35% |

The controls measured 3.496 and 3.328 tokens/s. Lower retention increased ready salience but also increased storage traffic and reduced delivery reliability. The result supports retaining `hm=4` for throughput, with moderate confidence given the small number of runs. It does not establish which policy gives better answers.

All four runs completed at normal pressure and restored Qwen. The first control and `hm=1` added no swap-outs. The `hm=0` run added 1,113,636,864 bytes (1.04 GiB), and the final control added 589,824 bytes. Both increases were already present after prefill, before measured decode, and did not increase during decode. These are system-wide counters, so the experiment cannot attribute them solely to the retention setting or GLM. The [retention report](results/2026-11-05-native-retention.json) preserves the phase counters, commands, prediction changes, and evidence hashes. This replay supplies no answer-quality score.

### Contiguous required base reads

Source `58f8e74` compared the original field-major archive with an expert-major archive. The latter stores each expert's 24 packed planes contiguously. Both independent Max-account Fable/high reviewers cleared the comparison and the corrected local runner before launch. All 42 converted layers passed byte equality and 756 projection checks beforehand. The GPU bank layout and quantization did not change.

Four isolated trials used ABBA order: original, contiguous, contiguous, original. Each used 160 base rows, a 32 GiB base limit, 16-token chunks, 128 prefill rows, and the same 128 forced inputs. Refinements used 3% allocation, the 4 GiB budget, and the unchanged `hm=4` policy. Base prediction and the cold fast path remained off.

| Measurement | Original archive | Contiguous archive |
|---|---:|---:|
| Decode rates, two trials | 3.006, 3.434 tokens/s | 3.775, 3.655 tokens/s |
| Median decode rate | 3.220 tokens/s | 3.715 tokens/s |
| Median blocking base-read time | 19.533 s | 16.376 s |
| Read time per miss, two trials | 3.096, 2.959 ms | 2.537, 2.539 ms |
| Median requested base bytes | 31.756 GB | 31.765 GB |
| Median first-token latency | 52.190 s | 50.058 s |
| Peak process footprint | 46.80 GiB | 46.78 GiB |
| Minimum reported free memory | 36% | 34% |

Median decode increased by 15.4%, and blocking base-read time decreased by 16.2%. Per-miss latency improved in both contiguous trials beyond the spread between controls. Confidence is high in the measured I/O reduction and moderate in generalizing the model speed gain. There were only two trials per layout. The first control's forward-to-route interval was 18.59 seconds, versus 13.89–14.87 seconds in the other trials. That interval includes pending GPU work, so the complete speed difference cannot be attributed solely to storage.

All four trials began and remained at pressure level 1. They added zero system swap-outs and restored Qwen. The 84 GiB external cap, 82 GiB inner cap, and 20% reported-free floor remained unchanged. Process disk reads were 40.11–40.61 GB per decode. Requested base reads were about 31.8 GB, with another 7.86–8.01 GB reported by the refinement reader. The process counter covers both readers, so comparing it only with base bytes would misstate cache effects.

Actual refined-call coverage stayed between 19.90% and 20.11%. Salience coverage stayed between 38.56% and 38.88%. Greedy predictions agreed with the first control on 127/129 and 128/129 positions for the contiguous trials. The second control agreed on 128/129. Readiness digests agreed on only 3/128, 1/128, and 5/128 steps respectively. Fixed inputs therefore did not establish identical executed precision or outputs. This experiment supplies no answer-quality score.

The timing counters attribute about 33.5 seconds of the roughly 50-second prefill to refinement reads. Installation adds about 2.1 seconds and may include waiting for prior GPU work. Publication timing has the same synchronization caveat. These counters can overlap background activity and must not be summed indiscriminately. The 15–30 tokens/s target remains unmet. See [the full comparison](results/2026-11-05-native-storage.json) and [archive validation](results/2026-11-05-native-storage-validation.json).

### Required-read diagnostics

Source `a206d8e` added selection miss histograms and a bounded CPU read probe. Two independent Max-account Fable/high reviewers approved one component sweep and one fixed-input model replay. All 66 runtime tests and five supervisor tests passed before launch.

The component sweep used widths 1, 2, 4, 4, 2, and 1 over the same 384 rows. The first serial trial requested 1.891 GB and physically read 1.897 GB. Its active throughput was 7.83 GB/s, with 0.578 ms median latency per read. Later trials physically read only 0.403 GB each, or 21.3% of requested bytes. Their apparent width gains are cache-confounded and cannot justify an SSD reader pool. Confidence is high in this measurement limitation.

The component's lifetime peak footprint was 49,988,328 bytes. It retained pressure level 1 and at least 72% reported free memory. System swap-outs did not increase, and Qwen recovered.

The histogram replay failed under pressure level 2 after 337.5 seconds. It used the preceding storage profile unchanged: 160 base rows, 32 GiB, chunk 16, prefill 128, 3% hot allocation, and `hm=4`. Peak footprint was 49,862,381,712 bytes (46.44 GiB). Reported free memory never fell below 36%. The external guard reported `MemoryError: Memory guard stopped the run to preserve desktop headroom`. Neither the 84 GiB external cap nor the 82 GiB inner cap was reached. No completed throughput or miss histogram was produced.

System compressor occupancy rose from 22.73 GiB before the replay to 53.81 GiB at the stop. System swap-outs increased by 83.1875 MiB. These counters cover the whole desktop and do not assign causality to one process. A mid-run process inventory showed substantial browser, Node, virtual-machine, and agent workloads. Their footprints overlap and must not be added as a physical-memory ledger. Qwen was restored and its health endpoint passed. The unchanged full-model replay will not be repeated under these conditions.

Fable recommended separating scatter overhead, destination type, and read contention before adding a pool. The isolated serial read's 0.578 ms median and the preceding model's roughly 2.54 ms mean are different workloads. Their gap motivated a controlled layout comparison, not a promised speedup. See [the initial evidence](results/2026-11-05-required-read-diagnostics.json) and [probe commands](README.md#required-read-concurrency-probe).

The layout probe at `ba42314` compared four paths using one 32-row bank at a time. Each layer supplied eight disjoint expert IDs per variant. Variant order rotated across layers, and destination pages were touched before timing. All 42 layers passed the common admission rule: every variant physically read at least 95% of requested bytes. The probe validated 336 complete sampled rows after timing. It did not execute the model, GPU projections, concurrent reads, or the refinement worker.

| Component path | Median read | Mean read | Median read plus copy |
|---|---:|---:|---:|
| Existing 24-plane scatter | 2.688 ms | 8.904 ms | 2.688 ms |
| Contiguous private buffer | 0.640 ms | 0.950 ms | 0.640 ms |
| Contiguous shared buffer | 0.721 ms | 0.990 ms | 0.721 ms |
| Private read, then bank copy | 0.683 ms | 0.971 ms | 0.881 ms |

The staged copy itself took 0.095 ms median. Medians of separate intervals do not add to the median combined interval. The staged path reduced mean read-plus-copy time in every layer. Its median paired ratio to scatter was 0.277. Scatter had a long tail: its 90th-percentile read was 25.66 ms. That tail differs from the preceding model workload, so component averages must not replace model timings in a throughput prediction.

Confidence is high in the observed component layout difference and moderate in attributing the model's remaining read delay to the same mechanism. The shared-buffer control stayed near the private-buffer control, which supports scatter overhead as a contributor. Staged reads reuse one private buffer, while the private control uses eight. Bank-slot alignment also differs. These limits matter for small differences, and the probe does not measure contention with GPU execution or refinement reads.

Both independent Max-account Fable/high reviewers approved the probe and a test-only follow-up before launch. All 69 runtime tests and five supervisor tests passed. A proposed cleanup error was not reproduced by the requested OS-read failure fixture; the reviewer withdrew it. No runtime cleanup change was made. The probe's lifetime peak was 197,886,816 bytes (188.72 MiB). Pressure remained at level 1 with at least 70% reported free memory. System swap-outs increased by 512 KiB, and Qwen recovered. This result motivated the optional staged-read implementation below. See [the full layout evidence](results/2026-11-05-required-read-layout.json).

### Staged required base reads

Source `c2142ed` adds optional staged required reads without changing quantized bytes or GPU bank layout. All 42 banks share one 16 MiB CPU buffer. A lock protects the entire read-and-copy interval. The bank still completes previous GPU use before replacing a slot and publishes the slot only after copying succeeds. Required reads remain synchronous; the existing refinement worker remains asynchronous. The default stays `scatter`.

Both independent Max-account Fable/high reviewers approved real-weight verification followed by the guarded comparison. All 73 runtime tests, six HTTP checks, four cold-dispatch checks, and five supervisor tests passed. The HTTP check initially failed because its optional dependency directory was absent from `PYTHONPATH`; it passed with the existing dependencies. No packages changed. Real-weight verification matched 36 finite GPU projections on layers 3 and 44, including eviction and reload. This exercises 48 routed visits, or 40 unique experts, per layer through staged fill. The separate 2,852,397,504-byte archive comparison uses `read_payload`, not staged fill. Verification peaked at 525,075,464 bytes, retained normal pressure, added no swap-outs, and restored Qwen.

The four model trials used ABBA order and identical 128-token forced input sequences. Every trial used 128 base rows, a 32 GiB upper budget, 16-token chunks, 128 prefill rows, and 3% hot allocation within 4 GiB. The predictor retained `hm=4`. Base prefetch and the cold fast path stayed off. Reducing the banks from 160 to 128 rows removed about 6.16 GiB of logical base storage. The external cap tightened to 48 GiB. The inner cap remained 82 GiB, so the external watchdog provided the lower footprint limit.

| Trial | Decode rate | First token | Required blocking | Blocking per miss |
|---|---:|---:|---:|---:|
| Scatter 1 | 3.319 tokens/s | 50.37 s | 20.223 s | 2.452 ms |
| Staged 1 | 5.390 tokens/s | 42.52 s | 6.699 s | 0.813 ms |
| Staged 2 | 5.229 tokens/s | 42.48 s | 6.715 s | 0.820 ms |
| Scatter 2 | 0.711 tokens/s | 133.04 s | 89.010 s | 10.851 ms |

The staged median was 5.309 tokens/s. Confidence is high in the recorded results and moderate in generalizing the gain. The second control is a substantial outlier: its forward-to-route interval also increased to 83.43 seconds, versus 13.63–14.78 seconds in the other runs. Normal pressure and zero swap do not explain that slowdown. Saved workload inventories and VM counters do not establish its cause. Do not present the aggregate percentage difference as a reliable speedup estimate or discard the second control.

Required blocking includes read, copy, and lock wait. Staged copies took 1.499–1.529 seconds per decode, and lock waits totalled 7–8 ms. Both completed histograms passed call-count, miss-count, and read-time identities. These identities also passed for both controls. Histograms contain read time only, so they are not cross-mode blocking totals. The first control spent 79.2% of read time in selections with at least two misses.

Requested base traffic remained 40.33–40.60 GB, with another 7.58–7.99 GB of published refinements. Decode process disk reads measured about 36.5 GB in staged mode versus 46.5 GB in scattered mode. Cache effects therefore also contribute to the comparison. The process and asynchronous publication counters have slightly different boundaries. Do not attribute the entire model improvement to scatter-call overhead alone.

All four trials retained pressure level 1, at least 35% reported free memory, and peaks below 40.71 GiB. They added zero system swap-outs and restored Qwen. Staged actual refined-call coverage was 19.68–19.73%, versus 19.94–20.02% in the controls. Ready salience was 38.27–38.30%, versus 38.61–38.80%. Wanted-call delivery fell to about 96.5%, versus 97.9–98.2%. Faster execution can change refinement readiness, so the next quality check must use the staged profile.

Both staged trials agreed with the first control on 128/129 greedy predictions. The second control agreed on 127/129. Readiness masks agreed on only 6/128, 9/128, and 10/128 steps respectively. The replay does not score autoregressive answer quality or establish identical precision schedules. The 15–30 tokens/s target remains unmet. See [the protocol, results, validation, and review evidence](results/2026-11-05-native-staged-base.json).

The subsequent twelve-case autoregressive quality suite stopped during its first case at pressure level 2. Its peak was 40,699,623,736 bytes (37.90 GiB), below the 48 GiB watchdog. System compressor occupancy rose from 19.43 to 56.91 GiB, while reported free memory remained at least 44%. No system swap-outs were added. Qwen recovered, and the remaining eleven cases did not launch. There is no completed answer or quality score from this attempt. The existing log does not identify the last model phase or layer. These system counters cannot attribute all compressed pages to the model. The unchanged profile will not be repeated. See [the failure receipt](results/2026-11-05-native-staged-quality-stop.json).

The optional diagnostics add flushed memory checkpoints and separate host timing intervals. The later trials at `f672982` used 64 base rows, eight-token chunks, staged reads, and 3% hot allocation under a 40 GiB outer watchdog. The ordinary fixed-input replay measured 4.59 tokens/s. Instrumented coding and synchronized replay measured 4.11 and 3.74 tokens/s respectively. The coding answer passed HumanEval/3. All three trials retained normal pressure, added no swap-outs, and restored Qwen. Peak footprint stayed below 28.23 GiB. The owner waived independent review before those trials. See [the completed diagnostic evidence](results/2026-11-05-native-diagnostics.json).

Confidence is high in these recorded measurements, but the intervals are host wall times, not pure GPU execution times. Synchronization, graph scheduling, allocation, and instrumentation affect them and can change refinement readiness. They do not establish a hard throughput ceiling or justify adding overlapping intervals. One passing coding case does not establish broader quality. See [diagnostic semantics](README.md#optional-execution-diagnostics).

### Anticipatory base reads

Source `0c30ab1` adds an opt-in native worker for predicted base rows. It reuses committed jT/EMA scores without changing the refinement policy. The worker reads private CPU buffers. The main thread installs only required rows after the previous GPU use completes. Missing predictions retain the synchronous fallback. Two independent Max-account Fable/high reviews approved the corrected implementation before testing. All 44 runtime unit tests and six server boundary tests passed.

The fresh control and `--base-prefetch 1` trial used 192 rows, a 40 GiB limit, the same 29-token prompt, and 128 identical forced inputs. Both used eight-token chunks, 160 prefill scratch rows, and 2% hot allocation. Each profile ran once.

| Measurement | Synchronous control | One predicted base per layer |
|---|---:|---:|
| Decode tokens/s | 3.558 | 3.452 |
| Prefill to first token | 52.13 s | 53.10 s |
| Required base misses | 5,437 | 5,454 |
| Required misses served by prediction | 0 | 576 |
| Synchronous base reads | 5,437 | 4,878 |
| Synchronous base read time | 16.79 s | 14.44 s |
| Worker base read time, overlapping | 0 s | 16.29 s |
| Required wait for predictions | 0 s | 0.027 s |
| Copy into GPU bank | 0 s | 0.135 s |
| All base-read traffic | 24.93 GiB | 47.01 GiB |
| Peak process footprint | 51.31 GiB | 51.37 GiB |
| Minimum reported free memory | 32% | 32% |

The prediction worker completed 5,375 reads: 576 were useful and 4,799 were discarded. One queued prediction was cancelled, and no predictive read failed. Completed prediction precision was 10.72%, covering 10.56% of required misses. Peak staging reservation was 75.63 MiB under the 128 MiB limit. This reservation overstates the private payload allocation and is not an additional process-footprint measurement. End-of-token draining took 0.00066 seconds in total. No prediction crosses a token boundary.

This trial reduced blocking read time but nearly doubled base traffic and lowered throughput by 3.0%. The feature remains disabled by default. Widening this predictor's base-read window is not supported by these results. Refinement salience priorities are not demonstrated to predict imminent base misses well. The route window also rose from 15.70 to 18.56 seconds, but it includes upstream GPU work and does not isolate CPU cost.

Both runs retained normal pressure, added no system swap-outs, and restored Qwen. They agreed on 127 of 129 greedy next-token predictions under teacher forcing. Readiness-dependent refinement precision and desktop activity remain confounders, so this is not a deterministic numerical-equivalence claim. Refined-call coverage was 15.75% versus 15.72%; salience coverage was 31.05% versus 30.98%. See [the complete results, cancellation accounting, and isolation receipts](results/2026-11-05-native-base-prefetch.json).

The autoregressive HumanEval/3 check passed the network-disabled grader and emitted the same 149 tokens as the earlier synchronous run. Prediction measured 3.86 tokens/s versus the earlier 4.62 tokens/s, with a 51.96 GiB peak. Only 407 of 6,248 completed predictions were used, or 6.51%. The other 5,841 were discarded; ten queued reads were cancelled. This is a single coding case and a comparison across different runs. Pressure remained normal with at least 31% reported free memory, and Qwen recovered. System swap-outs increased by 32 pages, or 0.5 MiB, across the run and cleanup.

Reproduce the comparison with the same `--fixed-json` trace and fresh output paths. Run the 192-row profile once without `--base-prefetch`, then once with `--base-prefetch 1`. Keep the host watchdog active. `read_seconds` and `read_bytes` remain synchronous counters. `prefetch_read_seconds` overlaps computation. Do not add it to wall time or the synchronous read duration.

### Mixed prefill chunk widths

Each row uses the same 34-token prompt, 160 base rows, and a 160-row four-bit scratch bank. Standard prefill runs first. Layer-major prefill runs in the same process after predictor reset, with the base banks already populated. The times include that difference in base residency. Within each width, logits, every cache array, and committed predictor state matched exactly between schedules. The table does not compare logits across widths.

| Chunk width | Standard prefill | Layer-major prefill | Identical cache arrays |
|---|---:|---:|---:|
| 4 | 105.98 s | 45.85 s | 145 |
| 8 | 92.54 s | 45.80 s | 145 |
| 16 | 79.18 s | 44.70 s | 167 |

### Native MTP experiment

The base-only target used 160 rows per layer, the prepared affine-8 backbone, a 19-token prompt, and greedy sampling. The verifier used the checkpoint's actual draft layer with affine-4 prepared weights. The decode measurement includes the draft prompt state and proposal, verification, and rollback work. Weight preparation took 0.91 s for one draft and 1.16 s for two. Each run emitted the same 24 tokens.

| Decode | Tokens/s | Accepted proposals | Accepted per verification round | Peak process footprint |
|---|---:|---:|---:|---:|
| Serial | 2.27 | N/A | N/A | 33.39 GiB |
| One draft | 2.55 | 10/13 | 0.77 | 38.68 GiB |
| Two drafts | 1.38 | 13/20 | 1.30 | 38.99 GiB |

One draft gained about 12% on this short run. Two drafts slowed it by about 40%. The results validate both acceptance and rejection on one greedy trajectory. They leave mixed-precision speculation, sampled-distribution agreement, and sustained throughput open.

### Loopback prefix experiment

The mixed prefill checks compared logits and complete model and predictor states. An identical 19-token prompt reused all 19 tokens. The shared-suffix case reused 16 and computed three. An equal-length prompt with different tokens missed. The 15-token request also missed the 16-token boundary. The original request recomputed after eviction. All eight checks logged results. The external deadline stopped the child before it wrote the consolidated report. A shorter eviction-only rerun completed normally and preserved exact state across both evictions.

The base-only HTTP test rendered an actual system message and tool schema. Its repeated request returned identical output. These times measure model time to first token, before the non-streaming HTTP response completes.

| Request | Rendered tokens | Reused tokens | Computed tokens | Model time to first token |
|---|---:|---:|---:|---:|
| Cold | 180 | 0 | 180 | 52.88 s |
| Identical | 180 | 180 | 0 | 4.54 ms |
| Shared suffix | 179 | 160 | 19 | 1.54 s |

## Grouped expert shard reads

The reader groups adjacent tensor ranges for base loads, synchronous upgrades, and background refinements. All 1,536 sampled expert tensor comparisons matched byte for byte. The test covered three layer 3 experts across eight shards. The sampled read-call counts fell from 408 to 216 for base loads, 360 to 192 for refinements, and 768 to 72 when loading all fields. The separate nine-tensor backbone comparison also matched. The storage and async regression checks passed. See the [real-weight checks](results/grouped-expert-io.json).

The fixed-route microbenchmark delivered all 336 refinements. Summed worker read time fell from 6.55 to 6.01 seconds. Median expert cycle time during delivery rose from 170 to 243 ms. The newer run completed 18 cycles during delivery, versus 31 in the previous run. These are separate runs. The component results show lower read time and slower concurrent cycles in the measured samples. See the [grouped microbenchmark](results/grouped-microbench-42layers.json).

Both full-model comparisons used the same checkpoint, prompt, output length, warm-up, and runtime settings. These settings include a 32 GiB base cache, a 4 GiB refinement budget, and a 2% hot allocation. The grouped comparison used clean source `b5c37f4`. See the [three completed requests](results/2026-11-05-grouped-2pct-128.json).

| Measurement | Individual reads | Grouped reads |
|---|---:|---:|
| Decode tokens/s, median | 1.217 | 1.638 |
| Decode tokens/s, range | 1.110–1.296 | 1.575–1.648 |
| Time to first token, median | 80.55 s | 61.64 s |
| Decode base-load time, median | 55.30 s | 34.31 s |
| Decode base-cache misses, median | 4,496 | 4,358 |
| Base-load time per miss, median across requests | 12.30 ms | 7.83 ms |
| Peak process footprint | 59.11 GiB | 59.96 GiB |
| Minimum reported free memory | 67% | 62% |

Generated sequences differed within both comparisons. Per-miss timing controls for the number of misses, while the expert sequence and storage conditions can still differ. Base loads consumed about 44% of measured decode time in the grouped runs. Subtracting only their time from the observed steps gives a 2.90 tokens/s median, assuming all other timings stay fixed. This calculation has limited predictive value. It shows the size of the remaining gap.

The first grouped attempt completed one request at 1.559 tokens/s. The watchdog stopped the second at pressure level 2, with 39% reported free memory and a 59.96 GiB process peak. DeepSeek recovered after the stop. The retry retained the same guards, ran at normal pressure, and restored DeepSeek. The reported medians use only the three completed retry requests. See the [pressure-stop receipt](results/2026-11-05-grouped-pressure-stop.json).

The 2% allocation delivered 16.25% actual hot-call coverage, 31.44% actual salience coverage, and 98.22% of wanted calls in time. Across 336 overlapping windows, top-budget forecasts assigned 34.98% salience and observed 32.88%. Raw jT rankings covered 49.14% of observed salience, versus 28.61% for EMA and 5.61% for the initial static mask. These are repeated-prompt diagnostics. The [three answer checks](results/2026-11-05-grouped-smoke.json) and 28 unit tests passed. Max-account Fable/high reviewed the reader before testing.

## Banked expert projection prototype

The banked Metal projection dispatches eight routed experts together for each of the gate, up, and down projections. The component benchmark packs the resident tensors before timing. It uses different inputs for each expert and compares against the existing serving implementation. Across four cases, relative RMS error was at most 0.00015%. The existing serial path also passed the independent CPU comparison at both precision levels. See [timings and isolation receipts](results/banked-expert-projections.json).

| Layer | Expert precision | Serial median | Banked median |
|---|---|---:|---:|
| 3 | 1.5-bit base | 1.80 ms | 1.29 ms |
| 3 | 4-bit refined | 2.38 ms | 1.85 ms |
| 44 | 1.5-bit base | 1.85 ms | 1.33 ms |
| 44 | 4-bit refined | 2.47 ms | 1.90 ms |

The benchmark excludes SSD reads, packing time, attention, and the predictor. The native generator uses persistent bounded base and refinement banks. These component timings measure expert computation and do not establish model tokens/s.

The device route variant produced identical results to the host route in all four component cases. An out-of-range device ID produced a nonfinite result in its own row while valid rows stayed finite. The check uses a bounded gather without waiting for the host to read the IDs. The serial path passed its independent CPU comparison after the alternating-mask offset change. See [device-route results](results/device-expert-projections.json).

### Compiled banked projection

A later probe compiled the existing banked expert function with all variable arrays passed explicitly. The immutable layout permutation remains a captured constant. Compiled and eager outputs matched exactly in all four cases, including changed scales, sign planes, and input vectors. Invalid device routes retained their NaN checks. The compiled agreement threshold was 1e-5 relative RMS.

| Layer and precision | Eager device route | Compiled | Relative speed |
|---|---:|---:|---:|
| Layer 3, base | 1.117 ms | 1.155 ms | 0.968x |
| Layer 3, refined | 1.788 ms | 1.718 ms | 1.041x |
| Layer 44, base | 1.125 ms | 1.131 ms | 0.994x |
| Layer 44, refined | 1.806 ms | 1.720 ms | 1.050x |

Compilation did not improve the base path and remains outside generation. Each timing follows five warmups and reports the median of 25 iterations. The first compiled call is recorded separately. Peak process footprint was 0.82 GiB under a 12 GiB external guard. Pressure remained normal, swap-outs did not increase, and Qwen recovered. See the [reviewed microbenchmark evidence](results/2026-11-05-compiled-banked-probe.json). These timings exclude model routing, loading, and backbone execution.

## Full-model validation after the reader correction

The [individual-read comparison](results/2026-11-04-preadv-2pct-128.json) used clean source `af4516d`, a 2% hot allocation, 32 GiB base cache, and 4 GiB refinement/staging budget. It retained the 84 GiB process guard, eight-token prefill chunks, and no memory wiring. One 16-token warm-up preceded three measured 128-token requests with the same interval-merging prompt as the earlier comparison.

| Measurement | Observed result |
|---|---|
| Decode throughput | 1.217 tokens/s median, 1.110-1.296 range |
| First-token latency | 80.55 seconds median, excluding startup |
| Peak process footprint | 59.11 GiB |
| System pressure | Normal throughout; minimum reported free memory 67% |
| Base-load time during decode | 51.23-60.23 seconds per request |
| Predictor diagnostic overhead | 0.18 seconds in the first request, included in decode timing |

The earlier same-configuration comparison measured 1.063 tokens/s median. Generated sequences differ within both comparisons because precision depends on readiness. These observations are not a fixed-token replay or a controlled estimate of the reader's speedup. Synchronous base loads still consume about half of decode time. Runtime logs also report reference PyTorch fallbacks for `causal_conv1d_update` and `fused_recurrent_kda`.

The corrected runtime passed [all three answer checks](results/2026-11-04-preadv-2pct-smoke.json), with a 53.55 GiB peak. It generated a complete [palindrome function](results/2026-11-04-preadv-2pct-coding.json) that passed 12 explicit cases and 200 seeded ASCII cases. That request generated 32 tokens, took 246.60 seconds to the first token, and peaked at 57.31 GiB. This validates one small coding task, not a general coding evaluation. All isolated runs restored the existing service.

## Controlled hot-pool comparison

These measurements used clean source `1638255`, before the positional-reader correction. Each setting passed the exact-reply, arithmetic, and ordering smoke checks.

| Hot pool | Median decode tokens/s | Range | Median first-token latency | Peak process footprint | Executed / desired hot routes |
|---|---:|---:|---:|---:|---:|
| [1%](results/2026-11-04-async-base32-1pct-16.json) | 2.392 | 2.057-2.512 | 62.18 s | 48.16 GiB | 11.3% / 11.5% |
| [2%](results/2026-11-04-async-base32-2pct-16.json) | 1.956 | 1.942-2.059 | 64.06 s | 48.25 GiB | 18.8% / 19.4% |
| [4%](results/2026-11-04-async-base32-4pct-16.json) | 1.326 | 1.174-1.440 | 70.15 s | 48.35 GiB | 29.1% / 31.2% |

All configurations used a 32 GiB base cache, 4 GiB refinement/staging budget, eight-token prefill chunks, asynchronous refinements, and no memory wiring. Each process discarded one 16-token warm-up and measured three identical-prompt requests of 16 tokens. Decode throughput includes 15 decode steps and excludes prefill and the first generated token. First-token latency excludes model startup.

The hot-pool percentage counts allocated experts, not the share of routed calls. The predictor concentrates the pool on frequently selected experts. Readiness changes the precision actually executed.

These short requests caused no base-cache evictions. Their repeated working sets fitted within the cache. The outputs are truncated coding responses, so they do not establish code correctness or sustained throughput. All three processes stayed at normal memory pressure and restored the existing service.

## Longer generation exposes base-cache misses

The [2% 128-token comparison](results/2026-11-04-async-base32-2pct-128.json) used the same source and configuration. It discarded one 16-token warm-up, then measured three 128-token requests.

| Measurement | Observed result |
|---|---|
| Decode throughput | 1.063 tokens/s median, 1.016-1.065 range |
| First-token latency | 89.80 seconds median |
| Peak process footprint | 58.29 GiB |
| Base evictions during decode | 2,453 / 4,316 / 4,499 |
| Synchronous base-load time | 59.09 / 58.90 / 62.64 seconds |
| System pressure | Normal throughout; minimum reported free memory 67% |

The base cache stayed within 32 GiB while eviction operated in the full model. Synchronous base misses consumed about half of decode time. Generated token sequences differed across requests, consistent with readiness-dependent precision. This is not a fixed-token replay or a quality evaluation.

## Background reads held the interpreter lock

The [initial microbenchmark](results/microbench-async-42layers.json) used eight real experts in each of 42 layers. It confirmed 336 base fallbacks and identical outputs while the read worker was deliberately blocked. Actual read delivery nevertheless slowed the expert cycles.

[Safetensors 0.8](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs) performs its NumPy `pread` inside a Python-attached section in `get_tensor_pread`. The corrected worker uses bounded `os.preadv` reads, which release the interpreter lock. At this stage, synchronous prefill still used the original shared Safetensors handles.

| Reader | Resident base cycle | Resident hot cycle | Median cycle during delivery | Total delivery | Peak process footprint |
|---|---:|---:|---:|---:|---:|
| Safetensors, `2500a97` | 106.43 ms | 149.17 ms | 1,031.05 ms | 5.88 s | 6.47 GiB |
| Positional reader, `bc11e13` | 95.49 ms | 120.37 ms | 170.46 ms | 6.68 s | 7.31 GiB |

The [corrected run](results/microbench-async-preadv-42layers.json) completed 31 expert cycles during delivery, versus six previously. It loaded all 336 refinements within the staging budget. Delivery itself took longer; the observed benefit was reduced interference with execution. These are separate runs, not a statistical estimate of a universal speedup.

The microbenchmark excludes attention, the predictor, base misses, and autoregressive token generation. Its cache setting is 256 MiB, versus 1 GiB in full-model runs. Delivery includes first-use kernel compilation and main-thread installation. Wall-clock overlap does not prove simultaneous GPU occupancy. No component timing is presented as model tokens/s.

The corrected reader passed [real-weight async checks](results/preadv-async-correctness.json), [synchronous storage regression](results/preadv-resident-store-correctness.json), and 22 unit tests. Fable/high reviewed the correction. Earlier changes also received Mythos/high review; the operator subsequently selected Max-account Fable/high for remaining reviews.

## Historical baseline and limits

The [historical canonical synchronous benchmark](results/2026-11-04-canonical-cache4.json) reached 0.227 tokens/s median, 131.03-second first-token latency, and 81.67 GiB peak. It used full base residency, a 21.74% hot pool, mmap, wiring, and different chunking. Its difference from later results cannot be attributed solely to asynchronous reads.

The full-base mmap and pread trials stopped under pressure at 42.89 GiB and 78.55 GiB. Neither generated an answer. Their failure receipts remain linked from [the memory procedure](ASYNC.md).

Confidence is high for the recorded measurements and tested correctness checks. General model quality, long-context operation, and full-model acceleration from Neural Accelerators or the Neural Engine remain unverified. The [native MTP experiment](#native-mtp-experiment) is separate from these measurements.

## Accelerator diagnostics

Two isolated local probes tested accelerator paths without changing production inference. Each used a 12 GiB parent guard, remained at normal system pressure, and restored the existing service.

The [TensorOps probe](results/tensorops-projection-timing.json) evaluated one real layer-3 gate projection, with 20 timed samples per variant. At batch size one, the existing fused projection took 0.607 ms median. Reconstructing the dense weights and executing Metal TensorOps took 0.909 ms. Keeping dense weights cached reduced this to 0.252 ms; ordinary MLX with the same cached weights took 0.277 ms. The cached variants exclude reconstruction and require additional weight memory. Their small difference includes dispatch overhead and differing output precision. It does not establish a model-level accelerator speedup. Batch sizes 3, 8, and 32 also passed numerical checks. The maximum TensorOps relative RMS error was 0.000061% on these cases.

The [Core ML probe](results/coreml-predictor-timing.json) converted the actual jT predictor and evaluated one step with 32 cached positions. PyTorch CPU took 3.347 ms median, Core ML CPU took 1.359 ms, and Core ML CPU plus Neural Engine took 1.331 ms. The compute plan preferred the Neural Engine for 389 operations and CPU for 16; another 561 had unknown placement. This is anticipated placement, not hardware-counter evidence. The Neural Engine configuration preserved all top-eight choices on the synthetic input, with 0.258% relative RMS error in probability values. That single input does not validate recurrent drift, full-window operation, or model quality. Core ML tools 9.0 warned that the installed PyTorch 2.15 version was untested.

The measurements do not justify integrating either probe into the production path yet. (The Core ML predictor was later integrated as an opt-in backend; see the next section.)

### Core ML predictor backend (opt-in)

`--predictor-backend coreml` (or `NQ_PREDICTOR_BACKEND=coreml`) runs jT through [coreml_predictor.py](coreml_predictor.py). PyTorch remains the default. The package is a stateful fp16 ML program with one function each for 1, 8 and 128 rows. The functions share their weights and a 256-slot key/value ring per block, which is held as Core ML state. The host computes the token embedding, a dense routing-share matrix, the RoPE angles, the mask and a one-hot ring write. The model computes everything else. The compute plan places every operation of the 1-row function on the Neural Engine. On this macOS, Core ML cannot build a CPU-only plan for the stateful program, so it supports only CPU_AND_NE (the default) and ALL. Prefix checkpoints read and write the ring through the existing `net.cache` interface.

The [offline replay](results/coreml-predictor-ane-offline.json) used [verify_coreml_predictor.py](verify_coreml_predictor.py). It replayed the routing recorded from a real six-prompt DFlash run: 312 preforwards and 310 commits. Mean per-layer Jaccard between the wanted sets was 0.994, and 87% of layer-steps were identical. As a sensitivity floor, PyTorch with 0.3% multiplicative noise gave 0.992 and 82%. The hysteretic policy amplifies tiny score differences at the budget boundary. Block-head relative RMS error was 0.22%, and the maximum absolute error was 0.0028. Top-budget overlap was 99.86%, and top-8 overlap was 99.81%. The wanted set covered 29.51% of routed calls (47.26% of salience) with Core ML and 29.50% (47.22%) with PyTorch.

Network latency on an M5 Max, median wall / process CPU, with PyTorch on 4 threads:

| Call | PyTorch CPU | Core ML CPU+NE |
|---|---|---|
| 1 row, full window | 4.33 / 5.03 ms | 1.91 / 0.38 ms |
| 8-row commit | 11.19 / 13.38 ms | 2.23 / 0.57 ms |
| 512-row prefill (4 x 128) | 135 / 212 ms | 24 / 9.9 ms |

The [online comparison](results/coreml-predictor-ane-online.json) ran two runs per backend of `native_dflash.py --prompt-dir /tmp/codebench --blocks -8 --batched-prefill --tokens 192`. Predictor wall time during decode fell from 6.5 s to 2.6 s. Whole-process CPU during decode fell from 40.4 s to 33.8 s, or from 1.30 to 1.06 cores. Aggregate decode speed was 33.25 and 32.94 tok/s with PyTorch, and 30.79 and 34.20 tok/s with Core ML. Ready-call fractions were 0.309/0.303 with PyTorch and 0.305/0.309 with Core ML. The predictor hook already overlaps GPU work, so decode speed is unchanged within run-to-run noise. Different readiness changes which experts run at 4 bits, so generations diverge after a few tokens; this noise dominates. Core ML also releases the GIL for most of a call. Against a competing Python thread, a one-row PyTorch call took 326 ms, and a Core ML call took 5 ms. The packed TensorOps path regressed single-token latency, while Core ML CPU and Neural Engine timings were close. Confidence is high in these recorded observations and low in any projected full-model benefit.

## MTP remains a separate implementation task

The upstream [Spark runtime](../spark/flash/README.md) reports 21.247 tokens/s for one sampled MTP2 probe on its NVIDIA path. The Mac reference decoder accepts one row, and its installed Transformers backbone has no MTP execution path. The native base-only verifier runs the real draft and cache rollback. Mixed MTP still needs refinement scheduling and predictor commits for accepted rows.

Neither accelerator probe executes MTP. The fixed-expert microbenchmark also excludes it. A 20-30 tokens/s target on this Mac remains unverified; enabling a configuration flag cannot supply the missing runtime work.

## Effective hot coverage and predictor calibration

The published [Spark preset](../spark/flash/run_spark.sh) allocates 2,630 refinement slots using its per-layer distribution. The native Mac experiment scales that distribution to 242 slots with a minimum per layer and integer rounding. Both configure a 20% fixed subset, 75% jT plus 25% exponential moving average, one-token refresh, retention multiplier (1 + 4), and a 64-token EMA half-life. The Mac uses the release-derived [committed predictor](../spark/flash/sampled_ahead_sentinel/jt_runtime.py). It advances the known target token before its forward and commits its routing only after evaluation. Spark's [sampled-ahead runtime](../spark/flash/sampled_ahead_sentinel/runtime.py) also carries the next sampled token ahead and commits accepted verification rows through a ledger.

Spark allows up to 64 pending throughput prefetches and uses separate spare slots. The Mac experiment allows two pending reads on one worker with a 64 MiB staging limit. Both publish only complete refinements and fall back to base precision when a refinement is unavailable. The Mac retains existing rows until a replacement can be installed, with the per-layer budget intact. These differences in capacity, dispatch, and publication prevent a direct comparison of quality or effective hot coverage.

Allocation counts slots among 12,096 experts. Effective hot coverage counts routed expert calls that actually execute at 4-bit. Desired coverage counts routed calls inside the predictor's requested set. These denominators differ.

The [historical intersection analysis](results/effective-hot-historical.json) separates delivery misses from cached hits outside the requested set. In each short 2% run, 976 of 5,040 routed calls were wanted; 934 of those were ready. Another 12 calls used previously cached refinements outside the wanted set. Thus actual coverage was 946/5,040, or 18.77%, and delivery covered 934/976, or 95.70%, of wanted calls. Subtracting desired and actual coverage alone would hide those 12 cached hits.

Generation reports now include `predictor_diagnostics`. They record per-layer call counts, normalized squared-weight salience coverage, and same-budget jT, exponential-moving-average (EMA), and static-mask comparisons. Those baselines rescore the observed routing; they do not simulate different generated sequences. Metrics time is included in throughput and reported separately.

Calibration compares each causal block forecast with the mean normalized squared routing weights over that token and the next 15 decode steps. This matches the shipped predictor's `train.py` block-loss target. The final 15 incomplete windows are excluded. Reports include total variation, cross entropy, probability-bin means, and predicted versus observed top-budget salience. Windows overlap and are not independent samples. A short 16-output-token run contains only 15 decode steps, so it has no full calibration window.

The individual-read comparison provides 381 decode steps and 128,016 routed expert calls. The following results pool those calls and exclude the warm-up.

| Measurement | Observed result |
|---|---:|
| Allocated slots | 242 / 12,096, or 2.0007% |
| Desired hot-call coverage | 16.37% |
| Actual hot-call coverage | 16.20% |
| Wanted calls delivered in time | 97.60% |
| Desired salience coverage | 32.21% |
| Actual salience coverage | 31.67% |
| Raw jT top-budget salience coverage | 49.85% |
| EMA top-budget salience coverage | 28.87% |
| Static initial-pool salience coverage | 5.58% |

Of 20,954 wanted calls, 20,451 received their refinement and 503 fell back. Another 282 calls used cached refinements outside the wanted set. Every per-layer total agrees with the existing global counters. The fixed subset occupies 48 of 242 slots. Its mask covered 0.93% of observed salience before accounting for readiness.

The 336 complete, overlapping calibration windows forecast 34.74% salience in their top-budget experts and observed 33.14%, a 1.59 percentage-point overestimate. Mean cross entropy was 4.387 nats, versus 5.663 for a uniform distribution. Mean total variation was 0.549, so the near aggregate top-budget match must not be mistaken for accurate forecasts for every expert and window.

These measurements support useful jT rankings on this prompt. The final policy captures less salience than raw jT ranking because it also applies EMA, fixed slots, and retention hysteresis. Removing those constraints could increase SSD churn; the ranking comparison does not measure that counterfactual. Confidence is high in the recorded counts and moderate in this policy-tuning lead.

These diagnostics measure routing prediction on mixed-precision execution. Three repeated-prompt requests are not a held-out quality suite. They do not establish answer accuracy, general calibration, or agreement with an FP8/BF16 reference. No predictor parameters were fitted on these requests. The complete palindrome request supplies a second prompt, with only 16 overlapping calibration windows.

### Repeated cache replay

Source `4253efc` adds two fixed-token passes through one loaded model. Expert banks remain allocated, while each pass creates fresh attention state and resets the predictor. Fourteen CPU tests passed, including per-pass counter deltas and evidence retention after failure. The default remains one pass.

Neither full-model trial completed a safe paired comparison. Both used 128 forced tokens, eight-token chunks, staged required reads, the split decoder, and 3% refinement slots. The outer guard remained 52 GiB, with normal pressure and at least 20% reported free memory required.

| Base rows | Completed passes | Initial decode tokens/s | Peak footprint | Guard pressure | Added swap-outs |
|---|---:|---:|---:|---:|---:|
| 160 | 0 | Unmeasured | 50.00 GB | 2 | 0 |
| 128 | 1 | 6.06 | 44.46 GB | 2 | 1.25 MiB |

The 128-row initial pass served 34,771 base-cache hits and required 8,237 misses. Its 3.00% refinement allocation served 19.66% of routed calls and 38.33% of routed salience. Requested refinements were ready for 96.03% of their routed calls. These are observed coverage counters, not an answer-accuracy score.

The second pass stopped under pressure. Its completed first-pass log remained available, but no warm-cache gain is established. Confidence is high in the recorded measurements and unknown for paired throughput or quality. Fable on Max/high returned a usage limit, so no fresh review was obtained. See [the results, hashes, and limitations](results/2026-11-06-native-warm-replay.json).

## Routed-expert window kernels: register pressure is a weak lever (2026-11-14)

`gpu_regs/regs.py` reads each window kernel's GPR count from the `__GPU_METADATA` section of a Metal binary archive. The field was calibrated on synthetic kernels: N live floats read as N + 3. `maxTotalThreadsPerThreadgroup` reports 1024 for every variant, so it gives no signal. Default GPR counts (gate/up pair, down): decode-1 base 46/36, mixed 72/67; verify-2 base 59/50, mixed 78/75; verify-8 base 79/67, mixed 78/75. The refined path sets the count in every mixed launch.

Opt-in variants, all bit-exact (`test_optional_kernel_variants_are_bit_exact`):

- `NQ_GATE_UP_PAIR=0`: separate gate and up launches. This removes the per-thread pointer selects and saves 6–12 GPRs.
- `NQ_SPLIT_PRECISION=static`: split launches compile only their own path. Base rows drop from 72–78 to 45–59 GPRs.
- `NQ_NARROW=1`: 16-bit lane setup. Saves 0–4 GPRs.
- `NQ_WINDOW_TG` = 32–512 threads. GPRs unchanged.
- `NQ_MIXED_BASE_SPECIALIZED`: fewer base copies. Smaller code, GPRs unchanged.

In interleaved layer benchmarks (`bench_expert_kernels.py`, L10/L11, 12 chained layers, 5×10 samples), none of these variants beat the noise. Every variant stayed within ±3% in every scenario, and the combination of all three GPR cuts measured −0.1% to +5%. A timing-only ablation that adds live registers gives a gentle linear slope with no occupancy cliff: +8/+16/+32 GPRs cost +1%/+3%/+8%. At 36–79 GPRs these kernels are limited by instruction issue, not occupancy. The earlier prefetch slowdown (+23%) therefore came from its extra work, not its registers. Data: `results/2026-11-14-register-variants-{tg,structure,padding}.json`.
