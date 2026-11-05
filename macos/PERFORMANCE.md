# Mac performance measurements

Owner: Andru. Measured: 2026-11-04 to 2026-11-05.

Native mixed execution passed one coding check at 4.62 tokens/s. A fixed-token replay reached 3.94 tokens/s, compared with 1.27 tokens/s for the reference path. These are separate workloads. Interactive latency remains high, and the 15–30 tokens/s target remains unmet.

Hardware: Apple M5 Max, 40 GPU cores, 128 GiB unified memory, macOS 27.2. Checkpoint: GLM-5.3-Flash NestQuant 1.5/4, revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`.

## Base cache and native execution

A fixed 128-token replay matched the reference store's 5,641 observed decode base misses at 32 GiB. On the same access trace, a 40 GiB cache would have missed 4,817 times. The trace produced the same miss count at 44, 48, and 56 GiB. This run starts cold. It measures one prompt and holds the observed routes fixed. It does not estimate the memory pressure or routing changes of a live larger cache.

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

In a separate trained layer-zero component probe, median Torch/MPS attention took 1.225 ms. With synchronization around individual fallbacks, the convolution update took 0.213 ms and the recurrent Kimi Delta Attention function took 0.410 ms. Those component windows do not sum to a single ordinary forward. MLX BF16 took 0.756 ms at 0.884% relative RMS difference. MLX affine-8 took 0.587 ms at 1.824%. Affine-4 took 0.450 ms at 19.588% and was rejected. This probe uses 32 recurrent positions and measures one attention layer.

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

The measurements do not justify integrating either probe into the production path yet. The packed TensorOps path regressed single-token latency, while Core ML CPU and Neural Engine timings were close. Confidence is high in these recorded observations and low in any projected full-model benefit.

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
