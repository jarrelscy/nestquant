# Mac performance measurements

Owner: Andru. Measured: 2026-11-04.

The bounded Mac runtime generates text within the tested memory budget, but interactive latency remains high. A corrected background reader improves a component measurement; full-model throughput after that correction is pending.

Hardware: Apple M5 Max, 40 GPU cores, 128 GiB unified memory, macOS 27.2. Checkpoint: GLM-5.3-Flash NestQuant 1.5/4, revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`.

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

[Safetensors 0.8](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs) performs its NumPy `pread` inside a Python-attached section in `get_tensor_pread`. The corrected worker uses bounded `os.preadv` reads, which release the interpreter lock. Synchronous prefill retains its original shared Safetensors handles.

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

Confidence is high for the recorded measurements and tested correctness checks. General model quality, long-context operation, MTP, and full-model acceleration from Neural Accelerators or the Neural Engine remain unverified. The experimental runtime does not implement MTP.

## Accelerator diagnostics

Two isolated local probes tested accelerator paths without changing production inference. Each used a 12 GiB parent guard, remained at normal system pressure, and restored the existing service.

The [TensorOps probe](results/tensorops-projection-timing.json) evaluated one real layer-3 gate projection, with 20 timed samples per variant. At batch size one, the existing fused projection took 0.607 ms median. Reconstructing the dense weights and executing Metal TensorOps took 0.909 ms. Keeping dense weights cached reduced this to 0.252 ms; ordinary MLX with the same cached weights took 0.277 ms. The cached variants exclude reconstruction and require additional weight memory. Their small difference includes dispatch overhead and differing output precision. It does not establish a model-level accelerator speedup. Batch sizes 3, 8, and 32 also passed numerical checks. The maximum TensorOps relative RMS error was 0.000061% on these cases.

The [Core ML probe](results/coreml-predictor-timing.json) converted the actual jT predictor and evaluated one step with 32 cached positions. PyTorch CPU took 3.347 ms median, Core ML CPU took 1.359 ms, and Core ML CPU plus Neural Engine took 1.331 ms. The compute plan preferred the Neural Engine for 389 operations and CPU for 16; another 561 had unknown placement. This is anticipated placement, not hardware-counter evidence. The Neural Engine configuration preserved all top-eight choices on the synthetic input, with 0.258% relative RMS error in probability values. That single input does not validate recurrent drift, full-window operation, or model quality. Core ML tools 9.0 warned that the installed PyTorch 2.15 version was untested.

The measurements do not justify integrating either probe into the production path yet. The packed TensorOps path regressed single-token latency, while Core ML CPU and Neural Engine timings were close. Confidence is high in these recorded observations and low in any projected full-model benefit.

## MTP remains a separate implementation task

The upstream [Spark runtime](../spark/flash/README.md) reports 21.247 tokens/s for one sampled MTP2 probe on its NVIDIA path. The Mac runtime does not implement multi-token prediction (MTP). Its decode adapter accepts one row, and its installed Transformers backbone has no MTP execution path. The checkpoint includes FP8 MTP expert shards, but their presence does not supply draft execution, batched verification, rejected-token cache rollback, or committed predictor updates.

Neither accelerator probe executes MTP. The fixed-expert microbenchmark also excludes it. A 20-30 tokens/s target on this Mac remains unverified; enabling a configuration flag cannot supply the missing runtime work.

## Effective hot coverage and predictor calibration

Allocation counts slots among 12,096 experts. Effective hot coverage counts routed expert calls that actually execute at 4-bit. Desired coverage counts routed calls inside the predictor's requested set. These denominators differ.

The [historical intersection analysis](results/effective-hot-historical.json) separates delivery misses from cached hits outside the requested set. In each short 2% run, 976 of 5,040 routed calls were wanted; 934 of those were ready. Another 12 calls used previously cached refinements outside the wanted set. Thus actual coverage was 946/5,040, or 18.77%, and delivery covered 934/976, or 95.70%, of wanted calls. Subtracting desired and actual coverage alone would hide those 12 cached hits.

Generation reports now include `predictor_diagnostics`. They record per-layer call counts, normalized squared-weight salience coverage, and same-budget jT, exponential-moving-average (EMA), and static-mask comparisons. Those baselines rescore the observed routing; they do not simulate different generated sequences. Metrics time is included in throughput and reported separately.

Calibration compares each causal block forecast with the mean normalized squared routing weights over that token and the next 15 decode steps. This matches the shipped predictor's `train.py` block-loss target. The final 15 incomplete windows are excluded. Reports include total variation, cross entropy, probability-bin means, and predicted versus observed top-budget salience. Windows overlap and are not independent samples. A short 16-output-token run contains only 15 decode steps, so it has no full calibration window.

These diagnostics measure routing prediction on the mixed-precision execution. They do not establish answer accuracy or agreement with an FP8/BF16 reference. No predictor parameters are fitted on these requests. Full-model observations with these diagnostics remain pending.
