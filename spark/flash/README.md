# Flash on one DGX Spark — candidate port

**GB10/aarch64 execution and unified-memory fit are not yet validated.** The Flash
runtime has served on one SM120 96 GiB discrete GPU with host-mapped residuals.
That does not establish one-Spark performance or memory fit. Do not use the old
`spark/start_spark.sh` (two-node b175/jF) to launch Flash.

## Build and run on the Spark

Use a native aarch64 machine with CUDA 13-capable drivers and Docker GPU support.
The build requires aarch64 wheels/build support for PyTorch, FlashInfer and
TileLang. Missing dependencies fail the build; TileLang must not be skipped.
The source is pinned to vLLM `487ecf187d3dfe74d2cf6119a92881dba403c219` with the
included GLM5Next patch, rather than the old full-GLM fork.

```bash
export NQ_FLASH_DATA=$HOME/nq-flash
spark/flash/run_spark.sh build
spark/flash/run_spark.sh fetch
spark/flash/run_spark.sh prepare
# Supply VLLM_API_KEY through your environment; do not put it in logs.
NUM_SPEC=1 spark/flash/run_spark.sh up
spark/flash/run_spark.sh logs
```

`prepare` reads TP8 shards and writes TP1 records one layer at a time, then makes
a separate block-FP8 backbone overlay. Keep the original checkpoint: the overlay
symlinks unchanged files. Preparation is offline and must not compete with serving
for Spark's shared RAM. Reserve at least 400 GB disk until measured locally;
source + TP1 + overlay + build caches coexist. No private calibration data is used.

Defaults: TP1, max_num_seqs1, eager, no prefix cache, FP8 KV, 131072 context,
utilization0.90, one probabilistic MTP token; `NUM_SPEC=2` is supported by the
committed-row ledger and has served on SM120. Native vision support remains.
No synthetic TPS cap on real Spark; `NQ_FLASH_TPS=35` explicitly applies one.
The unchanged published jT policy has zero fixed experts, 102 floating per layer
3–17 and 74 per layer18–44, plus eight physical spare slots overall. Two draft
steps use vLLM's metadata-rebuild fallback; no speed gain is guaranteed.

## Memory: mandatory first-boot verification

On discrete SM120 the runtime measured approximately74.74 GiB loaded device
weights plus27.37 GiB host-mapped slots, before KV, peak activations, predictor,
OS and other runtime allocations. **On GB10 those allocations share RAM.**
The launcher explicitly caps KV at4 GiB (`NQ_KV_BYTES`) instead of assuming the
CUDA memory profiler accounts for host-mapped slots. An explicit KV budget
supersedes utilization-based KV sizing; vLLM must validate that it holds128K.
This is a conservative candidate budget, not a measured guarantee.

Measure process/host peak RAM and CUDA memory during load, prefill, repeated
requests and MTP. The reported research115 GB estimate is not a live measurement.
If128K does not fit, `NQ_MAXLEN=65536 NQ_KV_BYTES=2147483648` is an explicit smaller
context option, retaining102/74. Do not silently reduce floating budgets or claim
the reference KLD for changed configurations. Any alternate allocation's quality
is unmeasured. The reported0.0691 KLD was not reproduced here.

## Issue #1 audit

[Issue #1](https://github.com/jarrelscy/nestquant/issues/1) concerns the two-node,
full-GLM b175 port, not this TP1 Flash path. The tester confirmed a CPU-predictor
new-request reset exception killed its background streaming thread, leaving a
landing wait unsatisfiable. The CPU workaround and reset fixes passed repeated
requests. The later GPU private-stream patch was proposed on
`spark-b175-gpustream` (07e9be9); the issue has no later GB10 confirmation and
remains open. Do not describe RoCE as the verified cause or the private-stream
patch as hardware-verified.

Flash addresses these mechanisms structurally:

- `IncrementalJT` creates a private CUDA stream, waits for predictor initialization,
  and executes all predictor tensor uploads/inference/reset under it. The CPU
  result transfer completes before the policy consumes the prediction.
- `CommittedPredictor.reset` recreates its own NumPy policy state and resets the
  jT cache; no jF-only `b_*` fields or NumPy `.zero_()` assumptions.
- There is no Python background streaming scheduler, distributed op log, or
  per-token landing-wait loop. Executor errors propagate through the request;
  read failures raise. This is fail-fast, not a claim of transparent fail-open
  continuation. Startup's initial pool drain is bounded by300 seconds.
- Slots are released only after mailbox application acknowledgments. Late upgrades
  fall back to the base. Rejected target verification rows never commit predictor
  history. Prefix restores and CUDA graphs are explicitly disabled.
- The image includes both missing SM121 DeepGEMM MQA headers. The 12.x-family
  attention dispatch remains subject to actual GB10 runtime validation.
- TP1 does not need the two-node NCCL/RoCE/verbs fixes. No transport tuning is
  implied by this launcher.

Before calling the port supported: build on native aarch64, run expert numerical
parity and attention checks on sm121, confirm peak memory, run at least20 repeated
requests (short/long/multimodal, MTP1 and2), and exercise read failure handling.
Existing SM120 results and CPU tests do not substitute for these GB10 checks.


## Explicit reduced-residency experiment

`NQ_FLASH_PRESET=spark_128K_74_46` selects zero fixed experts,74 floating per
layer3–17 and46 per layer18–44:2352 active slots, plus the same8 spares.
This runtime-only preset leaves the shipped predictor files and research presets
unchanged. Compared with102/74 it removes1176 slots, saving9,773,481,984 bytes
at8,310,784 bytes/record. The target is110–115 decimal GB combined GPU and host
serving memory on the SM120 simulation. Actual Spark fit and this allocation's
KLD are unmeasured; do not attach the published0.0691 KLD to it.


## U distribution

`NQ_FLASH_PRESET=spark_128K_U_2352` redistributes the same2352 active floating
slots using the U layer allocation. jT still selects experts; there
are zero fixed experts and8 physical spares. `u_distribution.json` records the
source means, mapping and exact budgets. For each layer, weight=reference mean routed
projection payload bpw minus1.5. Normalize these weights to2352 and use largest
remainders (layer ID breaks ties). This is a budget-normalized layer pattern,
with jT selecting whole-expert1.5/4-bit residency at runtime.

Layer3 gets147 slots; layers4/5 get110/105; layers27–38 mostly29–32;
layers41–44 get70/91/87/72. The shipped jT policy, refresh, EMA and causal/rejected
row handling are unchanged. Quality/KLD and native Spark memory fit are unmeasured.
The matching74/46 total-slot baseline measured113.80 decimal GB combined memory
at startup on SM120. Verify U peak usage independently under real workloads.
