# Current default: 75% jT / 25% EMA

On the `spark-flash` branch, `./start.sh` builds, downloads, prepares and launches
single-Spark Flash. `./start.sh up` reuses prepared artifacts. Set VLLM_API_KEY
privately before launch. Explicit legacy profiles remain `./start.sh up 2-4`
and `./start.sh up spark`; neither is the Flash launcher.

Defaults: flat50 U2630, 526 salience-ranked fixed + 2104 floating experts,
eight spare slots, jT block prediction mix .75, refresh each committed token,
EMA half-life64, hysteresis4, MTP2 probabilistic, FP4 MLA,262144 context,
max_num_seqs1, temperature.7/top_p.95, thinking off. Native pacing is uncapped.
Override predictor settings with NQ_FLASH_JT_PARAMS JSON. The shipped predictor
metadata describes the original training/reference policy; runtime overrides
are logged as nq_flash_jt_parameters. No jMT or single-token urgency is enabled.

Completed80-task/two-repeat SM120 benchmark: TrueScore87.03197, quality84.01199,
calibration94.80574, reliability88.54626, efficiency100, responsiveness66.91436.
This was one RTX PRO6000 plus host RAM,35TPS cap; native GB10 performance and
shared-memory fit remain unverified. These numbers are not a Spark measurement.
Every-token refresh increased observed SSD demand in six-prompt tests to roughly
1.5–2.8GB/s. Monitor delivered salience, selected-but-late salience, per-layer
coverage, outstanding reads, GPU mailbox backlog, host RAM, and emitted TPS.
A faster refresh is not a guarantee that upgrades arrive in time on Spark.

# Flash serving — U2630 + FP4 default

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
spark/flash/run_spark.sh up
spark/flash/run_spark.sh logs
```

`prepare` reads TP8 shards and writes TP1 records one layer at a time, then makes
a separate block-FP8 backbone overlay. Keep the original checkpoint: the overlay
symlinks unchanged files. Preparation is offline and must not compete with serving
for Spark's shared RAM. Reserve at least 400 GB disk until measured locally;
source + TP1 + overlay + build caches coexist. No private calibration data is used.

## Published serving defaults

U2630: zero fixed experts, 2,630 active floating slots with U layer proportions,
plus eight spares. jT selection/policy is unchanged. FP4 MLA cache uses E2M1
with FP16 group scales; DSA indexer stays FP8 and KDA state stays unchanged.
TP1, one sequence, eager, no prefix cache, 262,144 context, probabilistic MTP2,
thinking off, temperature0.7/top_p0.95, and no artificial TPS cap.
The launcher supplies the tested template and sampling defaults; requests can
still override sampling. `NQ_FLASH_TPS=0` is the native default. The SM120 benchmark used an explicit35
tok/s cap; `NQ_FLASH_TPS=35` reproduces that pacing, not native Spark hardware.

Native Spark's initial explicit KV budget is2 GiB (`NQ_KV_BYTES`). Check startup
capacity and actual shared-memory peak before claiming fit. It is not validated
on GB10; the SM120 experiment used profiled1.91 GiB and307,341-token capacity.
Never silently shrink context or expert allocation. Do not add Spark GPU memory
to total host RAM usage: both refer to the same physical shared memory.

Read [SPARK_MONITORING.md](SPARK_MONITORING.md) for exact runtime checks, coverage,
backlog/SSD diagnostics, memory accounting and benchmark reproduction limits.
[FP4_KV.md](FP4_KV.md) records numerical validation and unsupported modes.
Legacy `spark_128K` (102/74, FP8 KV) remains an explicit alternative:
`NQ_FLASH_PRESET=spark_128K NQ_FLASH_MLA_CACHE=fp8 NQ_MAXLEN=131072 NQ_KV_BYTES=4294967296`.
The research KLD0.0691 does not apply to the new default.

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
