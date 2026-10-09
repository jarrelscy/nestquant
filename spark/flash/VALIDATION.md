# Candidate validation, 2026-10-08

- 18 CPU tests passed (`tests/test_flash_*cpu.py`) using the public Flash checkpoint
  predictor code. CUDA was disabled. Includes FP4/trellis packing, ring extension,
  cache/policy parity, request reset, all MTP2 rejection boundaries, routing metrics,
  salience, cancellation and bounded delivery.
- `bash -n spark/flash/run_spark.sh` passed. Mock Docker argv checks passed for
  build/fetch/prepare/up; MTP2, TP1, maxlen131072, TPS35 emitted correctly; invalid
  draft count3 rejected. These checks do not establish a successful Docker build.
- Existing local SM120 runtime: repeated API requests and Spark Bench sessions,
  MTP1/MTP2, host-mapped upgrades. Latest MTP2 probe after CPU contention eased:
  25.18 tok/s (128 tokens), 21.97 tok/s (512 tokens), 67.20% draft acceptance,
  2.344 emitted tokens/step; cap35. This is not GB10 performance.
- Issue #1 reviewed through the 2026-10-07 final comment. CPU-reset fix verified
  by reporter; dedicated GPU-stream proposal not subsequently confirmed on GB10.
- No actual DGX Spark available. aarch64 dependency build, SM121 JIT execution,
  unified-memory accounting, full128K occupancy and GB10 throughput remain untested.
- No live serving files, containers, benchmark configuration or model weights were
  changed during this issue audit. Candidate packaging lives in a separate worktree.

Do not describe this release as hardware-validated or attach a measured Spark TPS.

## Prefetch admission experiment, 2026-10-09

The Flash loader counts outstanding residual reads separately from GPU mailbox
acknowledgments and demotions. Total transitions remain bounded at 64; demotions
remain bounded at 8 globally and one per layer. Per-layer 102/74 occupancy and
executor-owned slot release are unchanged. Capacity uses observed peak delivery,
not an EWMA that mistakes under-admission for falling SSD capacity. No new buffers.
This does not implement deferred-publication replacement reads into spare slots.

`NQ_FLASH_WAIT_FOR_UPGRADES=1` is an opt-in diagnostic: before each decode/verify
forward, drain the current jT desired pool. Rejected rows still do not advance jT.
It eliminates delivery misses for that batch's selected pool; it does not eliminate
within-batch predictor refresh lag or guarantee identical routing across MTP modes.
It is off by default and may reduce throughput or increase first-token latency.

21 CPU tests pass with CUDA disabled and the public predictor release. Added checks
cover read admission despite pending demotions, bounded acknowledgments and capacity
retention under low offered load. Actual SM120 API checks: two 1024-token prompts
(bicycle explanation and SQLite inventory CLI), seed42, temperature1, top_p0.95,
thinking off, TPS ceiling35, TP1, no prefix cache, same 102/74 pool. Results:

| Loader / drafts | Prose TPS | Hot / salience % | Code TPS | Hot / salience % |
|---|---:|---:|---:|---:|
| Original / MTP2 | 20.42 | 66.29 / 81.77 | 31.38 | 44.37 / 64.00 |
| Candidate / MTP2, warm repeat | 17.51 | 67.61 / 82.22 | 21.29 | 51.67 / 73.91 |
| Candidate / MTP1 | 17.55 | 70.15 / 83.77 | 20.51 | 55.39 / 78.73 |
| Candidate / MTP2, wait for pool | 16.77 | 72.16 / 86.23 | 21.70 | 58.02 / 80.48 |

Both waiting-mode probes reported zero desired-but-cold routes. Coverage telemetry
samples the committed prefix (1010/993 of 1024 rows in the waiting probes), not an
exact end-of-request snapshot. Latency includes streaming decode; waiting-mode
TTFT was 3.20/4.21 seconds. The baseline was measured on a warm, long-lived server;
candidates restarted. These are short screening probes, not paired fixed-token
replays or statistically established improvements: outputs and routing can differ.
No KLD or end-to-end quality claim, no GB10 measurement, and no production promotion.
The original thinking-off benchmark was stopped at the user's request, with 72/80
scenarios complete; its results must not mix with these subsequent configurations.
Raw local probe records: `/tmp/nestquant/flash-artifacts/prefetch-ab-*.json`.

### Follow-up isolation: CPU scan regression

The first admission patch above introduced excessive scans: its read budget could
remain open while every candidate layer was blocked by pending demotions. It still
sorted/scanned all wanted experts and rebuilt victim lists for each expert. The
follow-up filters ineligible layers first and computes eviction candidates once
per layer. Limits, jT membership and executor-owned acknowledgment remain intact.

A same-process old/new/old test on a 512-token code prompt reproduced 27.76/20.27/
27.43 TPS. Holding the seeded pool fixed gave 32.13 new vs 32.30 old on warm runs,
so the regression required active pool changes. Different generated outputs are
not a bit-exact performance control; all probes used temp0, seed42, MTP2, cap35.

CUDA event and CPU timers isolated the regressed version at 55.23 ms/step in poll
vs 2.04 old; expert GPU execution was 69.66 vs 60.53 ms. The repaired run averaged
about 6.4 ms polling and recovered 26.70 TPS vs 17.72 regressed with instrumentation.
Neighboring old-loader runs were 28.94 and 26.49 TPS. A synthetic blocked-pool CPU
probe fell from 9.73 to 0.050 ms per pump (30 iterations). The old scan, not just
increased 4-bit usage, was the principal cause of this performance regression.

An independent GPU experiment held input tensors, selected expert IDs and 4-bit
weights fixed, changing only residual storage. Across 42 layers with eight distinct
hot experts/layer, batches 1/2/3 took about 56.1/56.1/56.2 ms from host-mapped RAM,
versus 4.82/5.03/5.28 ms from VRAM; base-only 3.02/3.23/3.46 ms. Repeated host runs
agreed. Only 2.60 GiB of selected residuals were copied; tables restored afterward.
Host/VRAM output max-abs differences were 1.90e-5/2.41e-5/2.43e-5; no bit-exactness
claim (atomic reduction scheduling differs). This is expert-kernel timing, not
whole-model TPS or a DGX Spark result. Full diagnostics remain local under
`/tmp/nestquant/flash-artifacts/isolate-*` and `prefetch-ab-isolate-*.json`.

Final strict-wait isolation (same diagnostic process, repaired loader, MTP2,
512-token code prompt): async 27.14 TPS, 54.54% hot, 74.19% salience; strict wait
20.51 TPS, 59.52% hot, 82.20% salience. Explicit drain timers accumulated 7.383 s
across 188 decode steps (39.27 ms/step). Expert GPU time rose from 69.79 to 77.55
ms/step; unique hot experts across 42 layers from 416 to 463. Target forward time
85.03 to 93.60 ms, polling within the target forward 4.35 to 1.29 ms. Waiting
happens before that target-forward timer. Coverage/timing snapshots cover slightly
different prefixes, and generated outputs differ, so components are not an exact
wall-clock accounting identity. This directly distinguishes the repaired CPU bug
from the remaining strict-wait and additional PCIe-weight costs.

Final uninstrumented 1024-token MTP2 strict-wait probes: prose 17.40 TPS;
code 20.95 TPS, 60.49% hot and 82.57% salience, zero desired-but-cold routes.
Strict waits therefore retain a substantial speed/coverage tradeoff even after
repairing the admission scan. No benchmark restart or score mixing was performed.


### Async replacement admission, October 9

Decoupled the demotion window from the minimum read depth: default 32 pending
GPU demotions globally, still at most one per layer and 64 total transitions.
`NQ_FLASH_PREFETCH_MAX_DEMOTIONS` overrides this limit (clamped to half the total
transition limit). No additional allocation or change to jT selection, causal
state, expert budgets, MTP rejection handling, or mailbox-owned slot reuse.
This allows replacement reads across 42 layers to start sooner without drain.
23 CPU tests pass, including blocked layers, cancellation, pool capacity, and the
wider replacement window retaining slots until acknowledgment.

First uninstrumented async run, same 1024-token screening prompts, temperature1,
top_p .95, seed42, MTP2 probabilistic, cap35, thinking off: prose20.04 TPS,
85.40% hot salience (desired86.77%); code27.61 TPS,77.35% (desired79.47%).
Desired-but-cold salience was1.72%/2.67%. These are generated-output probes,
not fixed-routing controls; no universal80% coverage or quality claim follows.
Local raw records: `prefetch-ab-async-window32.json`.

Warm repeat: prose20.39 TPS,84.06% hot salience (desired85.84%); code26.09 TPS,
79.89% (desired81.09%). Telemetry covered1007/981 committed rows respectively
of1024 generated tokens. TTFT0.56/0.64s. Across the two screening runs, code
coverage77.35–79.89% at26.09–27.61 TPS; prose84.06–85.40% at20.04–20.39 TPS.
Repeat records: `prefetch-ab-async-window32-repeat.json`. The server remains
async MTP2, cap35, thinking off; the benchmark remains stopped. Actual Spark
hardware and end-to-end benchmark quality remain untested for this change.


### Reduced residency and U distribution

Added explicit runtime presets `spark_128K_74_46` and `spark_128K_U_2352`.
Both retain2352 active upgrades and8 spares,1176 fewer records than102/74,
saving9,773,481,984 bytes. Published predictor metadata is unchanged.
The74/46 startup gate measured24,955,281,408 host bytes plus88,847,941,632 GPU
bytes =113.80322304 decimal GB. This is a startup measurement, not native Spark
peak-memory validation. Its short run was stopped at user request to switch to U.
U preserves the total pool and jT dynamic selection with nonuniform layer budgets.
25 CPU tests pass, including exact shipped policy/causal-input parity with U
budgets, legacy-presets checks, repeated reset and speculative rejection tests.
Quality/KLD of either smaller allocation remains unmeasured.
