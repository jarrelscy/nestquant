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
