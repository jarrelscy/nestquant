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
