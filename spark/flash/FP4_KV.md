# Experimental Flash FP4 MLA cache

Implemented behind `NQ_FLASH_MLA_CACHE=fp4_g16`; default `fp8` is unchanged.
This implementation was developed separately from the benchmark. Isolated
SM120 GPU numerical checks now pass; end-to-end serving and quality validation
are still pending. It is lossy. The published FP8-KV KLD result does not apply
to it, and isolated attention timings are not serving TPS.

## Format and scope

A NoPE MLA token stores 512 latent values as E2M1 nibbles (low nibble first),
plus 32 little-endian FP16 scales, one per 16 values. Total: **320 bytes/token**.
For each group, store `FP16(max(max(abs(x))/6, 2^-24))`; choose the nearest E2M1
code using that rounded scale, ties to even code. The kernel compares input
magnitudes against midpoint × scale: GPU reciprocal division was found to
misclassify exact halfway cases and is not used. The supported finite range
is ±393024. Inputs outside it are invalid (`TRITON_DEBUG=1` diagnoses them).

The current FP8 sparse-MLA adapter stores **656 bytes/token**: 512 latent bytes,
16 scale bytes and 128 dummy rotary bytes. The new cache removes the dummy
rotary storage because Flash is NoPE. This is a custom layout, **not** the
FlashInfer NVFP4 cache wire format and not an FP4 tensor-core attention kernel.
Sparse attention unpacks only selected rows inside the kernel, uses native
FP16/BF16 tensor-core operands, and accumulates in FP32. No full-context FP16
cache or selected-KV materialization is retained. Scratch is bounded at 16
query tokens × 8 splits × 64 heads × 512 values × 4 bytes = 16 MiB, plus LSEs.

The 11 main MLA layers and the one MTP MLA layer use FP4. The 34 KDA layers'
recurrent state, DSA compressed FP8 indexer cache, and indexer tail buffers
remain unchanged. The existing sparse indexer supplies causal DSA+SWA indices;
attention uses one softmax over that union. It does not maintain predictor or
commit state. Rejected draft entries remain inaccessible through the existing
causal indices, and accepted-position overwrites replace whole records.

The integration stamps only applicable MLA specs with a distinct format ID.
The platform's preliminary one-token MLA sizing probe also uses 320 bytes,
so hybrid attention/KDA page alignment sees the true footprint. Packed byte
views preserve physical page strides, including padding or shared allocation
strides. Converting `slot` to a byte address always uses page and token strides;
flattening the cache as `slot*320` would corrupt a padded allocation. The backend
also accepts the full hybrid manager page (multiples of 64), avoiding virtual
64/256-token splitting across padded page boundaries.

Supported contract: GLM5Next Flash, TP1/DCP1, max_num_seqs=1, eager,
MTP off/1/2, no prefix cache, no KV connector/offload. Unsupported settings
raise at construction. CUDA graph capture also raises in kernel wrappers.
Graph, prefix-transfer and multi-GPU support are deliberately not claimed.
Pack assumes the runner supplies unique, valid slots (negative means padding).
Out-of-range indices are masked and diagnosed under `TRITON_DEBUG=1`.

## Memory accounting

For the actual model config, at 262,144 tokens across 12 MLA layers:

| Raw MLA rows, before page rounding/reserve | Bytes | GiB |
|---|---:|---:|
| Current 656-byte FP8 layout | 2,063,597,568 | 1.921875 |
| New 320-byte FP4 layout | 1,006,632,960 | 0.9375 |
| Difference | 1,056,964,608 | 0.984375 |

About 0.403 GB of the reduction removes dummy rotary padding; the remaining
0.654 GB is the reduction in latent+scale storage. At 8,310,784 bytes per hot
expert record the raw difference equals **127 records**. That is an estimate,
not an allocation promise: block rounding, null/reserved pages, fixed KDA
state, compressed indexer groups, runtime scratch and allocator behavior still
need measurement. A fixed `--kv-cache-memory-bytes` budget will normally buy
more token capacity instead of freeing GPU memory. Lower that explicit budget
only after a full 262K fit/peak test, then increase hot slots separately.

## Activation (after serving and quality validation)

The existing launcher forwards the opt-in variable:

```bash
NQ_FLASH_MLA_CACHE=fp4_g16 NQ_MAXLEN=262144 NUM_SPEC=2 \
  NQ_FLASH_PRESET=<chosen-U-preset> NQ_KV_BYTES=<validated-byte-budget> \
  spark/flash/run_spark.sh up
```

Keep CLI `--kv-cache-dtype fp8`: it selects the established sparse backend and
FP8 indexer. The explicit environment setting replaces only the MLA storage
and attention implementation. The startup log says `EXPERIMENTAL Flash FP4 MLA`.
Do not reuse a KV allocation from another format; start a fresh process.
Reverting means restart with `NQ_FLASH_MLA_CACHE=fp8` (or unset it).
The launcher targets aarch64 Spark; use the coordinated SM120 launcher for a
local test, adding the same environment variable and isolated worktree mount.
Do not run this on top of the live benchmark.

## Validation performed (2026-10-09)

- Eight CPU tests passed, including real Triton kernel code executed through
  its CPU interpreter against an independently enumerated FP64 oracle.
  Includes all E2M1 codes, halfway rounding, negative values, FP16 scale limits,
  zero rows, padded/permuted pages, partial/empty sparse sets, DSA2048 and
  DSA+SWA2176 widths, split softmax merge, masked draft rows, and slot reuse.
- All 33 Flash CPU tests passed together in interpreter mode; normal mode
  passes 31 with the two interpreter-only tests skipped.
- All pack/attention/merge variants compiled without a CUDA context for SM120
  and SM121, FP16/BF16 and both sparse widths. Attention reports 82,176 bytes
  shared memory, pack 256 bytes, merge zero. Compilation is not device testing.
- Actual installed vLLM classes checked in an isolated CPU-only container
  process: one-token alignment probe, stamped/merged MLA specs, unchanged
  non-Flash spec, backend shape and actual padded-cache reshape. CUDA context
  remained uninitialized. No live process or files were modified.
- GPU numerical checks passed on RTX PRO 6000 SM120 after benchmark/server
  were stopped by the coordinator. Packing matches the independent CPU format
  byte-for-byte, including non-unit-scale midpoint regression cases. Checked
  padded/permuted pages, addresses near 262K, guards, slot reuse and attention
  batches 1/2/3/4/17/65 at both 2048/2176 sparse widths.
- Maximum discrepancies against FP64 attention on the quantized cache:
  FP16 max-abs 7.90e-5, relative L2 2.83e-4; BF16 max-abs 6.49e-4,
  relative L2 0.00217. These measure kernel numerics, not FP4 model quality.
- Isolated attention GPU timings: 0.068–0.080 ms for batches 1–4,
  0.234–0.262 ms for batch 17, 0.722–0.804 ms for batch 65 (64 heads).
  These include bounded scratch allocation and merge, exclude cache packing,
  and do not establish a speedup over FP8 or an end-to-end TPS result.
  Log: `/tmp/nestquant/flash-artifacts/fp4-gpu-parity-fixed.log`.

Commands (use an environment with Torch, NumPy and Triton):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TRITON_INTERPRET=1 python -m unittest discover -s tests -p test_flash_fp4_kv_cpu.py -v
CUDA_VISIBLE_DEVICES='' python spark/flash/benchmarks/compile_fp4_kv.py
python spark/flash/benchmarks/check_fp4_vllm_specs.py --container glm53-flash-nestquant
```

Still required before promotion:

1. Run `benchmarks/check_fp4_kv_gpu.py` under compute-sanitizer when available:
   neither the current image nor standard host paths contain it, so memcheck
   remains untested. Run device parity and timing on actual SM121 Spark as well;
   that target has only been compiled offline.
2. Start an isolated model process, confirm hybrid group sizing and MTP cache
   ownership, run repeated requests, prefill chunk boundaries, long-context
   needle/coherence and stochastic generation checks. Explicitly test MTP off,
   MTP1 and MTP2. Compare quality against FP8 KV using the established teacher
   method when available; this implementation does not reproduce private KLD.
3. Measure peak GPU+host memory at 262K, real prefill/decode throughput and
   acceptance, then compare with FP8 at the same hot pool and request settings.
   End-to-end quality/performance and full-context memory fit remain **untested**.
