# Bounded asynchronous decode

The asynchronous decode path is experimental. Unit tests and a three-expert real-weight check pass. Full-model correctness, quality, and throughput have not been established for this path.

## Memory budget

The operator's total runtime ceiling is 96 GB, including the KV cache. On the tested 128 GiB Mac, use the stricter 84 GiB process guard (about 90.2 GB). Base weights, the text backbone, refinements, staging, KV state, and temporary allocations all count toward process footprint. The guard also refuses macOS memory-pressure states. These are sampled checks, not an operating-system allocation quota.

The runtime constructs `Glm5NextTextModel`. It does not instantiate or load the vision tower. The original checkpoint files remain intact on disk.

The first guarded 2% trial stopped during base loading when macOS reported pressure level 2. Its observed process peak was 42.89 GiB, below the 84 GiB cap. It produced no generated answer or throughput result. The supervisor restored the existing model service. See the [failed trial](results/2026-11-04-async-mmap-guard-trip.json).

The expert and backbone readers now use Safetensors 0.8's `pread` backend. Its [macOS implementation](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs) requests `F_NOCACHE` for reads, avoiding retained memory-mapped source pages. File-cache duplication is a plausible contributor to the failed load, not a confirmed sole cause. The memory guards remain unchanged. Exact byte comparisons and the asynchronous component check pass with this backend; full-model validation remains pending.

`--cache-gib` covers resident refinements and a fixed 64 MiB staging allowance. The reader reserves three times each expected payload plus 1 MiB for shard copies, merged arrays, and installation. There is one worker and at most two pending reads. Handles and metadata add fixed overhead outside the tensor budget.

The full desired set must fit within the resident portion of the cache. Capacity is checked before loading the backbone. With a 4 GiB budget, initial candidates are:

| Hot fraction | Slots | Maximum refinement bytes |
|---|---|---|
| 1% | 121 | 1,004,744,704 |
| 2% | 242 | 2,008,899,584 |
| 4% | 484 | 4,016,562,176 |

The 67,108,864-byte staging allowance fits beside each allocation. These are capacity-validated candidates, not quality or speed recommendations. The canonical 21.74% pool does not fit this budget. The earlier 24 GiB resident-hot experiment was stopped under memory pressure and must not be treated as a recommended configuration.

## Execution contract

The serving thread obtains the current jT desired set and delivery priorities. The background worker reads missing refinements into owned NumPy arrays through private safetensors handles. It does not call MLX or mutate the serving cache.

At layer boundaries, the serving thread installs completed refinements. An old resident remains available until a replacement arrives, subject to the configured per-layer slot count. Decode uses the ready level, falling back to the resident 1.5-bit base when a refinement is unavailable. Desired and executed hot shares are reported separately. This follows Spark's readiness behavior rather than its diagnostic wait-for-upgrades mode.

The scheduler discards completed reads that are no longer desired. Worker errors propagate. Request completion cancels queued reads and waits for the current read before reporting resource counters. Shutdown joins the worker before closing its handles or clearing storage.

Read-ahead overlaps SSD reads with decode. GPU installation still runs on the serving thread. The predictor still runs on CPU in float32. Prefill remains layer-major with synchronous reads and 4-bit selected experts. Asynchronous prefill and GPU predictor execution are not implemented.

Readiness depends on timing, so generated tokens can differ between otherwise identical runs. The existing deterministic wait-for-upgrades path remains available for diagnostics. Preserving the same predictor does not establish equal output quality between these modes.

## Validation and tuning

Run the bounded component checks first:

```sh
.venv-macos/bin/python -m unittest discover -s macos -p 'test_*.py' -v
.venv-macos/bin/python macos/verify_async.py --model "$MODEL_ROOT" --output async-experts.json
.venv-macos/bin/python macos/verify_io.py --model "$MODEL_ROOT" --output pread-io.json
```

After code review, run a full-model smoke check with multiple prompt chunks:

```sh
.venv-macos/bin/python macos/verify_generation.py \
  --model "$MODEL_ROOT" --output generation-async-2pct.json \
  --resident-base --async-refinements --hot-fraction 0.02 \
  --cache-gib 4 --memory-limit-gib 84 --prefill-chunk-size 8
```

Stop other model servers and downloads before measuring throughput. Use the same prompt, token count, chunk size, cache budget, and wiring setting for each candidate. A benchmark example is:

```sh
.venv-macos/bin/python macos/benchmark.py \
  --model "$MODEL_ROOT" --output-dir benchmark-async-2pct \
  --resident-base --async-refinements --hot-fraction 0.02 \
  --cache-gib 4 --memory-limit-gib 84 --tokens 128 --runs 3
```

Record peak process footprint and system headroom. Include a separate process watchdog when testing on an active desktop. The local test supervisor stops at 84 GiB footprint, a macOS pressure warning, or less than 20% reported free memory. It restores the normal model service afterward.

The local watchdog passed a live stop-and-restore test with a 128 MiB allocation and a 64 MiB limit. The test exercised native memory probes and terminated the child. It restored the existing service and preserved previously paused workers. This proves the supervisor lifecycle, not full-model memory safety. Its launch validation rejects resident-hot mode, caches above 4 GiB, and runtime limits above the watchdog limit.

Begin with the short multi-chunk smoke above. Run a 16-token comparison only after the smoke passes and its recorded memory peak leaves headroom. Extend to 128 tokens after inspecting the short comparison. Keep the same prompt and generation length across compared candidates. A guard trip is a failed trial and must remain in the results.

Keep inclusive decode throughput as the request-level measurement. The reports also expose the first decode step, throughput after that step, predictor time, reader time, installation time, and final reader-drain time. Reader time overlaps computation and must not be added to wall time. Separating early steps does not prove steady-state performance.

The benchmark summary includes desired and executed hot-expert fractions plus asynchronous fallback counts. Compare these alongside speed because readiness changes the precision actually used.

Reject candidates that fail the required answer or coding checks. Three short smoke cases do not replace broader quality evaluation or a teacher comparison. Do not increase cache residency to rescue a candidate that exceeds the memory budget.
