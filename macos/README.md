# Apple silicon runtime

This experimental runtime reads the published GLM-5.3-Flash NestQuant 1.5/4 checkpoint. The native path uses a prepared MLX backbone and bounded Metal expert banks. The reference path uses the Transformers text backbone on PyTorch MPS.

**Status, 2026-11-05.** The native 32 GiB base profile passed 10/12 coding cases at 3.55 median decode tokens/s with 2% hot allocation. A 3% profile also passed 10/12, at 3.47 tokens/s and 46.87 GiB peak footprint. Both retained normal pressure and restored the daily model service. These results remain below the 15-30 tokens/s target. See [coding evidence](PERFORMANCE.md#twelve-case-coding-subset) and [the allocation comparison](PERFORMANCE.md#native-hot-allocation-comparison). A separate 40 GiB run passed one coding case at 4.62 tokens/s, but broader safety at that size remains unproven.

The optional staged-read path later measured 5.23 and 5.39 tokens/s with 128 base rows, near 40.7 GiB peak footprint. Both retained normal pressure and added no swap-outs. The scattered controls varied from 3.32 to 0.71 tokens/s, so aggregate percentage improvements are unreliable. This fixed-input replay does not establish answer quality for the faster profile. The later quality suite stopped at pressure level 2 during its first case, below the 48 GiB cap. Qwen recovered, and no quality score was produced. See [the staged-read results](PERFORMANCE.md#staged-required-base-reads).

The asynchronous path reads refinements in the background and executes the precision that is ready. The grouped reader passed three full-model answer checks and delivered 16.25% actual hot-call coverage. The earlier reader also passed one complete coding task. The unchunked BF16 comparison retains its documented tolerance failure. See the [asynchronous path](ASYNC.md) and [the separate greedy MTP experiment](#greedy-mtp-experiment).

## Requirements

You need an Apple silicon Mac, Python 3.12, and the packages in [requirements.txt](requirements.txt). The tested machine runs macOS 27.2. Other Apple silicon models are untested.

The filtered serving checkpoint occupies 188,956,641,846 bytes. Reserve additional space for download metadata and temporary files. This path does not need the Spark repack or Docker image.

Install the dependencies in an isolated environment:

```sh
uv venv --python 3.12 .venv-macos
uv pip install --python .venv-macos/bin/python -r macos/requirements.txt
```

Download the pinned checkpoint:

```sh
export MODEL_ROOT="$HOME/Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit"
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv-macos/bin/hf download \
  jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit \
  --revision 4023870a6f2c6097e37fa5d2e54040db64b011e9 \
  --exclude 'experiments/*' --exclude 'evaluations/*' \
  --local-dir "$MODEL_ROOT" --max-workers 2
```

The download command reuses completed files. Interrupted files can require data to be downloaded again. Preserve incomplete files when retrying. Verify the runtime weights before inference:

```sh
.venv-macos/bin/python macos/verify_checkpoint.py \
  --model "$MODEL_ROOT" --output checkpoint-verification.json
```

This command checks the checkpoint layout and the reviewed predictor code hashes. It verifies 341 expert, backbone, and predictor weight files against their manifest checksums. Its target list matches the pinned Hugging Face metadata. All 341 runtime weight files passed full checksum verification on the tested Mac. Vision and speculative-decoding weights are outside this runtime check.

## Component validation

The decoder test needs only `layers/L3/manifest.json` and `layers/L3/tp0.safetensors`. It verifies the shard checksum before comparing Metal output with the independent CPU reference.

```sh
.venv-macos/bin/python macos/verify_decoder.py \
  --layer "$MODEL_ROOT/layers/L3" --output decoder-verification.json
.venv-macos/bin/python macos/verify_projection.py \
  --layer "$MODEL_ROOT/layers/L3" --output projection-verification.json
.venv-macos/bin/python macos/verify_backbone.py
.venv-macos/bin/python macos/verify_policy.py
.venv-macos/bin/python -m unittest discover -s macos -p test_validation.py -v
```

The following checks need all eight layer 3 shards. The policy check also needs the downloaded predictor files:

```sh
.venv-macos/bin/python macos/verify_store.py \
  --model "$MODEL_ROOT" --output resident-store-verification.json
.venv-macos/bin/python macos/verify_expert.py --resident-base \
  --model "$MODEL_ROOT" --output resident-expert-verification.json
.venv-macos/bin/python macos/verify_policy.py \
  --model "$MODEL_ROOT" --output predictor-verification.json
```

After the four nonexpert shards arrive, verify their checksums and exercise the trained backbone:

```sh
.venv-macos/bin/python macos/verify_checkpoint_backbone.py \
  --model "$MODEL_ROOT" --output trained-backbone-verification.json
```

This check loads every backbone tensor, then tests the first four trained layers with real layer 3 experts. It does not generate with the full model.

Historical component evidence: [complete layer 3 decoding](results/decoder-layer3-summary.json) and [runtime checksums](results/checkpoint-verification.json). To reproduce the complete decoder scope, run each `--shard` from 0 through 7 with `--experts` containing all integers from 0 through 287. The short command above uses the smaller default sample.

Observed results on the tested Mac:

| Check | Result | Scope |
|---|---|---|
| Ring decoding | Exact match for 14,495,514,624 weight values | All 288 experts, three projections, both levels, all eight L3 shards |
| Projection reconstruction | Relative root-mean-square error below 0.1% | Three experts, both levels, L3 TP0 |
| Fused matrix-vector multiplication | Relative root-mean-square error below 0.1% | Random inputs and real checkpoint projections |
| Batched multiplication | Relative root-mean-square error below 0.1% | Five-token batches and real checkpoint projections |
| Complete experts | Relative root-mean-square error below 0.1% | Three experts, all eight shards, both levels, single-token and five-token inputs |
| Text backbone | CPU and MPS agree within the test tolerance | Random small GLM model, both attention types and cached decode |
| Trained backbone load | All 1,177 model tensors and the output head are finite on MPS | Four nonexpert shards verified with SHA-256 |
| Trained prefix | Bitwise match with matching chunks | Four trained layers, real L3 experts, two chunk sizes, and two subsequent decode tokens |
| Layer-major prefill | Bitwise match to standard execution with matching chunks | A 37-token prompt, sparse attention selection, two chunk sizes, and three subsequent decode tokens |
| Hot-expert allocation | Slot totals and fixed subsets pass | Minimum allocation through 100% hot experts |
| Resident base storage | Outputs match the bounded loader exactly | All L3 base experts loaded; three experts checked at both levels; refinement eviction and reload |
| Resident hot set | Retained refinements survive pool changes; oversized pools are rejected | Real L3 experts, repeated pools, overlapping pools, and subsequent hot/cold reads |

The larger BF16 integration test is bitwise identical to ordinary Transformers execution with matching chunks. Its cached decode differs by 1.3086% from an unchunked run, exceeding the separate 1% check. Ordinary Transformers chunking has the same difference. The failing unchunked comparison remains available through `verify_integration.py --reference unchunked`; `--reference chunked` checks bitwise scheduling equivalence. Trained-model quality remains unverified.

The optional `--compute-dtype float32` control upcasts the backbone after identical BF16 initialization. Cached-decode error falls to 0.00427%, supporting rounding sensitivity in this fixture. It leaves the expert path's float16 boundaries unchanged. The original BF16 criterion still fails. See [the precision comparison](results/2026-11-05-chunk-precision-control.json).

These results do not establish full-model output quality or throughput. The GPU timings in the decoder report measure a component, not generation speed.

## Generation interface

The following experimental command uses bounded base storage. Stop other model workloads and use the external watchdog described in [the validation procedure](ASYNC.md).

```sh
.venv-macos/bin/python macos/generate.py \
  --model "$MODEL_ROOT" \
  --prompt 'Reply with exactly: GLM_MAC_OK' \
  --max-tokens 16 \
  --hot-fraction 0.02 --async-refinements \
  --cache-gib 4 --resident-base --base-cache-gib 32 \
  --memory-limit-gib 84 --prefill-chunk-size 8 \
  --output generation.json
```

The default allocation is 2,630 hot experts across 42 routed layers. The predictor mixes 75% jT with 25% exponential moving average. It refreshes after every committed token. Twenty percent of allocated slots are fixed experts.

`--hot-fraction` changes the total hot-expert allocation. Other allocations preserve the canonical layer weighting approximately, with integer counts between two and 288 experts per layer. `--cache-gib` separately bounds cached expert tensors. Cold experts load only the base planes. Hot experts additionally load their refinements.

The default mode uses a bounded least-recently-used cache for selected experts. It does not retain every base expert. Add `--resident-base` to preload and retain all base experts before inference. This retains 59,052,758,208 bytes (55.00 GiB) outside `--cache-gib`. In this mode, the bounded cache holds refinements and shares base tensors. Evicting a refinement does not evict its base. The all-hot smoke run with resident base tensors and a 4 GiB refinement cache peaked at 81.62 GiB of process physical footprint.

To bound the separate base cache, add `--base-cache-gib 32` alongside `--resident-base`. This skips full base preloading and evicts least-recently-used bases with their refinements. Base misses are synchronous. This experimental mode preserves stored precision and passed the documented short full-model smoke. Use the [guarded validation procedure](ASYNC.md) before benchmarking; full base preloading is not a desktop memory recommendation.

These two modes load refinements when a routed expert needs them. A larger hot fraction changes reconstruction precision without guaranteeing a matching increase in resident memory. Measure cache behavior and SSD reads together.

Add `--resident-hot` with `--resident-base` to keep every predicted hot expert's refinements resident during decode. Before each decode step, the runtime removes refinements outside the predicted set and loads missing refinements. Refinements that remain hot retain their existing buffers. Cold experts use the resident base tensors. Updates are synchronous and their cost is included in decode timing.

The runtime checks the worst possible refinement allocation for the configured per-layer slot counts before loading the backbone. An insufficient `--cache-gib` fails instead of silently evicting hot experts. The canonical 2,630-slot allocation requires at most 20.314 GiB of refinements. A 24 GiB test was stopped under memory pressure on this Mac. Do not use it as an initial recommendation. This requirement excludes base tensors, the backbone, attention caches, and temporary allocations. Full-model memory and performance of resident-hot mode remain unverified.

Add `--wired-memory` to keep MLX allocations resident during inference up to the device's recommended working set. The runtime restores the previous process limit when it closes. This uses [MLX memory wiring](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html) without changing system limits. Wiring does not reduce the model's memory requirement. Its performance effect remains under measurement.

Prefill processes every prompt chunk through one layer before advancing. Every selected prefill expert uses its 4-bit reconstruction. The runtime retains that layer's cached tensors between chunks when the cache budget permits. This avoids reducing prefill precision to the decode allocation. It does not imply equality with the unquantized model.

The initial interface supports one text sequence, greedy sampling, and a 4,096-token validation ceiling. Vision, speculative decoding, an HTTP endpoint, and quantized attention caches are not implemented. The backbone retains the checkpoint's original nonexpert precision, with FP8 weights expanded for MPS. This differs from the additional backbone quantization in the published Spark recipe.

Generation uses the repository's `spark/flash/thinking_off.jinja` template by default. Add `--thinking` to use the checkpoint's reasoning template. The checkpoint template ignores `enable_thinking=false`, so passing that keyword alone does not disable reasoning.

If the loader reports a mapping or shape mismatch, stop inference and check the checkpoint revision and dependency versions. Do not ignore missing weights. If memory pressure rises, stop the run and inspect its receipt. Preserve the process guard and bound the separate base cache before another trial.

## Native packed experiment

The native path is an explicit alternative to the PyTorch reference. Install `mlx-vlm==0.7.12` in the Mac environment. Use the verified checkpoint and keep the original files available for refinements. Prepare the two derived weight directories on the internal SSD. The base and backbone files together need about 66 GiB of disk space. The preparation command checks the checkpoint layout and retains a disk reserve.

```sh
export MODEL="$MODEL_ROOT"
export BASE_ARCHIVE="$HOME/Models/GLM/resident-banks-v1"
export BACKBONE_ARCHIVE="$HOME/Models/GLM/backbone-affine8-v1"
export RESULTS="$PWD/macos-results"
mkdir -p "$RESULTS"
uv pip install --python .venv-macos/bin/python -r macos/requirements.txt 'mlx-vlm==0.7.12'
.venv-macos/bin/python macos/native_assets.py --model "$MODEL" --output-dir "$BASE_ARCHIVE"
.venv-macos/bin/python macos/native_backbone.py --model "$MODEL" --output-dir "$BACKBONE_ARCHIVE"
```

Run one large model at a time. Keep the host memory watchdog active. Test the prefill order before running mixed inference. The verification command compares logits and the contents of each cache array.

```sh
python macos/native_generate.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --backbone "$BACKBONE_ARCHIVE" --verify-prefill --output "$RESULTS/prefill.json"
python macos/native_generate.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --backbone "$BACKBONE_ARCHIVE" --rows 96 --chunk-size 4 --refinements \
  --prompt 'Reply with exactly GLM_RESIDENT_OK and no other text.' \
  --output "$RESULTS/native-smoke.json"
```

The native path reads missing base rows from disk into fixed slots. It records row hits, misses, eviction, bytes, and read time separately from the refinement scheduler. It uses four-bit refinements during prefill with `--refinements`. It uses the existing jT predictor during ordinary decode. It keeps hidden states and expert outputs on the device and reads routing IDs for storage selection. Its 4,096-token validation bound includes input and output. A pressure stop means the profile exceeded the live system budget. Inspect the pressure and allocation receipts before another attempt.

The opt-in `--base-prefetch 1` experiment predicts base reads from the committed jT/EMA scores. It requires `--refinements` and uses a separate 128 MiB staging reservation inside the process and pressure guards. One worker reads private CPU buffers up to four layers ahead. Only actually routed rows can enter the persistent GPU bank. Missing predictions still trigger required reads. Speculative I/O failures receive one synchronous fallback when required. Allocation failures and malformed payloads stop the run. The first controlled replay measured 3.45 tokens/s versus 3.56 without prediction, so the option remains disabled by default. See [the read-ahead comparison](PERFORMANCE.md#anticipatory-base-reads).

Use `--verify-prefill --refinements` to compare both schedules with four-bit prefill and check the committed predictor state. The comparison uses the same chunk size for both schedules. Choose `--prefill-rows` to bound the scratch bank for one layer. Decode slots, this scratch bank, and the asynchronous worker share a default 4 GiB refinement limit. The separate base banks default to 32 GiB. Explicit paged budgets of 40 or 44 GiB allow 192 or 224 rows per layer. The process and system-pressure guards remain active. Larger budgets require their own measured headroom.

The optional `--refinement-budget-gib 6` admits a separately bounded allocation for supervised experiments. It requires `--refinements` and does not change process or system guards. `--hot-fraction 0.05` requires this six-GiB budget. Five-percent hot allocation with 128 prefill rows reserves 5.737 GiB, including the worker. This profile has no model benchmark or quality result. Keep the default until its review and runtime checks pass.

The optional `--cold-fastpath` uses base-only projection operations when every selected decode expert lacks a ready refinement. It still publishes completed reads, records predictor inputs, and retains both banks' output lifetimes. Prefill keeps the existing mixed projection path. The flag requires `--refinements` and remains disabled by default. Real-weight all-cold comparisons matched the reference projection at layers 3 and 44. No model speedup is established.

Refinement reports separate `prefill_reader_seconds`, `prefill_install_seconds`, and `decode_publication_seconds`. Reader time includes positional reads, assembly, validation, and payload copies. Installation can include waiting for the bank's previous GPU use. Publication measures the eligibility check and installation, including any such wait. Background worker time can overlap these windows, so do not add all counters to estimate elapsed time.

The experimental complete-base mode requires `--full-base --rows 288 --base-budget-gib 56`. It loads every prepared base plane directly into persistent storage before prefill. It requests 68 GiB of process residency for the base and backbone. Base-only routing then uses GPU expert IDs without extracting them on the CPU. This mode rejects refinements until combined memory use is validated. Full-model generation with this mode has not yet passed the guarded benchmark gate.

For controlled startup experiments, `--full-base-wired-gib` accepts 0, 32, 56, or 68 GiB. Zero requests no persistent MLX wiring. It does not prevent driver residency during GPU work. `--preload-mib-s` accepts 0, 256, 512, or 1024 MiB/s. Zero preserves unpaced loading. These options require `--full-base` and retain every memory guard. Completed and failed layer events record read time, pacing time, and memory samples. Full-base `read_seconds` measures positional reads only; pacing and guard time are separate.

The complete base contains 55.46 GiB. Base-only residency passed on the 128 GiB test machine. Adding the 9.79 GiB backbone triggered pressure with both the stock MLX reader and the direct persistent loader. These are workload-dependent results, not a fixed hardware capacity limit. See the [residency measurements](PERFORMANCE.md#later-full-base-residency-checks).

The guarded `--rows 192 --base-budget-gib 40 --chunk-size 8 --refinements --prefill-rows 160` profile passed one coding check at 4.62 tokens/s. A 224-row profile reached 3.94 tokens/s on a separate fixed-token replay. Both retain the process and system guards. Keep these profiles supervised until longer-context memory use and broader quality are measured. See the [cache comparison](PERFORMANCE.md#larger-bounded-caches).

Run the real-weight comparison before testing complete-base generation. It loads one full layer at a time and rejects additional base reads during device-routed computation.

```sh
python macos/verify_resident.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --output "$RESULTS/resident-parity.json"
MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=250 python macos/probe_resident_memory.py \
  --archive "$BASE_ARCHIVE" --reader mlx --wire-base --touch-each-layer \
  --output "$RESULTS/base-residency.json"
```

The residency probe requires the external host watchdog. Add `--model "$MODEL" --backbone "$BACKBONE_ARCHIVE" --wired-gib 68` to measure the backbone with the base. The refresh option repeats process residency requests; it does not change a system memory limit. Its implementation is in [MLX 0.32.4](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/resident.cpp).

Use `--reader mlx-nocache` to measure MLX-owned storage loaded through an uncached file handle. The reader evaluates all arrays before closing the handle. This combined probe also stopped under pressure on the test machine. The mode is a diagnostic option, not a validated full-model serving profile.

The complete-base loader records pressure and system VM counters after each layer. It also samples these counters after loading, prefill, and decode. Shared imported storage appears in MLX active-memory accounting and its residency set. These measurements overlap with the process footprint. The complete-base route timer measures graph construction without the paged path's synchronization, so compare complete decode time between modes.

Run the recurrent attention probe with the prepared environment to compare the reference fallbacks with the native backbone kernels.

```sh
python macos/verify_recurrent.py --model "$MODEL" --profile-components \
  --output "$RESULTS/recurrent.json"
```

Replay a fixed sequence with the reference runtime to compare cache capacities and base read modes. Set `$FIXED` to the [run receipts](results/2026-11-05-native-memory-replay.json) to reproduce the recorded token sequence. Use fresh result files. Add `--profile-crossings` to a separate run to split the explicit MPS drain from the host conversion.

```sh
python macos/replay_cache.py --model "$MODEL" --fixed-json "$FIXED" \
  --tokens 128 --output "$RESULTS/replay-sync.json"
python macos/replay_cache.py --model "$MODEL" --fixed-json "$FIXED" \
  --tokens 128 --async-base --output "$RESULTS/replay-async.json"
python macos/native_generate.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --backbone "$BACKBONE_ARCHIVE" --rows 160 --chunk-size 8 \
  --refinements --prefill-rows 160 --fixed-json "$FIXED" --max-tokens 128 \
  --prompt 'Write a Python function that merges overlapping intervals. Explain its algorithm, edge cases, and time complexity. Include examples.' \
  --output "$RESULTS/replay-native.json"
```

See [the measured results](PERFORMANCE.md#base-cache-and-native-execution) for memory accounting, throughput and the constraints on each comparison.

### Contiguous base archive

The optional expert-major archive stores each base expert's packed planes together on disk. Required fills use one positional scatter read into the existing bank buffers when the operating system completes the read. Partial reads continue from the remaining buffers. This preserves tensor bytes and the GPU bank layout. It adds no prediction and never skips required weights. Private asynchronous payloads remain compatible with the existing base-read worker.

Preparation needs another 55.46 GiB of disk space and preserves a 100 GiB free-space reserve. It retains the original base archive. Preparation and validation enforce a 4 GiB inner memory guard. Keep the external host watchdog active. The tested wrapper used an 8 GiB limit for these component operations.

1. Select a fresh destination for the derived archive.

   ```sh
   export EXPERT_ARCHIVE="$HOME/Models/GLM/expert-major-banks-v1"
   ```

2. Prepare the contiguous archive from the verified base archive.

   ```sh
   python macos/expert_major_base.py --source "$BASE_ARCHIVE" --output-dir "$EXPERT_ARCHIVE"
   ```

3. Verify every expert's bytes and the sampled projections, including eviction and reload.

   ```sh
   python macos/verify_expert_major.py --source "$BASE_ARCHIVE" --archive "$EXPERT_ARCHIVE" \
     --output "$RESULTS/expert-major-parity.json"
   ```

Both commands accept `--layers` for an explicit subset. Preparation refuses existing destination files. After interruption, preserve completed file/sidecar pairs. Remove only the incomplete derived files for the failed layer before retrying that layer. A completed archive needs both its Safetensors file and matching sidecar. The sidecar records the file hash, source hash, revision, and file fingerprint.

Use `--archive "$EXPERT_ARCHIVE"` with the native paged generator to select this format after validation and review. The expert-major format explicitly rejects full-base preload. The original archive remains available for that separate experiment. The reviewed four-run comparison measured 3.715 median tokens/s with this layout versus 3.220 with the original. Blocking base-read time fell by 16.2%. This fixed-input replay does not establish answer quality. See [the comparison and its limits](PERFORMANCE.md#contiguous-required-base-reads).

All 42 converted layers passed byte-for-byte checks over 59,553,531,072 bytes and 756 sampled projection comparisons. Validation included eviction and reload. The check peaked at 519,078,944 bytes of process footprint, retained normal pressure, and restored Qwen. System swap-outs increased by 393,216 bytes during the check and cleanup. These system counters do not attribute swaps to the verifier. See [the component evidence](results/2026-11-05-native-storage-validation.json).

### Required-read concurrency probe

The component probe measures private CPU-buffer reads at widths 1, 2, and 4. It uses the validated expert-major archive and preserves source files. The fixed sequence contains distinct expert rows across all 42 layers. Six trials use widths 1, 2, 4, 4, 2, and 1. Each trial excludes eight warm-up reads.

Keep the external watchdog at 8 GiB. The probe enforces a 4 GiB inner guard and allocates four bounded row buffers. It joins worker threads before closing files, including after failure. Each read checks its length and the first and last 64 bytes. The separate archive verifier supplies complete byte equality.

```sh
python macos/probe_required_reads.py --archive "$EXPERT_ARCHIVE" --reads 384 \
  --output "$RESULTS/read-scaling.json"
```

Active batch time includes thread dispatch, completion, and boundary checks. Wall time also includes memory checks. Neither measure establishes model throughput or the performance of writes into shared GPU banks. The first completed sweep was cache-confounded: later passes physically read only 21.3% of requested bytes. Its width ratios do not establish SSD scaling. See [the diagnostic results](PERFORMANCE.md#required-read-diagnostics).

Native bank statistics also report `miss_batch_0` through `miss_batch_8` and the `miss_batch_ge9` overflow bucket. Each bucket records successful selection calls and their required-read seconds. Duplicate routed IDs count once. Timing excludes speculative-read waits and waiting for prior GPU use. Decode selects eight experts, so its histogram can support exact projections for different reader widths. Such projections remain estimates until model testing.

The single-layer layout probe compares the actual scattered bank fill with private contiguous, shared contiguous, and staged-copy reads. It uses 32 disjoint expert IDs per layer and rotates variant order. It touches destination pages before timing. After all four variants, it validates two complete rows per variant against the archive. It closes each layer before opening the next.

```sh
python macos/probe_required_layout.py --archive "$EXPERT_ARCHIVE" \
  --output "$RESULTS/read-layout.json"
```

Use the same 8 GiB external watchdog and 4 GiB inner guard. Logical buffers total 50 expert rows, approximately 236 MiB with this archive. This probe runs without model execution, refinement contention, or concurrent reads. Summaries include only layers where every variant physically reads at least 95% of its requested bytes. A passing byte check does not establish storage comparability. The completed run admitted all 42 layers. Median scattered reads took 2.688 ms, versus 0.881 ms for a contiguous read and copy combined. This component result does not establish a model speedup. See [the comparison](PERFORMANCE.md#required-read-diagnostics).

The native generator accepts experimental `--base-read-mode staged` with an expert-major archive. The default remains `scatter`. Staged mode reads each required row into one shared 16 MiB CPU buffer, then copies its planes into the existing bank. A lock protects the complete read-and-copy interval across banks. Previous GPU use still completes before slot replacement. Required reads remain synchronous, and asynchronous refinements retain their existing worker.

The base-row budget excludes this separate 16 MiB reservation. The whole-process guards still cover it. Output records `base_read_mode`, `base_staging_bytes`, `read_copy_seconds`, and `read_staging_wait_seconds`. Required-read timers exclude copying and lock waits. Staged mode rejects field-major archives and full-base preload. Real-weight projection checks and the reviewed 128-row replay passed. Answer-quality confirmation for this profile remains pending.

Use `--base-read-mode staged` with `verify_expert_major.py` to compare the candidate's GPU projections against the original archive. This verification retains eviction and reload checks. It reports candidate read, copy, and wait counters.

### Optional execution diagnostics

Use `--replay-passes 2 --fixed-json <trace.json> --max-tokens 128` to compare initial and warm execution within one loaded model. Keep the external memory watchdog active. This option retains base and refinement banks but creates a fresh attention cache and resets the predictor for each pass. It excludes other diagnostic modes and prefill verification. The default remains one pass.

The existing result fields describe the first pass. The `warm_replay` field contains the second pass. Each completed pass also emits a flushed `replay_pass_complete` JSON record. Preserve that log if a later pass stops before the result file is written.

Compare actual misses, read bytes, evictions, and decode time before attributing a change to storage. Retained banks do not guarantee zero misses. Refinement readiness and greedy predictions can differ despite identical forced tokens. Compare `replay_steps` and `greedy_next_ids` across passes. This diagnostic does not establish answer quality, a throughput ceiling, or long-context capacity. Two guarded trials stopped under pressure, so a complete paired result remains pending. See [the retained results](results/2026-11-06-native-warm-replay.json).

The owner waived the Fable review gate before three guarded full-model trials. The 64-row ordinary replay measured 4.59 tokens/s. Instrumented coding and synchronized replay measured 4.11 and 3.74 tokens/s respectively. The coding answer passed HumanEval/3. All three runs retained normal pressure, added no swap-outs, and restored Qwen. These diagnostics have no independent Fable approval. See [the diagnostic evidence](results/2026-11-05-native-diagnostics.json).

Add `--diagnostics /absolute/path/trace.jsonl` to a supervised native generation command. Use a fresh path that differs from the result path. The runtime flushes each JSON record. Records identify loaded backbone modules, prepared base banks, prefill layer boundaries, and completed decode steps. Memory records retain process, MLX, compressor, and pressure measurements. A guard stop can therefore leave useful evidence without a completed result file.

Add `--profile-sync` to that diagnostic command to evaluate each routed layer's output during decode. This mode requires paged banks and at most 128 output tokens. It records selected expert IDs, actual refinement slots, and host wall intervals. The intervals cover route extraction, base selection, refinement publication, graph construction, and the explicit output wait. Per-step records also separate policy preparation, model execution, predictor commit, and token selection. Diagnostics require ordinary layer-major generation and exclude the prefill-equivalence command.

The intervals are not pure GPU timings. Base selection can wait for previous GPU use. Refinement publication can wait for another bank. Graph construction can schedule GPU work or wait for allocation. Evaluating the output adds a synchronization boundary and can change refinement readiness. Memory records and flushed writes also add overhead. The result marks its timings as diagnostic. Do not report them as ordinary throughput or sum overlapping model, layer, and read intervals.

The default path adds no layer evaluation or diagnostic file. Existing memory guards remain active. These trials used 64 base rows, eight-token chunks, 3% hot allocation, and a 40 GiB outer watchdog. Their peaks stayed below 28.23 GiB. One passing coding case does not establish broader quality.

### Optional paged residency

Add `--paged-wired-gib 32` to request up to 32 GiB of resident MLX allocations during paged generation. Use `--paged-wired-gib 0` for an explicit zero-residency control. Omit the option to preserve existing allocator behavior. Full-base generation uses its separate residency option and rejects this setting.

This setting changes residency, not expert allocation or process guards. It can increase system memory pressure. Keep the external watchdog and normal-pressure requirement active. The option passed 80 runtime tests. A guarded four-run comparison measured a 4.65 tokens/s median with 32 GiB residency, versus 4.09 without residency. The 13.7% gain applies to this fixed-input replay. A subsequent coding answer passed HumanEval/3 at 4.22 tokens/s. All five runs retained normal pressure, added no swap-outs, stayed below 28.23 GiB peak, and restored Qwen. Broader quality remains unmeasured. See [the residency evidence](results/2026-11-05-native-paged-residency.json). See [MLX residency semantics](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html).

A later exploratory screen measured 5.77 tokens/s with 96 base rows and the existing cold fast path. Its coding check passed HumanEval/3 at 5.13 tokens/s, but system swap-outs increased by 556.80 MiB during prefill. Pressure remained normal and Qwen recovered. System counters cannot attribute all paging to GLM. This profile is not promoted and will not be repeated unchanged. See [the screen and coding evidence](results/2026-11-05-native-profile-screen.json).

### Compact backbone experiment

The backbone preparer accepts `--bits 4` for eligible linear weights and `--embedding-bits 8` for embeddings. Use a separate, empty output directory. Both options are experimental. The default remains eight-bit linears and original embeddings. Protected float32 parameters retain their representation.

The archive records its precision. The loader accepts legacy eight-bit archives and rejects unsupported precision metadata. All 85 runtime tests passed, including parameter and output round trips for compact fixtures. These tests validate serialization and loading, not model quality. Guarded conversion produced 5.51 GiB of tensors with eight-bit embeddings, versus 9.79 GiB previously. All 669 protected tensors matched byte-for-byte. Conversion peaked at 12.14 GiB, retained normal pressure, added no swap-outs, and restored Qwen. The first coding answer passed HumanEval/3 at 5.05 tokens/s and 30.11 GiB peak. Its suite then stopped on a 128 KiB increase in system swap-outs during startup. Pressure remained normal, and no further swap-outs appeared during prefill or decode. Eleven cases did not launch. Broader quality and full-base feasibility remain unverified. See [the conversion and stopped quality suite](results/2026-11-05-native-compact-backbone.json).

### Banked threadgroup probe

Add `--threadgroups` to the supervised `microbench_banked.py` command to compare 64, 128, 256, and 512 threads per Metal group. Use a 4 GiB external watchdog. The probe checks layers 3 and 44 with eight cached experts. It covers base, refined, mixed, and all-cold mixed projections. Every candidate must match the 256-thread output exactly before timing and after each timed evaluation. Each scenario warms every candidate three times and measures 25 samples per candidate in a shuffled order. Guard checks and equality checks are outside timing.

The probe preserves each SIMD reduction and all quantized arithmetic. It changes how SIMD groups share a threadgroup. The production runtime keeps 256 threads. All 82 runtime tests passed, including both residual patterns and invalid-route isolation. Real-weight checks passed at layers 3 and 44. Timing differences were small and inconsistent, so the runtime keeps 256 threads. The probe peaked at 731.64 MiB, retained normal pressure, added no swap-outs, and restored Qwen. See [the component measurements](results/2026-11-05-native-threadgroups.json). Component timings do not establish model throughput.

### Loopback chat and prefix reuse

The experimental native server accepts one greedy, non-streaming text request at a time. It binds to loopback. Its OpenAI-compatible routes require a local model ID and send no requests to OpenAI. The server accepts function schemas and renders them with the pinned chat template. It returns model tool calls for the client to execute.

Keep the host memory watchdog active. Start the server with a verified checkpoint and prepared weights.

```sh
.venv-macos/bin/python macos/native_server.py --model "$MODEL" \
  --archive "$BASE_ARCHIVE" --backbone "$BACKBONE_ARCHIVE" --port 18085
curl --fail http://127.0.0.1:18085/v1/models
curl --fail http://127.0.0.1:18085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-nestquant-local","temperature":0,"max_tokens":3,"messages":[{"role":"user","content":"Reply with exactly GLM_OK."}]}'
```

Send the same request again to verify prefix reuse. Check `usage.prompt_tokens_details.cached_tokens` and `timings.computed_prompt_tokens`. Time to first token is measured inside the model. The HTTP server returns the complete response after decode. The cache uses exact rendered token prefixes and keeps attention, recurrent, pooling, and committed predictor state together. Its 512 MiB budget covers retained checkpoints. Requests can still need memory for the active model state.

The server rejects unsupported stop sequences, probability output, response formats, sampling controls, and penalties. An integer seed has no effect on greedy sampling. An inference failure terminates the process because model state may be incomplete. Restart it through the supervised launcher. Run `python macos/verify_native_server.py` with the native serving dependencies to verify these request boundaries.

Add `--refinements` for four-bit prefill and asynchronous mixed decode. Run the state checks with the same model and both prepared directories.

```sh
.venv-macos/bin/python macos/verify_prefix.py --model "$MODEL" \
  --archive "$BASE_ARCHIVE" --backbone "$BACKBONE_ARCHIVE" --refinements \
  --output "$RESULTS/prefix-state.json"
```

### Greedy MTP experiment

The MTP verifier loads and checks the checkpoint's FP8 draft shards. It runs the native draft head and batches verification with cache rollback. It uses base-only target experts and compares generated tokens with a serial reference. The ordinary mixed CLI and native server do not use MTP.

```sh
.venv-macos/bin/python macos/verify_mtp.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --backbone "$BACKBONE_ARCHIVE" --mtp 0 --output "$RESULTS/serial.json"
.venv-macos/bin/python macos/verify_mtp.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --backbone "$BACKBONE_ARCHIVE" --mtp 1 --reference "$RESULTS/serial.json" \
  --output "$RESULTS/mtp1.json"
```

## KL divergence harness

`kld_harness.py` scores the reference runtime with teacher-forced KL divergence. It reports pooled and per-item divergence, top-1 agreement, decode rate, desired and executed hot share, and expert-store counters. Use a fresh `--arm` name for each run.

Score layer-major prefill against stored BF16 teacher logits. Each teacher file holds `[n - 1, vocabulary]` logits for an `n`-token window, where row `i` predicts token `i + 1`:

```sh
.venv-macos/bin/python macos/kld_harness.py \
  --model "$MODEL_ROOT" --output-dir kld-results --arm teacher-prefill --mode teacher-prefill \
  --tokens window-0000.tokens.npy --teacher-logits window-0000.safetensors \
  --resident-base --base-cache-gib 32 --cache-gib 4 --memory-limit-gib 84 --prefill-chunk-size 64
```

Score decode on generated traces in two steps. A trace is a JSON file with `prompt_ids` and `token_ids`, where the prompt is a prefix of the tokens. The [10 November traces](results/2026-11-10-kld-traces/) are examples. First save the all-4-bit prefill logits of each complete trace:

```sh
.venv-macos/bin/python macos/kld_harness.py \
  --model "$MODEL_ROOT" --output-dir kld-results --arm reference --mode reference-prefill \
  --tokens traces/*.json --reference-dir kld-reference \
  --resident-base --base-cache-gib 32 --cache-gib 4 --memory-limit-gib 84 --prefill-chunk-size 64
```

Then prefill each prompt, force the generated tokens through single-token decode, and compare each step with the reference row at the same input position:

```sh
.venv-macos/bin/python macos/kld_harness.py \
  --model "$MODEL_ROOT" --output-dir kld-results --arm async-4pct --mode decode \
  --tokens traces/*.json --reference-dir kld-reference --max-decode-tokens 160 \
  --resident-base --base-cache-gib 32 --cache-gib 4 --memory-limit-gib 84 \
  --async-refinements --hot-fraction 0.04 --prefill-chunk-size 64
```

Omit `--base-cache-gib` to load the complete 55.46 GiB base. `--guard-above-default-cap` raises the process guard to `--memory-limit-gib` after loading. Use it only under an external host watchdog.

The arithmetic and row alignment have unit tests:

```sh
cd macos && ../.venv-macos/bin/python -m unittest test_kld_math test_kld_alignment -v
```

## Recommended practice on a 128 GiB Mac Studio

These rules come from the measurements in [performance measurements](PERFORMANCE.md).

**Before a run**

- Verify the checkpoint checksums again after any download or deletion. A stored verification record can outlive the files it describes.
- Measure free plus file-backed memory with `vm_stat`. Do not rely on the free percentage from `memory_pressure`, which counts inactive anonymous pages. On 10 November it reported 86% free while only 75.5 GiB was reclaimable.
- Keep a 16 GiB host reserve. Full-base residency on the reference runtime needs about 98 GiB of free plus file-backed memory at launch for a 4 GiB refinement cache.
- Do not stop unrelated applications from a benchmark. The machine's owner decides that.
- Set `LC_ALL=C` for any script that compares `ps -o lstart` output. The date format follows the locale, and a mismatch reads as a missing process.

**Sizing**

- One refinement slot uses about 8.3 MB. One GiB of refinement cache holds about 1% of the 12,096 slots, or 2.9 experts per layer.
- Increase the refinement cache in steps. Inspect the peak footprint and the lowest host headroom after each step.
- Use layer-major prefill for prompts. It executes every routed expert at 4 bits within a bounded footprint.

**Measuring quality**

- Check the KL arithmetic against published tokenwise values before trusting new numbers. `kld_math.tokenwise_kld` reproduced a published window mean to 4e-15.
- Score decode on text that the model generated, with thinking off, not on stock text.
- Check reference alignment before scoring. The reference's top prediction matched the generated next token at 92% to 99% of positions. A one-position shift matched under 1%.
- Run an all-4-bit decode arm. It gives the noise floor between decode and prefill.
- Report an interval for each mean. With 1,280 positions, the 95% interval is about ±0.03, so the third decimal does not decide a goal.
- Report desired and executed hot share, asynchronous fallbacks and synchronous base reads with each decode value. A bounded base cache slows steps and flatters asynchronous readiness.
- State the reference. A decode value against a 4-bit reference excludes the 4-bit error against BF16.

## Benchmark requirements

Verify the checkpoint and run the short answer checks before benchmarking a configuration:

```sh
.venv-macos/bin/python macos/verify_generation.py \
  --model "$MODEL_ROOT" --output generation-async-2pct.json \
  --resident-base --base-cache-gib 32 --async-refinements --hot-fraction 0.02 \
  --cache-gib 4 --memory-limit-gib 84 --prefill-chunk-size 8
```

After a passing smoke and recorded memory headroom, run the same configuration through the [guarded benchmark procedure](ASYNC.md). Start with 16 tokens and three measured requests. Extend the length only after inspecting actual memory use. Use fresh output files and directories.

The driver discards one warm-up and preserves all measured responses and token IDs. Every measured request must reach its requested length. An early end preserves evidence but prevents a successful throughput summary. Decode throughput excludes prefill and the first generated token.

The [historical all-hot and canonical checks](results/2026-11-04-smoke.json) passed three cases at source `1df6588`. The 24 GiB resident-hot experiment was stopped under memory pressure. Do not repeat it as a recommended configuration. Bounded 1%, 2%, and 4% results are in [performance measurements](PERFORMANCE.md).

Keep prompts, output limits, sampling, chunking, cache budgets, and warm-up procedures fixed within a comparison. Record desired and executed precision, synchronous base misses, evictions, SSD reads, and peak process memory. Reject candidates that fail answer or coding checks. Short smoke tests do not establish general quality.

Reports separate logical tensor reads from process-attributed disk reads. The expert-forward timer includes loading. Reader time can overlap execution and must not be added to elapsed time. Process peaks include model loading and earlier requests. A process restart does not clear the operating-system file cache.

These reference benchmarks cover one greedy text sequence with a 4,096-token validation ceiling. The native chat and MTP experiments use separate commands and measurements.

## Lossless resident archive experiment

The optional `nestquant-resident-base-v2` format packs eight binary signs per byte. It also aliases identical gate/up low-rank correction matrices. Conversion rejects nonbinary signs or unequal corrections. It checks source hashes and compares every reconstructed byte after saving each layer. Existing archives remain unchanged.

The compact format supports `native_assets.read_bank` and banked projections. Paged banks and scalar projections reject it. Full-generation integration remains pending. Do not infer safe residency or throughput from the format change.

1. Prepare a separate archive with at least 100 GiB of remaining disk reserve.
2. Verify real GPU projections against the original archive and scalar references.
3. Run the supervised resident-memory probe before attempting generation.

```sh
.venv-macos/bin/python macos/compact_base.py --archive "$BASE_ARCHIVE" --output-dir "$COMPACT_BASE_ARCHIVE"
.venv-macos/bin/python macos/verify_compact_base.py --model "$MODEL" --archive "$BASE_ARCHIVE" --compact-archive "$COMPACT_BASE_ARCHIVE" --output compact-base-verification.json
```

Conversion and GPU verification enforce a 6 GiB process guard. The local supervisor uses an 8 GiB outer cap and preserves pressure/free-memory checks. All 88 runtime tests and five supervisor tests passed. The first suite invocation failed four imports because its optional dependency path was absent: `ModuleNotFoundError: No module named 'mlx_vlm'`. The corrected invocation used the existing dependencies without installing packages. The full-base memory probe stopped at layer 42 under pressure. No generation result is claimed for this format. See [the conversion and GPU proof](results/2026-11-05-native-lossless-base.json).

The optional `microbench_banked.py --stages` probe compares full, compiled, and raw Metal projection timings. Raw timing uses already evaluated rotated inputs. These timings are not an additive latency ledger. The probe checks output stability and keeps compiled projections outside generation.

Use `microbench_banked.py --decoder-variants` to compare the original integer decoder with packed reads and a sum lookup. The probe checks eight cached experts at layers 3 and 44. It covers base, refined, mixed, and cold modes under the same 4 GiB guards. It does not change generation defaults.

The decoder comparison also includes a cooperative SIMD variant. Each lane group loads a compressed ring once and shares its words through lane shuffles. The byte decoder remains the default.

## Fast decode with DFlash2 speculative decoding (2026-11-14)

This section records the fastest configuration measured so far on the 128 GiB M5 Max, on AC power. It keeps
the full 1.5-bit base resident and the 4-bit refinement hot set at 15% of expert slots.

### What changed

- `decoder.py`: the routed-expert kernel decodes each lane's 16 consecutive ring positions from directly
  loaded 32-bit words, builds the fp16 value `1024 + code` from its bits (`0x6400 + code`), applies the
  affine map with one `half2` fma per weight pair, reads (A', C) for 4-bit rows from a 65,536-entry table,
  and shares one decode across rows routed to the same expert (verify blocks). Outputs are bit-identical to
  the previous kernel except for float32 summation order (`LANE_POSITIONS = 8` restores bit-exact output).
- `native_generate.py` / `native_refinements.py`: decode routes 4-bit refinements on the GPU through a
  per-layer slot table, so a decode step has no per-layer host round trip; refinement reads are published at
  token boundaries; the predictor runs while the GPU evaluates the next token (pipelined loop).
- `backbone_fast.py`: fused hyper-connection kernels, block attention and earlier GPU submission for steps of
  1–8 tokens.
- `native_prefill.py`: batched layer-major prefill (all routed experts at 4 bits), up to 64K tokens.
- `native_dflash.py`: DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`) and DSpark drafters through mlx-vlm's DFlash
  loop, with target hidden-state capture for glm5_next (output of layer i, hyper-connection streams
  averaged), an adaptive verify-block policy fitted to measured round cost, and predictor commits of accepted
  rows only.

### Recommended command

```sh
.venv-macos/bin/python macos/native_dflash.py --model "$MODEL" --archive "$BASE_ARCHIVE" \
  --backbone "$HOME/Models/GLM/backbone-affine4-e8-v1" --drafter "$HOME/Models/GLM/drafters/dflash2" \
  --hot-fraction 0.15 --budget-gib 16 --batched-prefill --blocks -8 \
  --prompt-file prompt.txt --tokens 512 --output result.json
```

Prepare the 4-bit backbone with `native_backbone.py --bits 4 --embedding-bits 8`. Add `--draft-rows-base`
to verify draft rows with the 1.5-bit base only; it is faster, but it raises acceptance on some tasks, which
means some committed tokens are effectively decided at 1.5 bits. Its task quality has not been validated.

### Coding benchmark (192 generated tokens per task, decode tok/s)

| Task | Plain decode | DFlash2 adaptive | DFlash2 adaptive, `--draft-rows-base` |
|---|---:|---:|---:|
| Copy and edit a class | 20.3 | 40.1 | 62.7 |
| Fix a bug | 18.5 | 46.9 | 51.1 |
| Refactor with type hints | 21.3 | 38.3 | 55.2 |
| New function from a spec | 22.7 | 26.3 | 42.3 |
| Write unit tests | 22.6 | 30.1 | 35.0 |
| Explain code in prose | 22.6 | 22.0 | 25.3 |
| Average | 21.3 | 33.9 | 45.3 |

Peak process footprint was 84.5 GiB. Batched prefill brought time to first token to 9–18 s for prompts of
100–1,300 tokens; on long prompts it measured 403 tok/s at 45K tokens. Single runs vary by about ±15%
because different outputs change the acceptance path. The audit in `dflash_diag.py` found acceptance
consistent with the drafter itself: target precision changes it by 1–3%, and prompts change it from 51% to
75% per draft.
