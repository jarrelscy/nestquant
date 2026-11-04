# Apple silicon runtime

This experimental runtime reads the published GLM-5.3-Flash NestQuant 1.5/4 checkpoint. The native path uses a prepared MLX backbone and bounded Metal expert banks. The reference path uses the Transformers text backbone on PyTorch MPS.

**Status, 2026-11-05.** The native path reached 3.52 tokens/s on a fixed 128-token replay at 45.73 GiB peak process footprint. The reference reached 1.27 tokens/s on the same inputs at 57.87 GiB. A separate native mixed run passed HumanEval/3 at 2.04 tokens/s. See [measurements and limits](PERFORMANCE.md#matched-native-replay).

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

Use `--verify-prefill --refinements` to compare both schedules with four-bit prefill and check the committed predictor state. The comparison uses the same chunk size for both schedules. Choose `--prefill-rows` to bound the scratch bank for one layer. Decode slots, this scratch bank, and the asynchronous worker share a 4 GiB refinement limit. The separate base banks have a 32 GiB limit. A configuration that exceeds either limit fails before prefill.

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
