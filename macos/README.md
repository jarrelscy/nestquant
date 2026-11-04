# Apple silicon runtime

This experimental runtime reads the published GLM-5.3-Flash NestQuant 1.5/4 checkpoint directly. Metal kernels decode the expert weights and compute expert projections. PyTorch MPS runs the Transformers text backbone.

**Status, 2026-11-04:** full-checkpoint generation passed three smoke checks at both all-hot and canonical mixed-precision allocations on an Apple M5 Max with 128 GiB memory. Decoder, expert, and matching-chunk backbone checks pass. The first formal benchmark reached 0.227 tokens/second median. General quality evaluation and hot-set comparisons remain pending. See [performance measurements](PERFORMANCE.md). The unchunked BF16 comparison has a documented tolerance failure below.

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

The generation interface supports the following command. The tested configuration retains base experts:

```sh
.venv-macos/bin/python macos/generate.py \
  --model "$MODEL_ROOT" \
  --prompt 'Reply with exactly: GLM_MAC_OK' \
  --max-tokens 16 \
  --hot-fraction 0.21742724867724866 \
  --cache-gib 4 --resident-base \
  --prefill-chunk-size 256 \
  --output generation.json
```

The default allocation is 2,630 hot experts across 42 routed layers. The predictor mixes 75% jT with 25% exponential moving average. It refreshes after every committed token. Twenty percent of allocated slots are fixed experts.

`--hot-fraction` changes the total hot-expert allocation. Other allocations preserve the canonical layer weighting approximately, with integer counts between two and 288 experts per layer. `--cache-gib` separately bounds cached expert tensors. Cold experts load only the base planes. Hot experts additionally load their refinements.

The default mode uses a bounded least-recently-used cache for selected experts. It does not retain every base expert. Add `--resident-base` to preload and retain all base experts before inference. This retains 59,052,758,208 bytes (55.00 GiB) outside `--cache-gib`. In this mode, the bounded cache holds refinements and shares base tensors. Evicting a refinement does not evict its base. The all-hot smoke run with resident base tensors and a 4 GiB refinement cache peaked at 81.62 GiB of process physical footprint.

These two modes load refinements when a routed expert needs them. A larger hot fraction changes reconstruction precision without guaranteeing a matching increase in resident memory. Measure cache behavior and SSD reads together.

Add `--resident-hot` with `--resident-base` to keep every predicted hot expert's refinements resident during decode. Before each decode step, the runtime removes refinements outside the predicted set and loads missing refinements. Refinements that remain hot retain their existing buffers. Cold experts use the resident base tensors. Updates are synchronous and their cost is included in decode timing.

The runtime checks the worst possible refinement allocation for the configured per-layer slot counts before loading the backbone. An insufficient `--cache-gib` fails instead of silently evicting hot experts. The canonical 2,630-slot allocation requires at most 20.314 GiB of refinements. Use a 24 GiB cache for its initial resident-hot test. This requirement excludes base tensors, the backbone, attention caches, and temporary allocations. Full-model memory and performance remain unverified.

Add `--wired-memory` to keep MLX allocations resident during inference up to the device's recommended working set. The runtime restores the previous process limit when it closes. This uses [MLX memory wiring](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html) without changing system limits. Wiring does not reduce the model's memory requirement. Its performance effect remains under measurement.

Prefill processes every prompt chunk through one layer before advancing. Every selected prefill expert uses its 4-bit reconstruction. The runtime retains that layer's cached tensors between chunks when the cache budget permits. This avoids reducing prefill precision to the decode allocation. It does not imply equality with the unquantized model.

The initial interface supports one text sequence, greedy sampling, and a 4,096-token validation ceiling. Vision, speculative decoding, an HTTP endpoint, and quantized attention caches are not implemented. The backbone retains the checkpoint's original nonexpert precision, with FP8 weights expanded for MPS. This differs from the additional backbone quantization in the published Spark recipe.

Generation uses the repository's `spark/flash/thinking_off.jinja` template by default. Add `--thinking` to use the checkpoint's reasoning template. The checkpoint template ignores `enable_thinking=false`, so passing that keyword alone does not disable reasoning.

If the loader reports a mapping or shape mismatch, stop inference and check the checkpoint revision and dependency versions. Do not ignore missing weights. If memory pressure rises, reduce the expert cache and prefill chunk size before retrying. Remove `--resident-base` when the base allocation leaves insufficient memory.

## Benchmark requirements

Run the full-model smoke checks after the download and checksum verification finish:

```sh
.venv-macos/bin/python macos/verify_generation.py \
  --model "$MODEL_ROOT" --hot-fraction 1 --resident-base --cache-gib 4 \
  --output generation-all-hot.json
.venv-macos/bin/python macos/verify_generation.py \
  --model "$MODEL_ROOT" --hot-fraction 0.21742724867724866 \
  --resident-base --cache-gib 4 --output generation-canonical.json
.venv-macos/bin/python macos/verify_generation.py \
  --model "$MODEL_ROOT" --hot-fraction 0.21742724867724866 \
  --resident-base --resident-hot --cache-gib 24 \
  --output generation-resident-hot.json
```

The first two commands passed all three cases. The third awaits full-model validation with the complete hot set resident. See the [recorded smoke results](results/2026-11-04-smoke.json) for outputs, resource measurements, and source revision. They test an exact reply, integer arithmetic, and JSON ordering. They do not establish general model quality. The driver saves generated text and reports a failed check without changing its expected answer.

The benchmark driver completed the documented baseline measurement. Full generation must succeed before benchmarking a different configuration:

```sh
.venv-macos/bin/python macos/benchmark.py \
  --model "$MODEL_ROOT" \
  --output-dir benchmark-canonical-cache4 \
  --tokens 128 --runs 3 \
  --hot-fraction 0.21742724867724866 \
  --cache-gib 4 --prefill-chunk-size 256
```

The driver discards one warm-up and records three measured requests. It preserves each response and its token IDs. Every measured request must reach the requested output length. An early stop preserves that run's evidence but prevents a successful throughput summary. Choose a prompt that produces a sufficiently long answer at every compared setting. Decode throughput excludes prefill and the first generated token. A fresh output directory prevents earlier measurements from being overwritten.

For tuning, use this sequence:

1. Stop other model servers. Wait for this checkpoint's download to finish. Suspend other downloads during measurements. Resume them afterward.
2. Compare bounded storage with `--resident-base` at the canonical hot fraction and a 4 GiB cache.
3. Compare hot fractions of 10%, 21.74%, and 35% with the same storage mode and cache budget.
4. Compare cache budgets of 4, 16, and 32 GiB at the most useful precision setting.
5. Compare resident-base storage with and without `--resident-hot`, using the canonical fraction and the same 24 GiB cache.
6. With `--resident-hot`, compare hot fractions of 10%, 15%, and 21.74% using a 24 GiB cache.
7. Compare prefill chunks of 128, 256, and 512 tokens using the same longer prompt.
8. Check each candidate's answers and coding tests before selecting a default.

Keep prompts, output limits, sampling, and warm-up procedures identical within each comparison. Record prefill throughput, decode throughput, peak memory, SSD reads, and output correctness. These candidate settings are a proposed experiment, not measured recommendations.

Increase cache budgets only when measured memory leaves room for runtime allocations. A 35% resident-hot allocation requires at most 32.696 GiB of refinements. On the tested Mac, this approaches the recommended Metal working set after adding the backbone and base tensors. Lower fractions are the initial resident-hot candidates. Prefill still uses its configured bounded cache for selected 4-bit experts, so its memory peak can exceed the subsequent hot-set allocation.

Reports separate logical base and refinement tensor bytes from process-attributed disk reads. Operating-system caching can satisfy tensor reads without physical SSD reads. The expert-forward timer includes expert loading. The process memory peak includes model loading and earlier requests in that process. Warm-up does not guarantee that every expert is cached, and a process restart does not clear the operating-system file cache.

`hot_sync_seconds` measures synchronous hot-set updates. It overlaps the loader's timing and is already included in decode throughput. Do not add these timers together. Request reset restores the predictor policy, while the weight cache can retain tensors. Layer-major prefill can replace cached refinements before the next decode pool is established.

Full-model smoke results are recorded above. The first formal throughput result is recorded in [performance measurements](PERFORMANCE.md). Hot-fraction comparisons and recommended performance settings remain pending.
