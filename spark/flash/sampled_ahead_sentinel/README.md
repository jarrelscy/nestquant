# Accepted Spark sampled-ahead runtime — 2026-10-13

The accepted native Spark configuration measured **decode KLD 0.09006743684523977** against the BF16 teacher across confirmation windows 0000–0003: **8,188/8,188 positions**, top-1 agreement **90.3273%**, student perplexity **2.87414**, teacher perplexity **2.77338**. KLD is full-vocabulary KL(teacher || student), using teacher-forced decode and the included scorer.

A separate **single 512-token sampled generation probe** measured **21.24687885 decode tokens/s**. It used a one-token prompt, temperature 0.7/top_p 0.95, seed 1234, MTP2, no teacher forcing and no requested logprobs. The formula is `(completion_tokens - 1) / engine_decode_seconds`, using the request decode-time histogram delta. This is not a full benchmark throughput result or a general workload guarantee.

**SparkBench TrueScore 87.5 belongs to the earlier readfirst configuration.** The full SparkBench rerun for this accepted runtime is pending. Its result must not be inferred from KLD or this speed probe. LMPF and subsequent device-publication candidates are not included.

## Exact runtime and budgets

`tested-files.sha256.json` pins byte-identical files from the accepted run. The runtime advances jT from sampled tokens before the next forward, retains/reuses the 512 MiB startup sentinel through prewarm/profile, then releases it with corrected memory accounting. It includes the fresh-state KDA initialization fix and the tested loader/indexer patches. No extra permanent model allocation is introduced by the sentinel.

The base image is **linux/arm64**:

```
docker.io/jarrelscy/nestquant@sha256:3859a4e0018ec915ca0d4a3f9488158026fc2525534051433e2b37a729102364
```

The accepted overlays are bind-mounted over that immutable image; pulling the image alone does not include these updates. The unchanged image supplies the verified post-layer publication and publication observer modules. Startup fails closed if runtime/module hashes do not match.

Preserved settings: U050 flat50 allocation, **2,630 active slots = 526 fixed + 2,104 floating**, 8 spares, jT75%/EMA25% updated every committed token, G1, half-life64/hysteresis4, TP1/one sequence, MTP2 probabilistic, 262,144 context, 2 GiB KV, FP4-g16 MLA/FP8 KV, eager, prefix caching off, native uncapped, temperature0.7/top_p0.95/thinking off. Memory guard: below4GiB immediate stop, or15seconds below5GiB/above50% full-memory PSI. Host ZFS telemetry is required; do not bypass guards on another platform.

## Start serving

Use native ARM64 DGX Spark with Docker GPU support and the tested ZFS host-memory telemetry exposed at `/proc/spl/kstat/zfs/arcstats`. Start on an idle machine. Keep at least400GB disk for checkpoint, packed weights and caches; diagnostic teacher/student logits need additional space.

```bash
git clone --branch spark-flash https://github.com/jarrelscy/nestquant.git
cd nestquant
git checkout spark-sampled-ahead-sentinel-20261013
export NQ_FLASH_DATA="$HOME/nq-flash"
export NQ_FLASH_IMAGE='docker.io/jarrelscy/nestquant@sha256:3859a4e0018ec915ca0d4a3f9488158026fc2525534051433e2b37a729102364'
export NQ_FLASH_CONTAINER=nq-spark-kld-sampled-ahead-sentinel
# Set VLLM_API_KEY privately in your environment; do not put it in shell history.
: "${VLLM_API_KEY:?Set your serving API key privately}"
docker pull "$NQ_FLASH_IMAGE"
# Skip fetch/prepare when these exact checkpoint and packed files already exist.
bash spark/flash/sampled_ahead_sentinel/run_spark_kld.sh fetch
bash spark/flash/sampled_ahead_sentinel/run_spark_kld.sh prepare
python3 spark/flash/sampled_ahead_sentinel/recipe.py stage
python3 spark/flash/sampled_ahead_sentinel/recipe.py up
docker logs -f "$NQ_FLASH_CONTAINER"
```

`fetch` pins model revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`. `prepare` is CPU-only and packs one layer at a time; retain the checkpoint because the backbone references unchanged files. Do not prepare weights while serving. The staged exact overlays live at `$NQ_FLASH_DATA/kld-sampled-ahead-sentinel`. The launcher verifies hashes, architecture, idle-container state and guards before starting. API: `http://127.0.0.1:8001/v1`, model `glm-5.3-flash-nq`. Keep the listener local or use an authenticated tunnel.

The capture hook does not force tokens unless its `force` file exists. Normal serving after `up` does not create that file. For graceful shutdown with evidence preserved:

```bash
python3 spark/flash/sampled_ahead_sentinel/recipe.py down
```

Archive previous diagnostic directories and remove/rename the stopped named container explicitly before a fresh diagnostic. The wrapper refuses to overwrite a prior run. The portable wrapper was syntax and staging/hash checked; the unchanged mounted runtime, capture client and scorer are the files actually exercised on Spark. A new end-to-end GPU execution of this wrapper is not claimed.

## Reproduce the four-window decode KLD and speed probe

Download teacher artifacts on the idle Spark **before** `up` (or download on another CPU machine and transfer the verified files):

```bash
python3 spark/flash/sampled_ahead_sentinel/recipe.py fetch-teacher
python3 spark/flash/sampled_ahead_sentinel/recipe.py up
python3 spark/flash/sampled_ahead_sentinel/recipe.py kld
python3 spark/flash/sampled_ahead_sentinel/recipe.py score
```

Teacher dataset: `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`, immutable revision `95f4fdd94bf29989db2e0d1054e4931f55edb6aa`. Fetch verifies SHA256 for all four teacher files and token arrays against the accepted run. The client checks startup sentinel reuse/release, THP restoration, exact2047positions per window, no missing/duplicate positions and token identity. It then runs the separate512-token speed probe with forcing absent. The client has a one-hour host deadline and shuts down serving before CPU scoring. The scorer requires all four windows and8188positions; partial panels cannot report complete.

Outputs remain in the local data directory: `results-remote.json`, `speed-result.json`, `output/`, routing/publication/timing observations, guard logs and module provenance. They are not bundled into this Git repository. Public trace publication is managed separately on Hugging Face.

## Validation

```bash
python3 spark/flash/sampled_ahead_sentinel/test_sentinel.py
python3 spark/flash/sampled_ahead_sentinel/recipe.py stage
```

The stage command checks every tested source hash and copies those exact bytes. Existing SparkBench reproduction instructions and historical evidence remain in [the prior recipe](../README.md#reproduce-the-benchmark); use this runtime for the pending rerun, without attributing the earlier87.5 score to it.
