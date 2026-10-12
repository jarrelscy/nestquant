# GLM-5.3-Flash NestQuant — single DGX Spark

**Spark Bench TrueScore: 87.5 · Quality: 85.4 · 160/160 runs completed · Zero errors · PASS**

Measured on one native ARM64 DGX Spark GB10 on 2026-10-12: all 80 scenarios, two repeats each, using the official Spark Bench scorer on a separate machine. The run used native uncapped serving and the configuration below.

[Results, official scores, all 160 transcripts, routing telemetry and hashes](https://github.com/jarrelscy/spark-bench/tree/1a3c1f3534a55bbfe794beb2f884c2b30687d105/results/nestquant-spark-native-readfirst-20261012) · [Validated runtime source](https://github.com/jarrelscy/nestquant/tree/spark-readfirst-full80-20261012)

NestQuant stores a 1.5-bit base for routed experts and loads residuals to execute selected experts at 4 bits. This model requires the NestQuant runtime; stock Transformers, vLLM and MLX cannot load it directly.

## Configuration that scored 87.5

| Setting | Validated Spark default |
|---|---|
| Predictor | 75% jT / 25% EMA, updated every committed token (G=1) |
| Active expert slots | 2,630: 526 fixed (20%) + 2,104 floating; 8 additional spares |
| Layer allocation | U050 / `spark_256K_U_2630_flat50` |
| EMA / hysteresis | Half-life 64 / 4 |
| Speculative decoding | MTP2, probabilistic |
| Context / KV budget | 262,144 tokens / 2 GiB |
| Attention cache | FP4-g16 MLA; FP8 KV |
| Serving | TP1, one sequence, eager, prefix cache disabled |
| Sampling | Temperature 0.7, top_p 0.95, thinking off |
| Pacing | Native, uncapped |

The measured median positive generation-log interval was **17.2 tokens/s**, with a maximum of **22.2 tokens/s**. These are serving-log intervals, not an isolated decode throughput test; the benchmark's dedicated throughput sweep was skipped.

## Start on Spark

The validated **linux/arm64** image is published as `docker.io/jarrelscy/nestquant:spark-flash-jt75-g1-20261011`. The commands below pin its verified immutable digest.

Requirements: native ARM64 Spark, CUDA 13-capable drivers, Docker with GPU support, and at least 400 GB of disk for the checkpoint, packed weights, backbone and caches. The tested memory guard requires host ZFS telemetry at `/proc/spl/kstat/zfs/arcstats`; this recipe fails closed when that source is absent. Other host memory-monitoring environments have not been validated by this run.

```bash
git clone --branch spark-flash https://github.com/jarrelscy/nestquant.git
cd nestquant
git checkout spark-readfirst-recipe-20261012
export NQ_FLASH_DATA="$HOME/nq-flash"
export NQ_FLASH_IMAGE="docker.io/jarrelscy/nestquant@sha256:3859a4e0018ec915ca0d4a3f9488158026fc2525534051433e2b37a729102364"
# Set VLLM_API_KEY privately in your environment.
: "${NQ_FLASH_IMAGE:?Validated image digest is required}"
: "${VLLM_API_KEY:?Set your serving API key}"
docker pull "$NQ_FLASH_IMAGE"
bash spark/flash/readfirst/run_spark.sh fetch
bash spark/flash/readfirst/run_spark.sh prepare
bash spark/flash/readfirst/run_spark.sh up
bash spark/flash/readfirst/run_spark.sh logs
```

`fetch` pins checkpoint revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`. `prepare` packs TP1 weights on CPU, one layer at a time, and creates the FP8 backbone overlay. Keep the downloaded checkpoint because the overlay symlinks unchanged files. Do not run preparation or large image transfers alongside serving. Subsequent starts reuse prepared files and need only `up`.

The OpenAI-compatible endpoint is `http://127.0.0.1:8001/v1`, model `glm-5.3-flash-nq`. Wait for serving readiness in the logs. Use an authenticated SSH tunnel for access from another machine.

The launcher starts the memory guard. It stops serving below 4 GiB available host memory immediately, or after 15 seconds below 5 GiB or above 50% full memory PSI. Preserve these guards and the model budgets. Spark host and GPU allocations share physical RAM. A completed benchmark does not guarantee startup under competing host workloads.

## Reproduce the benchmark

Run scoring on a separate machine, using Spark Bench commit `f2ac7876c1782047fe4ac37290a1466daa3a1c05`. Install that checkout's dependencies. Forward local port 18975 to the Spark's localhost port 8001 and set `SPARK_BENCH_API_KEY` privately to the serving key.

```bash
export SPARK_BENCH_FORCE_TEMPERATURE=0.7
export SPARK_BENCH_EXTRA_BODY='{"top_p":0.95,"chat_template_kwargs":{"enable_thinking":false}}'
export SPARK_BENCH_CONTEXT=262144
export SPARK_BENCH_SEED_PER_REPEAT=42
unset SPARK_BENCH_REASONING_EFFORT
python -u spark_bench.py eval --endpoint http://127.0.0.1:18975/v1 \
  --model glm-5.3-flash-nq --label spark-native-full80-readfirst \
  --topology single-dgx-spark-gb10 --parallelism TP1 \
  --spec-decode mtp2-probabilistic --repeats 2 --temperature 0.7 \
  --thinking off --uncapped --tier all --skip-throughput \
  --timeout 7200 --out-dir results
```

This runs all 80 scenarios with repeat seeds 43 and 44. Generated trajectories can differ despite matched settings. The published evidence includes the exact scorer outputs and per-repeat transcripts, so 87.5 is an observed result, not a promised score for every rerun.

The local SM120 reference scored TrueScore 87.03197 and quality 84.01199 with a 35-token/s cap; it is a different hardware/pacing comparison. Matched Spark decode KLD for the current 75/25 setup is pending measurement.
