#!/usr/bin/env bash
# Single-node Flash only. No other containers are stopped or recreated.
set -euo pipefail
ROOT=${NQ_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
DATA=${NQ_FLASH_DATA:-$HOME/nq-flash}
IMAGE=${NQ_FLASH_IMAGE:?Set NQ_FLASH_IMAGE to the published immutable ARM64 digest}
NAME=${NQ_FLASH_CONTAINER:-nq-spark-flash}
THINKING=${NQ_THINKING:-off}
mkdir -p "$DATA"
DATA=$(cd "$DATA" && pwd)
common=(--security-opt seccomp=unconfined --cap-add IPC_LOCK --ipc host -v "$DATA:/data")
case "$THINKING" in
 on)
  [[ -f "$ROOT/spark/flash/sampled_ahead_sentinel/thinking_on.jinja" ]] || { echo 'Missing pinned thinking-on template.' >&2; exit 1; }
  thinking_mount=(-v "$ROOT/spark/flash/sampled_ahead_sentinel/thinking_on.jinja:/diag/thinking_on.jinja:ro")
  chat_template=/diag/thinking_on.jinja
  ;;
 off)
  thinking_mount=()
  chat_template=/nq/spark/flash/thinking_off.jinja
  ;;
 *) echo 'NQ_THINKING must be on or off.' >&2; exit 2 ;;
esac
case ${1:-help} in
 build) docker build -f "$ROOT/spark/flash/Dockerfile" -t "$IMAGE" "$ROOT" ;;
 fetch)
  docker run "${common[@]}" -e HF_TOKEN --entrypoint python "$IMAGE" -c 'from huggingface_hub import snapshot_download; snapshot_download("jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit", revision="4023870a6f2c6097e37fa5d2e54040db64b011e9", local_dir="/data/checkpoint", ignore_patterns=["experiments/**", "evaluations/**"])'
  ;;
 prepare)
  # CPU-only, one layer at a time; do not run concurrently with serving on Spark.
  docker run "${common[@]}" -e NQ_REPACK_DEVICE=cpu -e NQ_BUILD=/data/build -e NQ_DEFS=NQ_RK_CODES=0x405,NQ_RK_GU=0x5,NQ_RK_DN=0x401,NQ_BK_CODES=0x21,NQ_SWIGLU_LIMIT=10 --entrypoint bash "$IMAGE" -c '
    python spark/flash/nq_flash_layout.py /data/checkpoint
    python streaming/repack.py /data/checkpoint /data/repack-tp1 1 3-44
    python spark/flash/nq_flash_layout.py /data/checkpoint --repack /data/repack-tp1
    python spark/flash/convert_backbone.py /data/checkpoint /data/fp8-backbone'
  ;;
 up)
  # This recipe retains the tested LXC host-memory source and thresholds.
  [[ -r /proc/spl/kstat/zfs/arcstats ]] || { echo 'Required host memory telemetry is unavailable.' >&2; exit 1; }
  # Fetching, hashing, and packing ~183 GiB leaves enough clean checkpoint pages
  # resident to exhaust Spark unified memory while the expert planes are mapped.
  # Evict only clean page cache after all writes are durable; never weaken the
  # memory guard or rely on the kernel OOM path.
  sudo -n true >/dev/null 2>&1 || { echo 'Passwordless sudo is required to release build page cache before serving.' >&2; exit 1; }
  sync
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  python3 - <<'CHECK_MEMORY'
import pathlib
v={s[0]:int(s[2]) for line in pathlib.Path('/proc/spl/kstat/zfs/arcstats').read_text().splitlines()[2:] if len(s:=line.split())==3}
if v['memory_available_bytes']<5*1024**3:raise SystemExit('Insufficient host memory headroom to start')
CHECK_MEMORY
  [[ $(uname -m) == aarch64 ]] || { echo 'This launcher targets native aarch64 DGX Spark.' >&2; exit 1; }
  [[ -f $DATA/fp8-backbone/config.json && -f $DATA/repack-tp1/rank0.json ]] || { echo 'Run fetch and prepare first.' >&2; exit 1; }
  ns=${NUM_SPEC:-2}; [[ $ns == 1 || $ns == 2 ]] || { echo 'NUM_SPEC must be 1 or 2.' >&2; exit 1; }
  # Default KV ceiling is explicit because host-mapped slots consume unified RAM.
  # vLLM must still validate max_model_len; it must not silently shrink the preset.
  mkdir -p "$DATA/readfirst-runtime" "$DATA/cache"
  docker run -d --name "$NAME" "${thinking_mount[@]}" -v "$DATA/kld-sampled-ahead-sentinel/startup_sentinel.py:/diag/startup_sentinel.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/sparse_attn_indexer_kpool.py:/opt/vllm/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/packet_timing.py:/diag/packet_timing.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/loader_memory_diagnostic.py:/diag/loader_memory_diagnostic.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/prewarm_mla.py:/diag/prewarm_mla.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/nq_flash_plugin.py:/nq/spark/flash/nq_flash_plugin.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/startup_thp.py:/diag/startup_thp.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/gdn_attn_initcheck.py:/opt/vllm/vllm/v1/attention/backends/gdn_attn.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/runtime.py:/nq/spark/flash/nq_flash_runtime.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/jt_runtime.py:/nq/spark/flash/jt_runtime.py:ro" -v "$DATA/kld-startup-cache/torchinductor_root:/tmp/torchinductor_root" -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root "${common[@]}" --gpus all -p "127.0.0.1:${NQ_PORT:-8001}:8000" \
   -v "$DATA/kld-sampled-ahead-sentinel/sitecustomize.py:/diag/sitecustomize.py:ro" -v "$DATA/kld-sampled-ahead-sentinel/nq_spark_kld.py:/diag/nq_spark_kld.py:ro" -e NQ_KLD_DIR=/data/kld-sampled-ahead-sentinel/output -e NQ_KLD_FKNOB=/data/kld-sampled-ahead-sentinel/force -e NQ_FLASH_JT_TRACE=/data/kld-sampled-ahead-sentinel/jt-trace.pkl -e NQ_KLD_RING=64 -e PYTHONPATH=/diag:/nq/spark/flash -v /proc/spl/kstat/zfs/arcstats:/diag-host-arcstats:ro -e NQ_PUBLICATION_OBSERVER=/data/kld-sampled-ahead-sentinel/publication.jsonl -e VLLM_API_KEY -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=2 -e OMP_NUM_THREADS=4 \
   -e NQ_FLASH_MODEL=/data/fp8-backbone -e NQ_REPACK=/data/repack-tp1 -e NQ_BUILD=/data/build \
   -e NQ_FLASH_PRESET="${NQ_FLASH_PRESET:-spark_256K_U_2630_flat50}" -e NQ_UNIFIED=1 -e NQ_FLASH_SPARE_SLOTS=8 \
   -e NQ_FLASH_PREFETCH=throughput -e NQ_FLASH_PREFETCH_MAX_PENDING=64 \
   -e NQ_FLASH_PREFETCH_MAX_DEMOTIONS=32 -e NQ_FLASH_PREFETCH_SECONDS=0.1 \
   -e NQ_FLASH_WAIT_FOR_UPGRADES=0 \
   -e NQ_FLASH_FIXED_FRACTION="${NQ_FLASH_FIXED_FRACTION:-0.2}" \
   -e NQ_FLASH_JT_PARAMS='{"mix":0.75,"hm":4.0,"G":1,"half_life":64.0}' \
   -e NQ_FLASH_MLA_CACHE="${NQ_FLASH_MLA_CACHE:-fp4_g16}" \
   -e NQ_FLASH_ROUTING_STATS=/data/kld-sampled-ahead-sentinel/routing.json -e NQ_FLASH_TPS="${NQ_FLASH_TPS:-0}" \
   -e NQ_DEFS=NQ_RK_CODES=0x405,NQ_RK_GU=0x5,NQ_RK_DN=0x401,NQ_BK_CODES=0x21,NQ_SWIGLU_LIMIT=10 \
   -v "$DATA/cache:/root/.cache" "$IMAGE" \
   serve /data/fp8-backbone --served-model-name glm-5.3-flash-nq \
   --tensor-parallel-size 1 --max-num-seqs 1 --max-model-len "${NQ_MAXLEN:-262144}" \
   --gpu-memory-utilization 0.50 --kv-cache-memory-bytes "${NQ_KV_BYTES:-2147483648}" \
   --max-num-batched-tokens 1024 --enforce-eager --no-enable-prefix-caching --kv-cache-dtype fp8 \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$ns,\"draft_sample_method\":\"probabilistic\"}" \
   --chat-template "$chat_template" \
   --override-generation-config '{"temperature":0.7,"top_p":0.95}' \
   --trust-remote-code --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
   --limit-mm-per-prompt '{"image":1,"video":0}'
  ;;
 logs) docker logs --tail 100 -f "$NAME" ;;
 *) echo 'Usage: spark/flash/run_spark.sh build|fetch|prepare|up|logs'; exit 2 ;;
esac
