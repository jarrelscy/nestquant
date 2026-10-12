#!/usr/bin/env bash
# Single-node Flash only. No other containers are stopped or recreated.
set -euo pipefail
ROOT=${NQ_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
DATA=${NQ_FLASH_DATA:-$HOME/nq-flash}
IMAGE=${NQ_FLASH_IMAGE:?Set NQ_FLASH_IMAGE to the published immutable ARM64 digest}
NAME=${NQ_FLASH_CONTAINER:-nq-spark-flash}
mkdir -p "$DATA"
DATA=$(cd "$DATA" && pwd)
common=(--security-opt seccomp=unconfined --cap-add IPC_LOCK --ipc host -v "$DATA:/data")
case ${1:-help} in
 build) docker build -f "$ROOT/spark/flash/Dockerfile" -t "$IMAGE" "$ROOT" ;;
 fetch)
  docker run "${common[@]}" -e HF_TOKEN --entrypoint python "$IMAGE" -c 'from huggingface_hub import snapshot_download; snapshot_download("jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit", revision="4023870a6f2c6097e37fa5d2e54040db64b011e9", local_dir="/data/checkpoint")'
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
  docker run -d --name "$NAME" "${common[@]}" --gpus all -p "127.0.0.1:${NQ_PORT:-8001}:8000" \
   -e PYTHONPATH=/diag:/nq/spark/flash -v /proc/spl/kstat/zfs/arcstats:/diag-host-arcstats:ro -e NQ_PUBLICATION_OBSERVER=/data/readfirst-runtime/publication.jsonl -e VLLM_API_KEY -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=2 -e OMP_NUM_THREADS=4 \
   -e NQ_FLASH_MODEL=/data/fp8-backbone -e NQ_REPACK=/data/repack-tp1 -e NQ_BUILD=/data/build \
   -e NQ_FLASH_PRESET="${NQ_FLASH_PRESET:-spark_256K_U_2630_flat50}" -e NQ_UNIFIED=1 -e NQ_FLASH_SPARE_SLOTS=8 \
   -e NQ_FLASH_PREFETCH=throughput -e NQ_FLASH_PREFETCH_MAX_PENDING=64 \
   -e NQ_FLASH_PREFETCH_MAX_DEMOTIONS=32 -e NQ_FLASH_PREFETCH_SECONDS=0.1 \
   -e NQ_FLASH_WAIT_FOR_UPGRADES=0 \
   -e NQ_FLASH_FIXED_FRACTION="${NQ_FLASH_FIXED_FRACTION:-0.2}" \
   -e NQ_FLASH_JT_PARAMS='{"mix":0.75,"hm":4.0,"G":1,"half_life":64.0}' \
   -e NQ_FLASH_MLA_CACHE="${NQ_FLASH_MLA_CACHE:-fp4_g16}" \
   -e NQ_FLASH_ROUTING_STATS=/data/readfirst-runtime/routing.json -e NQ_FLASH_TPS="${NQ_FLASH_TPS:-0}" \
   -e NQ_DEFS=NQ_RK_CODES=0x405,NQ_RK_GU=0x5,NQ_RK_DN=0x401,NQ_BK_CODES=0x21,NQ_SWIGLU_LIMIT=10 \
   -v "$DATA/cache:/root/.cache" "$IMAGE" \
   serve /data/fp8-backbone --served-model-name glm-5.3-flash-nq \
   --tensor-parallel-size 1 --max-num-seqs 1 --max-model-len "${NQ_MAXLEN:-262144}" \
   --gpu-memory-utilization 0.50 --kv-cache-memory-bytes "${NQ_KV_BYTES:-2147483648}" \
   --max-num-batched-tokens 1024 --enforce-eager --no-enable-prefix-caching --kv-cache-dtype fp8 \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$ns,\"draft_sample_method\":\"probabilistic\"}" \
   --chat-template /nq/spark/flash/thinking_off.jinja \
   --override-generation-config '{"temperature":0.7,"top_p":0.95}' \
   --trust-remote-code --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
   --limit-mm-per-prompt '{"image":1,"video":0}'
  NQ_FLASH_CONTAINER="$NAME" NQ_GUARD_LOG_DIR="${NQ_GUARD_LOG_DIR:-$DATA/readfirst-guard}" nohup python3 "$(dirname "${BASH_SOURCE[0]}")/guard.py" > "$DATA/readfirst-guard-supervisor.log" 2>&1 &
  guard_pid=$!
  sleep 1
  if ! kill -0 "$guard_pid" 2>/dev/null; then
    docker stop "$NAME"
    echo 'Memory guard did not remain active; serving stopped.' >&2
    exit 1
  fi
  ;;
 logs) docker logs --tail 100 -f "$NAME" ;;
 *) echo 'Usage: spark/flash/run_spark.sh build|fetch|prepare|up|logs'; exit 2 ;;
esac
