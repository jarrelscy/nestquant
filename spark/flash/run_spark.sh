#!/usr/bin/env bash
# Single-node Flash only. No other containers are stopped or recreated.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${NQ_FLASH_DATA:-$HOME/nq-flash}
IMAGE=${NQ_FLASH_IMAGE:-nestquant-spark:flash}
NAME=${NQ_FLASH_CONTAINER:-nq-spark-flash}
mkdir -p "$DATA"
DATA=$(cd "$DATA" && pwd)
common=(--rm --security-opt seccomp=unconfined --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ipc host -v "$ROOT:/nq:ro" -v "$DATA:/data")
case ${1:-help} in
 build) docker build -f "$ROOT/spark/flash/Dockerfile" -t "$IMAGE" "$ROOT" ;;
 fetch)
  docker run "${common[@]}" -e HF_TOKEN --entrypoint python "$IMAGE" -c 'from huggingface_hub import snapshot_download; snapshot_download("jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit", local_dir="/data/checkpoint")'
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
  [[ $(uname -m) == aarch64 ]] || { echo 'This launcher targets native aarch64 DGX Spark.' >&2; exit 1; }
  [[ -f $DATA/fp8-backbone/config.json && -f $DATA/repack-tp1/rank0.json ]] || { echo 'Run fetch and prepare first.' >&2; exit 1; }
  ns=${NUM_SPEC:-2}; [[ $ns == 1 || $ns == 2 ]] || { echo 'NUM_SPEC must be 1 or 2.' >&2; exit 1; }
  # Default KV ceiling is explicit because host-mapped slots consume unified RAM.
  # vLLM must still validate max_model_len; it must not silently shrink the preset.
  docker run -d --name "$NAME" "${common[@]}" --gpus all -p "${NQ_PORT:-8001}:8000" \
   -e VLLM_API_KEY -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=4 -e OMP_NUM_THREADS=4 \
   -e NQ_FLASH_MODEL=/data/fp8-backbone -e NQ_REPACK=/data/repack-tp1 -e NQ_BUILD=/data/build \
   -e NQ_FLASH_PRESET="${NQ_FLASH_PRESET:-spark_256K_U_2630}" -e NQ_UNIFIED=1 -e NQ_FLASH_SPARE_SLOTS=8 \
   -e NQ_FLASH_PREFETCH=throughput -e NQ_FLASH_PREFETCH_MAX_PENDING=64 \
   -e NQ_FLASH_PREFETCH_MAX_DEMOTIONS=32 -e NQ_FLASH_PREFETCH_SECONDS=0.1 \
   -e NQ_FLASH_WAIT_FOR_UPGRADES=0 \
   -e NQ_FLASH_MLA_CACHE="${NQ_FLASH_MLA_CACHE:-fp4_g16}" \
   -e NQ_FLASH_ROUTING_STATS=/data/routing.json -e NQ_FLASH_TPS="${NQ_FLASH_TPS:-0}" \
   -e NQ_DEFS=NQ_RK_CODES=0x405,NQ_RK_GU=0x5,NQ_RK_DN=0x401,NQ_BK_CODES=0x21,NQ_SWIGLU_LIMIT=10 \
   -v "$DATA/cache:/root/.cache" "$IMAGE" \
   --model /data/fp8-backbone --served-model-name glm-5.3-flash-nq \
   --tensor-parallel-size 1 --max-num-seqs 1 --max-model-len "${NQ_MAXLEN:-262144}" \
   --gpu-memory-utilization 0.90 --kv-cache-memory-bytes "${NQ_KV_BYTES:-2147483648}" \
   --max-num-batched-tokens 1024 --enforce-eager --no-enable-prefix-caching --kv-cache-dtype fp8 \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$ns,\"draft_sample_method\":\"probabilistic\"}" \
   --chat-template /nq/spark/flash/thinking_off.jinja \
   --override-generation-config '{"temperature":0.7,"top_p":0.95}' \
   --trust-remote-code --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
   --limit-mm-per-prompt '{"image":1,"video":0}'
  ;;
 logs) docker logs --tail 100 -f "$NAME" ;;
 *) echo 'Usage: spark/flash/run_spark.sh build|fetch|prepare|up|logs'; exit 2 ;;
esac
