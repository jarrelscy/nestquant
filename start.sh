#!/bin/bash
# NestQuant GLM-5.3 serve: OpenAI-compatible API on :8001, served as "glm-5.3-nq" (alias "local").
#
#   ./start.sh [up|down|logs|smoke|config]        (up is the default)
#
# Boots the production config. All serving defaults live in sm120/serve/docker-compose.standalone.yaml;
# any of them can be overridden from the environment (e.g. NQ_MAXLEN=400000 NQ_SLOTS_PER_LAYER=124 ./start.sh).
# Hardware: 4x RTX PRO 6000 Blackwell (SM120, 96 GB each), TP4 + DCP4, MTP ns=3.
#
# Host paths (default root = $HOME/.local/share/nestquant; override with NQ_STATE):
#   NQ_IMAGE          serving image, public on Docker Hub, pulled if absent
#   NQ_MODELS_ROOT    host dir mounted at /data/models (/data/models); NQ_MODEL_DIR = base checkpoint path inside it
#                     (container path, default /data/models/jarrelscy/GLM-5.3-NQ-base). The base = everything except the
#                     routed experts (attention, shared experts, dense layers, MTP layer, vision, tokenizer, ~44 GB); it is
#                     downloaded from NQ_BASE_REPO (base/) if absent
#   NQ_REPACK_DIR     NestQuant records (rank*.json/bin + res/), downloaded from NQ_REPACK_REPO if absent (~393 GB)
#   NQ_REPACK_ALT_DIR second-NVMe directory; populated automatically; unset = single drive
#   NQ_PREDICTOR_DIR  jF predictor + delta table, downloaded from NQ_REPACK_REPO (serving/predictor/) if absent
#   NQ_LIBURING_DIR   liburing 2.5 install (include/, lib/), built in the image if absent
#   NQ_LGB_DIR        lightgbm + narwhals + scipy for the predictor (numpy comes from the image), installed if absent
#   NQ_BUILD_DIR NQ_VLLM_CACHE NQ_LMCACHE_DIR NQ_TRITON_CACHE_DIR NQ_FLASHINFER_CACHE NQ_TORCHEXT_CACHE NQ_DBG_DIR NQ_PROFILES_DIR
#                     writable build/cache dirs, created if absent
# API key: VLLM_API_KEY from the environment, else from $NQ_ENV_FILE (default ./.env, gitignored), else no auth.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export COMPOSE_FILE="$HERE/sm120/serve/docker-compose.standalone.yaml" COMPOSE_PROJECT_NAME=nestquant
export NQ_REPO=${NQ_REPO:-$HERE}
# Stateful predictor/streaming setup is validated for one active sequence only.
case "${1:-up}" in up|config)
  if [ "${NQ_MAX_NUM_SEQS:-1}" != 1 ] || [ "${MAX_NUM_SEQS:-1}" != 1 ]; then
    echo "NestQuant serving requires max_num_seqs=1. Unset conflicting NQ_MAX_NUM_SEQS/MAX_NUM_SEQS overrides." >&2
    exit 2
  fi ;;
esac
export NQ_MAX_NUM_SEQS=1
export NQ_IMAGE=${NQ_IMAGE:-jarrelscy/glm53-nestquant-sm120:fixes12-mtp-buffer-rng-20260917}
# Portable host paths. Set NQ_STATE to a large NVMe filesystem before first boot.
export NQ_STATE=${NQ_STATE:-${HOME}/.local/share/nestquant}
export NQ_REPACK_DIR=${NQ_REPACK_DIR:-$NQ_STATE/records}
# An explicitly configured second directory is populated automatically before boot.
# Without one, use single-drive reads; never silently create a second huge copy on the same drive.
export NQ_REPACK_ALT_DIR=${NQ_REPACK_ALT_DIR:-$NQ_REPACK_DIR}
export NQ_PREDICTOR_DIR=${NQ_PREDICTOR_DIR:-$NQ_STATE/predictor}
export NQ_LIBURING_DIR=${NQ_LIBURING_DIR:-$NQ_STATE/liburing}
export NQ_LGB_DIR=${NQ_LGB_DIR:-$NQ_STATE/pylgb}
export NQ_BUILD_DIR=${NQ_BUILD_DIR:-$NQ_STATE/build}
export NQ_MODELS_ROOT=${NQ_MODELS_ROOT:-$NQ_STATE/models}
export NQ_MODEL_DIR=${NQ_MODEL_DIR:-/data/models/jarrelscy/GLM-5.3-NQ-base}
export NQ_VLLM_CACHE=${NQ_VLLM_CACHE:-$NQ_STATE/vllm-cache}
export NQ_LMCACHE_DIR=${NQ_LMCACHE_DIR:-$NQ_STATE/lmcache}
export NQ_TRITON_CACHE_DIR=${NQ_TRITON_CACHE_DIR:-$NQ_STATE/triton}
export NQ_FLASHINFER_CACHE=${NQ_FLASHINFER_CACHE:-$NQ_STATE/flashinfer}
export NQ_TORCHEXT_CACHE=${NQ_TORCHEXT_CACHE:-$NQ_STATE/torch-extensions}
export NQ_DBG_DIR=${NQ_DBG_DIR:-$NQ_STATE/debug}
export NQ_PROFILES_DIR=${NQ_PROFILES_DIR:-$NQ_STATE/profiles}
NQ_REPACK_REPO=${NQ_REPACK_REPO:-${NQ_HF_REPO:-jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit}}   # NestQuant records + predictor
NQ_BASE_REPO=${NQ_BASE_REPO:-jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit}                      # base checkpoint (base/)

NQ_ENV_FILE=${NQ_ENV_FILE:-$HERE/.env}
if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$NQ_ENV_FILE" ]; then VLLM_API_KEY=$(grep -oP 'VLLM_API_KEY=\K\S+' "$NQ_ENV_FILE" || true); fi
export VLLM_API_KEY=${VLLM_API_KEY:-}
auth=(); [ -n "$VLLM_API_KEY" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")

# run a one-off shell in the serving image (CPU only), files it writes are handed back to the caller
inimg(){ local v=$1; shift; docker run --rm --entrypoint bash -v "$v" "$NQ_IMAGE" -c "$* && chown -R $(id -u):$(id -g) ${v#*:}"; }
hfget(){
  local repo=$1; shift
  if command -v hf >/dev/null; then
    hf download "$repo" --repo-type model "$@"
  else
    # Isolated CLI install; no changes to system Python.
    local cli="$NQ_STATE/hf-cli/bin/hf"
    if [ ! -x "$cli" ]; then
      python3 -m venv "$NQ_STATE/hf-cli"
      "$NQ_STATE/hf-cli/bin/pip" install -q huggingface_hub
    fi
    "$cli" download "$repo" --repo-type model "$@"
  fi
}
# Print only an allowlist: compose config also contains VLLM_API_KEY.
show_config(){ docker compose config --format json | python3 "$HERE/sm120/serve/tools/launch_config.py"; }
# every shard the checkpoint index names is present
shards_ok(){ [ -f "$1/config.json" ] && [ -f "$1/model.safetensors.index.json" ] && python3 -c "import json,os,sys
d=sys.argv[1];sys.exit(any(not os.path.exists(os.path.join(d,f)) for f in set(json.load(open(d+'/model.safetensors.index.json'))['weight_map'].values())))" "$1"; }

case "${1:-up}" in
up)
  show_config
  if docker ps --format '{{.Ports}}' | grep -q ':8001->'; then echo "port 8001 is in use; stop the running model first"; exit 1; fi
  docker image inspect "$NQ_IMAGE" >/dev/null 2>&1 || docker pull "$NQ_IMAGE"

  # base checkpoint (base/ on HF -> NQ_MODELS_ROOT/<NQ_MODEL_DIR below /data/models>)
  case "$NQ_MODEL_DIR" in /data/models/*) ;; *) echo "NQ_MODEL_DIR must be a container path under /data/models"; exit 1 ;; esac
  MD=$NQ_MODELS_ROOT/${NQ_MODEL_DIR#/data/models/}
  if ! shards_ok "$MD"; then
    echo "downloading the base checkpoint to $MD (~44 GB, once) ..."; mkdir -p "$MD"
    hfget "$NQ_BASE_REPO" --include 'base/*' --local-dir "$MD/.dl"
    mv -f "$MD/.dl/base/"* "$MD/" && rm -rf "$MD/.dl"
    shards_ok "$MD" || { echo "base checkpoint at $MD is incomplete"; exit 1; }
  fi

  # NestQuant records (no repacking needed)
  RP=$NQ_REPACK_DIR
  if ! ls "$RP"/rank{0,1,2,3}.json >/dev/null 2>&1; then
    echo "downloading the NestQuant records to $RP (~393 GB, once) ..."; mkdir -p "$RP"
    hfget "$NQ_REPACK_REPO" --include 'rank*.json' --include 'rank*.bin' --include 'res/*' --include 'artifact_stamp.json' --local-dir "$RP"
  fi
  echo "NQ layers in repack: $(python3 -c "import json;print(sorted(int(k) for k in json.load(open('$RP/rank0.json'))['layers']))")"

  # Copy records atomically to the configured second NVMe before starting workers.
  if [ "${NQ_IO_MODE-dual}" = dual ]; then
    if [ "$NQ_REPACK_ALT_DIR" -ef "$RP" ] || [ "$NQ_REPACK_ALT_DIR" = "$RP" ]; then
      export NQ_IO_MODE=''
      echo "Single-drive reads: set NQ_REPACK_ALT_DIR on a second NVMe for reference dual-drive throughput."
    else
      python3 "$HERE/sm120/serve/tools/prepare_dual.py" "$RP" "$NQ_REPACK_ALT_DIR"
    fi
  fi
  show_config

  # jF predictor (serving/predictor/ on HF -> NQ_PREDICTOR_DIR)
  if [ ! -f "$NQ_PREDICTOR_DIR/joint/jF.pt" ]; then
    echo "downloading the predictor to $NQ_PREDICTOR_DIR ..."; mkdir -p "$NQ_PREDICTOR_DIR"
    hfget "$NQ_REPACK_REPO" --include 'serving/predictor/*' --local-dir "$NQ_PREDICTOR_DIR/.dl"
    cp -r "$NQ_PREDICTOR_DIR/.dl/serving/predictor/." "$NQ_PREDICTOR_DIR/"
  fi

  # liburing (static lib for the io_uring streamer) and the predictor's Python deps, built in the image
  [ -f "$NQ_LIBURING_DIR/lib/liburing.a" ] || { mkdir -p "$NQ_LIBURING_DIR"; inimg "$NQ_LIBURING_DIR:/out" \
    'git clone -q --depth 1 -b liburing-2.5 https://github.com/axboe/liburing /tmp/lu && cd /tmp/lu && ./configure --prefix=/out >/dev/null && make -s -j8 -C src && make -s -C src install'; }
  [ -d "$NQ_LGB_DIR/lightgbm" ] || { mkdir -p "$NQ_LGB_DIR"; inimg "$NQ_LGB_DIR:/nqlgb" \
    'uv pip install -q --no-deps --python-version 3.12 --target /nqlgb lightgbm==4.7.0 narwhals==2.26.0 scipy==1.18.1'; }

  mkdir -p "$NQ_BUILD_DIR" "${NQ_VLLM_CACHE:-/data/Jarrel/nq-serve/vllm-cache-step2}" "${NQ_DBG_DIR:-/data/Jarrel/nq-serve/dbg}" \
           "${NQ_LMCACHE_DIR:-/data/lmcache/glm5.3-nq}" "${NQ_TRITON_CACHE_DIR:-/data/triton_cache/glm5.3-arvq}" \
           "${NQ_PROFILES_DIR:-/data/profiles/glm5.3-arvq}" "${NQ_FLASHINFER_CACHE:-/data/Jarrel/nq-serve/flashinfer-cache}" \
           "${NQ_TORCHEXT_CACHE:-/data/Jarrel/nq-serve/torch-extensions}"
  # leftover in-boot knob files would override the env settings
  rm -f /dev/shm/nq_la_ctl /dev/shm/nq_pf_off /dev/shm/nq_sr_ctl /dev/shm/nq_pb_off /dev/shm/nq_pb_kv_off /dev/shm/nq_pb_free_cap \
        /dev/shm/nq_tier_drop /dev/shm/nq_dec_block /dev/shm/nq_dec_block_firstn /dev/shm/nq_dec_async /dev/shm/nq_dec_async_switch \
        /dev/shm/nq_hit_carry /dev/shm/nq_pred_inputs /dev/shm/nq_tap_ctl /dev/shm/nq_pf_block

  # build the NestQuant kernels once for the image's torch (the 4 workers would otherwise race on the build)
  docker run --rm --gpus '"device=0"' --entrypoint bash -e NQ_BUILD=/nqbuild -e LIBURING=/data/Jarrel/liburing \
    -e CUDA_HOME=/opt/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13 -v "$NQ_REPO":/nq:ro -v "$NQ_BUILD_DIR":/nqbuild \
    -v "$NQ_LIBURING_DIR":/data/Jarrel/liburing:ro "$NQ_IMAGE" \
    -c 'cd /nq/sm120 && /opt/vllm/.venv/bin/python -c "import build;build.get();build.get_sal()" && cd ../streaming && /opt/vllm/.venv/bin/python -c "import stream_engine as S;S.mod();import hostcore;hostcore.mod()"'

  docker compose up -d
  echo "waiting for /v1/models (first boot compiles and takes longer) ..."
  for i in $(seq 1 360); do
    curl -sf "${auth[@]}" localhost:8001/v1/models >/dev/null 2>&1 && { echo ready; exit 0; }
    docker ps --format '{{.Names}}' | grep -q '^glm53-nestquant$' || { echo "container exited"; docker logs --tail 80 glm53-nestquant; exit 1; }
    sleep 10
  done
  echo "timeout"; exit 1 ;;
config) show_config ;;
down)  docker compose down ;;
logs)  docker logs -f glm53-nestquant ;;
smoke)
  curl -s "${auth[@]}" -H 'Content-Type: application/json' localhost:8001/v1/completions \
    -d '{"model":"local","prompt":"The capital of France is","max_tokens":24,"temperature":0}' \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
  docker logs glm53-nestquant 2>&1 | grep NestQuant | tail -8 ;;
*) echo "usage: $0 [up|down|logs|smoke|config]"; exit 2 ;;
esac
