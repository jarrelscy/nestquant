#!/bin/bash
# NestQuant GLM-5.3 serve: OpenAI-compatible API on :8001, served as "glm-5.3-nq" (alias "local").
#
#   ./start.sh [up|down|logs|smoke] [2-4|1.75]   (up and 2-4 are the defaults)
#
# Builds: 2-4 = jarrelscy/GLM-5.3-NestQuant-2-4bit (production). 1.75 = jarrelscy/GLM-5.3-NestQuant-1.75-4bit
# (1.75-bit base, nq-res-v2): its own records + kernel build (NQ_DEFS), 98 slots/layer at 1M, one drive, and the
# same base checkpoint and predictor as 2-4 (downloaded from the 2-4 repo). Same as NQ_VARIANT=1.75.
#
# Boots the production config. All serving defaults live in sm120/serve/docker-compose.standalone.yaml;
# any of them can be overridden from the environment (e.g. NQ_MAXLEN=400000 NQ_SLOTS_PER_LAYER=124 ./start.sh).
# Hardware: 4x RTX PRO 6000 Blackwell (SM120, 96 GB each), TP4 + DCP4, MTP ns=3.
#
# Host paths (defaults = the reference box):
#   NQ_IMAGE          serving image, public on Docker Hub, pulled if absent
#   NQ_MODELS_ROOT    host dir mounted at /data/models (/data/models); NQ_MODEL_DIR = base checkpoint path inside it
#                     (container path, default /data/models/jarrelscy/GLM-5.3-NQ-base). The base = everything except the
#                     routed experts (attention, shared experts, dense layers, MTP layer, vision, tokenizer, ~44 GB); it is
#                     downloaded from NQ_BASE_REPO (base/) if absent
#   NQ_REPACK_DIR     NestQuant records (rank*.json/bin + res/), downloaded from NQ_REPACK_REPO if absent (~393 GB)
#   NQ_REPACK_ALT_DIR copy of rank*.bin, rank*.json, artifact_stamp.json on a second NVMe; without it (or =none) reads use one drive
#   NQ_PREDICTOR_DIR  jF predictor + delta table, downloaded from NQ_PREDICTOR_REPO (serving/predictor/) if absent
#   NQ_LIBURING_DIR   liburing 2.5 install (include/, lib/), built in the image if absent
#   NQ_LGB_DIR        lightgbm + narwhals + scipy for the predictor (numpy comes from the image), installed if absent
#   NQ_BUILD_DIR NQ_VLLM_CACHE NQ_LMCACHE_DIR NQ_TRITON_CACHE_DIR NQ_FLASHINFER_CACHE NQ_TORCHEXT_CACHE NQ_DBG_DIR NQ_PROFILES_DIR
#                     writable build/cache dirs, created if absent
# API key: VLLM_API_KEY from the environment, else from $NQ_ENV_FILE (default ./.env, gitignored), else no auth.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export COMPOSE_FILE="$HERE/sm120/serve/docker-compose.standalone.yaml" COMPOSE_PROJECT_NAME=nestquant
export NQ_REPO=${NQ_REPO:-$HERE}
NQ_VARIANT=${2:-${NQ_VARIANT:-2-4}}
case "$NQ_VARIANT" in
2-4) ;;
1.75)   # 1.75-bit base + 4-bit residual (base K code 1, down residual code 9): records ~400 GB, no second-drive copy by default
  export NQ_REPACK_REPO=${NQ_REPACK_REPO:-jarrelscy/GLM-5.3-NestQuant-1.75-4bit}
  export NQ_BASE_REPO=${NQ_BASE_REPO:-jarrelscy/GLM-5.3-NestQuant-1.75-4bit}       # its own base/ (same bytes as 2-4's base/)
  export NQ_PREDICTOR_REPO=${NQ_PREDICTOR_REPO:-jarrelscy/GLM-5.3-NestQuant-1.75-4bit}  # its own serving/predictor/
  export NQ_REPACK_DIR=${NQ_REPACK_DIR:-/home/jarrelscy/nq-175/hf}
  export NQ_REPACK_ALT_DIR=${NQ_REPACK_ALT_DIR:-none}
  # the smaller resident base frees ~3.6 GiB/GPU at 80 slots: 98 slots/layer (95 floating) keeps the 1M KV pool of 2-4
  export NQ_SLOTS_PER_LAYER=${NQ_SLOTS_PER_LAYER:-98}
  export NQ_VLLM_CACHE=${NQ_VLLM_CACHE:-/data/Jarrel/nq-serve/vllm-cache-step2-b175} ;;
*) echo "unknown build '$NQ_VARIANT' (2-4 or 1.75)"; exit 2 ;;
esac
export NQ_IMAGE=${NQ_IMAGE:-jarrelscy/glm53-nestquant-sm120:fixes12-mtp-buffer-rng-20260917}
export NQ_REPACK_DIR=${NQ_REPACK_DIR:-/home/jarrelscy/nq-p4rec/hf}
export NQ_REPACK_ALT_DIR=${NQ_REPACK_ALT_DIR:-/data/Jarrel/nq-p4rec-nvme1}
export NQ_PREDICTOR_DIR=${NQ_PREDICTOR_DIR:-/data/Jarrel/nq-serve/predictor}
export NQ_LIBURING_DIR=${NQ_LIBURING_DIR:-/data/Jarrel/liburing}
export NQ_LGB_DIR=${NQ_LGB_DIR:-/data/Jarrel/nq-dev/pylgb}
export NQ_BUILD_DIR=${NQ_BUILD_DIR:-/data/Jarrel/nq-build-container}
export NQ_MODELS_ROOT=${NQ_MODELS_ROOT:-/data/models}
export NQ_MODEL_DIR=${NQ_MODEL_DIR:-/data/models/jarrelscy/GLM-5.3-NQ-base}
NQ_REPACK_REPO=${NQ_REPACK_REPO:-${NQ_HF_REPO:-jarrelscy/GLM-5.3-NestQuant-2-4bit}}   # NestQuant records
NQ_BASE_REPO=${NQ_BASE_REPO:-jarrelscy/GLM-5.3-NestQuant-2-4bit}                      # base checkpoint (base/), 2-4 default; the 1.75 preset uses its own repo
NQ_PREDICTOR_REPO=${NQ_PREDICTOR_REPO:-jarrelscy/GLM-5.3-NestQuant-2-4bit}            # jF predictor (serving/predictor/), 2-4 default; 1.75 uses its own

NQ_ENV_FILE=${NQ_ENV_FILE:-$HERE/.env}
if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$NQ_ENV_FILE" ]; then VLLM_API_KEY=$(grep -oP 'VLLM_API_KEY=\K\S+' "$NQ_ENV_FILE" || true); fi
export VLLM_API_KEY=${VLLM_API_KEY:-}
auth=(); [ -n "$VLLM_API_KEY" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")

# run a one-off shell in the serving image (CPU only), files it writes are handed back to the caller
inimg(){ local v=$1; shift; docker run --rm --entrypoint bash -v "$v" "$NQ_IMAGE" -c "$* && chown -R $(id -u):$(id -g) ${v#*:}"; }
hfget(){ local repo=$1; shift; command -v hf >/dev/null || { echo "need the 'hf' CLI: pip install -U 'huggingface_hub[hf_transfer]'"; exit 1; }
         HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$repo" --repo-type model "$@"; }
# every shard the checkpoint index names is present
shards_ok(){ [ -f "$1/config.json" ] && [ -f "$1/model.safetensors.index.json" ] && python3 -c "import json,os,sys
d=sys.argv[1];sys.exit(any(not os.path.exists(os.path.join(d,f)) for f in set(json.load(open(d+'/model.safetensors.index.json'))['weight_map'].values())))" "$1"; }

case "${1:-up}" in
up)
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
    echo "downloading the NestQuant records from $NQ_REPACK_REPO to $RP (~393 GB, once) ..."; mkdir -p "$RP"
    hfget "$NQ_REPACK_REPO" --include 'rank*.json' --include 'rank*.bin' --include 'res/*' --include 'artifact_stamp.json' --local-dir "$RP"
  fi
  echo "NQ layers in repack: $(python3 -c "import json;print(sorted(int(k) for k in json.load(open('$RP/rank0.json'))['layers']))")"
  # nq-res-v2 records (1.75-bit base) need a kernel built with base code 1 + down residual code 9
  if [ "$(python3 -c "import json;print(json.load(open('$RP/rank0.json'))['rec_bytes'])")" = 2854912 ] && [ -z "${NQ_DEFS:-}" ]; then
    export NQ_DEFS=NQ_RK_CODES=0x209,NQ_RK_GU=0x9,NQ_RK_DN=0x201,NQ_BK_CODES=0x3
    echo "1.75-4 bit records: kernel built with $NQ_DEFS"
  fi

  # dual-drive reads need an identical copy of the record files on a second drive
  [ "$NQ_REPACK_ALT_DIR" = none ] && export NQ_IO_MODE='' NQ_REPACK_ALT_DIR=$RP
  if [ "${NQ_IO_MODE-dual}" = dual ]; then
    ok=1; for f in rank0.json rank1.json rank2.json rank3.json artifact_stamp.json; do cmp -s "$RP/$f" "$NQ_REPACK_ALT_DIR/$f" || ok=0; done
    if [ $ok = 0 ]; then
      echo "no record copy at NQ_REPACK_ALT_DIR=$NQ_REPACK_ALT_DIR: reading from one drive (copy rank*.bin, rank*.json, artifact_stamp.json there for dual)"
      export NQ_IO_MODE='' NQ_REPACK_ALT_DIR=$RP
    fi
  fi

  # jF predictor (serving/predictor/ on HF -> NQ_PREDICTOR_DIR)
  if [ ! -f "$NQ_PREDICTOR_DIR/joint/jF.pt" ]; then
    echo "downloading the predictor to $NQ_PREDICTOR_DIR ..."; mkdir -p "$NQ_PREDICTOR_DIR"
    hfget "$NQ_PREDICTOR_REPO" --include 'serving/predictor/*' --local-dir "$NQ_PREDICTOR_DIR/.dl"
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
  docker run --rm --gpus '"device=0"' --entrypoint bash -e NQ_BUILD=/nqbuild -e LIBURING=/data/Jarrel/liburing -e NQ_DEFS="${NQ_DEFS:-}" \
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
down)  docker compose down ;;
logs)  docker logs -f glm53-nestquant ;;
smoke)
  curl -s "${auth[@]}" -H 'Content-Type: application/json' localhost:8001/v1/completions \
    -d '{"model":"local","prompt":"The capital of France is","max_tokens":24,"temperature":0}' \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
  docker logs glm53-nestquant 2>&1 | grep NestQuant | tail -8 ;;
*) echo "usage: $0 [up|down|logs|smoke] [2-4|1.75]"; exit 2 ;;
esac
