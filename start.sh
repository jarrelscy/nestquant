#!/bin/bash
# One-command NestQuant serve. OpenAI-compatible API on :8001, served as "glm-5.3-nq".
#
#   ./start.sh [up|down|logs|smoke]
#
# Self-contained: uses only sm120/serve/docker-compose.standalone.yaml from this repo. Requires the
# local artifacts and the SM120 Docker image described in README "Serving (SM120, vLLM)". Every host
# path is an env var with a default matching the reference box; override for a different host, e.g.
#   NQ_REPACK_DIR=/mnt/repack NQ_MODELS_ROOT=/mnt/models ./start.sh
#
# Hardware: 4x RTX PRO 6000 Blackwell (SM120, 96 GB each). TP4 + DCP4, MTP ns=3.
#
# Key env vars (see the compose file for the full set):
#   NQ_IMAGE          serving image tag (public on Docker Hub) (default jarrelscy/glm53-nestquant-sm120:fixes12-...)
#   NQ_MODELS_ROOT    host dir mounted at /data/models     (default /data/models); holds NQ_MODEL_DIR
#   NQ_MODEL_DIR      base checkpoint path (container)     (default .../GLM-5.3-Vision-...-ARVQ-hybrid-...)
#   NQ_REPACK_DIR     NestQuant repack (rankN.json + planes) (default /home/jarrelscy/nq-p4rec/hf)
#   NQ_REPACK_REPO    HF repo to fetch the repack from if absent (default jarrelscy/GLM-5.3-NestQuant-2-4bit)
#   NQ_PREDICTOR_DIR  predictor dir (jF.pt, gbdt, delta)   (default /data/Jarrel/nq-serve/predictor)
#   NQ_LIBURING_DIR   liburing install                     (default /data/Jarrel/liburing)
#   NQ_REPO           this repo                            (default /data/Jarrel/nestquant)
#   NQ_PREDICTOR      ema | gbdt | joint/jF                (default image default)
#   NQ_SERVED_NAME NQ_MAXLEN NQ_UTIL NQ_MAX_NUM_SEQS NUM_SPEC ...  (serving sizing)
#   VLLM_API_KEY      API key; if unset, read from $NQ_ENV_FILE (default ./.env, gitignored), else no auth
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export COMPOSE_FILE="$HERE/sm120/serve/docker-compose.standalone.yaml"
export COMPOSE_PROJECT_NAME=nestquant
export NQ_REPO=${NQ_REPO:-$HERE}
NQ_IMAGE=${NQ_IMAGE:-jarrelscy/glm53-nestquant-sm120:fixes12-mtp-buffer-rng-20260917}
export NQ_IMAGE

# VLLM_API_KEY: honour the environment first, else pull it from a gitignored local env file, else none.
NQ_ENV_FILE=${NQ_ENV_FILE:-$HERE/.env}
if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$NQ_ENV_FILE" ]; then
  VLLM_API_KEY=$(grep -oP 'VLLM_API_KEY=\K\S+' "$NQ_ENV_FILE" || true)
fi
export VLLM_API_KEY=${VLLM_API_KEY:-}
auth=(); [ -n "$VLLM_API_KEY" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")

case "${1:-up}" in
up)
  RP=${NQ_REPACK_DIR:-/home/jarrelscy/nq-p4rec/hf}
  NQ_REPACK_REPO=${NQ_REPACK_REPO:-jarrelscy/GLM-5.3-NestQuant-2-4bit}
  # Fetch the serve-ready NestQuant repack from HF if it isn't here yet (no repacking needed).
  need=0; for r in 0 1 2 3; do [ -f "$RP/rank$r.json" ] || need=1; done
  if [ "$need" = 1 ]; then
    echo "repack not found at $RP — downloading $NQ_REPACK_REPO from HF (~366 GB, once) ..."
    command -v hf >/dev/null 2>&1 || { echo "need the 'hf' CLI: pip install -U 'huggingface_hub[hf_transfer]'"; exit 1; }
    mkdir -p "$RP"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$NQ_REPACK_REPO" --repo-type model \
      --include 'rank*.json' 'rank*.bin' 'res/*' 'artifact_stamp.json' --local-dir "$RP"
  fi
  for r in 0 1 2 3; do
    [ -f "$RP/rank$r.json" ] || { echo "repack still missing rank$r.json at $RP after download. Set NQ_REPACK_DIR / NQ_REPACK_REPO."; exit 1; }
  done
  echo "NQ layers in repack: $(python3 -c "import json;print(sorted(int(k) for k in json.load(open('$RP/rank0.json'))['layers']))")"
  # Pull the SM120 serving image if it isn't already present (the kernel build below runs it before compose).
  if ! docker image inspect "$NQ_IMAGE" >/dev/null 2>&1; then
    echo "serving image $NQ_IMAGE not local — pulling from Docker Hub (~20 GB, once) ..."
    docker pull "$NQ_IMAGE"
  fi
  if docker ps --format '{{.Ports}}' | grep -q ':8001->'; then
    echo "port 8001 is already in use; stop the running model first"; exit 1
  fi
  # writable host dirs the compose mounts
  mkdir -p "${NQ_BUILD_DIR:-/data/Jarrel/nq-build-container}" \
           "${NQ_VLLM_CACHE:-/data/Jarrel/nq-serve/vllm-cache}" \
           "${NQ_DBG_DIR:-/data/Jarrel/nq-serve/dbg}" \
           "${NQ_LMCACHE_DIR:-/data/lmcache/glm5.3-nq}" \
           "${NQ_TRITON_CACHE_DIR:-/data/triton_cache/glm5.3-arvq}" \
           "${NQ_PROFILES_DIR:-/data/profiles/glm5.3-arvq}" \
           "${NQ_FLASHINFER_CACHE:-/data/Jarrel/nq-serve/flashinfer-cache}" \
           "${NQ_TORCHEXT_CACHE:-/data/Jarrel/nq-serve/torch-extensions}"
  # GBDT floating-set predictor deps (the image ships none); appended to sys.path, shadows nothing
  LGB=${NQ_LGB_DIR:-/data/Jarrel/nq-dev/pylgb}
  [ -d "$LGB/lightgbm" ] || uv pip install -q --python-version 3.12 --target "$LGB" lightgbm==4.7.0 narwhals scipy
  # Build the NestQuant kernels once for the image's torch (the 4 workers would otherwise race on it)
  docker run --rm --gpus '"device=0"' --entrypoint bash \
    -e NQ_BUILD=/nqbuild -e LIBURING=/data/Jarrel/liburing \
    -e CUDA_HOME=/opt/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13 \
    -v "$NQ_REPO":/nq:ro \
    -v "${NQ_BUILD_DIR:-/data/Jarrel/nq-build-container}":/nqbuild \
    -v "${NQ_LIBURING_DIR:-/data/Jarrel/liburing}":/data/Jarrel/liburing:ro "$NQ_IMAGE" \
    -c 'cd /nq/sm120 && /opt/vllm/.venv/bin/python -c "import build;build.get();build.get_sal()" && cd ../streaming && /opt/vllm/.venv/bin/python -c "import stream_engine as S;S.mod()"'
  docker compose up -d
  echo "waiting for /v1/models (loading takes a while) ..."
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
    -d '{"model":"glm-5.3-nq","prompt":"The capital of France is","max_tokens":24,"temperature":0}' \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
  docker logs glm53-nestquant 2>&1 | grep NestQuant | tail -8 ;;
*) echo "usage: $0 [up|down|logs|smoke]"; exit 2 ;;
esac
