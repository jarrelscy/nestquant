#!/bin/bash
# One-command NestQuant serve on the SM120 box (4x RTX PRO 6000, TP4+DCP4, MTP ns=3), OpenAI API on :8001.
#   sm120/serve/serve_nq.sh [up|down|logs|smoke]
# Env: NQ_REPACK_DIR (records + resident planes from streaming/repack.py, default /home/jarrelscy/nq-p4rec/prod),
#      NQ_LAYERS (e.g. 3-18; default every layer in the repack), NQ_STREAM (0 = fixed set only), NQ_MAXLEN, NQ_UTIL,
#      NQ_SLOTS_PER_LAYER, NQ_CAP_GBPS (upgrade budget, aggregate GB/s over the 4 ranks, default 0 = uncapped), NQ_PREDICTOR (ema | gbdt, default streaming/scheduler.py DEFAULT_PREDICTOR), NQ_SERVED_NAME (default glm-5.3-nq; alias "local" always works).
# Layers missing from the repack serve with the production ARVQ experts.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd);HA=${HA:-/home/jarrelscy/homeassistant}
export COMPOSE_FILE="$HA/docker-compose.yaml:$HA/docker-compose.glm53-arvq.yaml:$HERE/docker-compose.nq.yaml"
export COMPOSE_PROJECT_NAME=nestquant
IMG=glm53-arvq-sm120:fixes12-mtp-buffer-rng-20260917
key(){ grep -oP 'VLLM_API_KEY=\K\S+' "$HA/.env"; }
case "${1:-up}" in
up)
  RP=${NQ_REPACK_DIR:-/home/jarrelscy/nq-p4rec/prod}
  for r in 0 1 2 3; do [ -f "$RP/rank$r.json" ] || { echo "no repack at $RP (rank$r.json)"; exit 1; }; done
  echo "NQ layers in repack: $(python3 -c "import json;print(sorted(int(k) for k in json.load(open('$RP/rank0.json'))['layers']))")"
  if docker ps --format '{{.Ports}}' | grep -q ':8001->'; then echo "port 8001 is in use; stop the running model first (switch.sh)"; exit 1; fi
  mkdir -p /data/Jarrel/nq-build-container /data/Jarrel/nq-serve/vllm-cache
  # GBDT floating-set predictor deps (the image has none of them); appended to sys.path, so nothing in the image is shadowed
  LGB=${NQ_LGB_DIR:-/data/Jarrel/nq-dev/pylgb}
  [ -d "$LGB/lightgbm" ] || uv pip install -q --python-version 3.12 --target "$LGB" lightgbm==4.7.0 narwhals scipy
  # build the NestQuant kernels once for the image's torch (the 4 workers would otherwise race on the build)
  docker run --rm --gpus '"device=0"' --entrypoint bash -e NQ_BUILD=/nqbuild -e LIBURING=/data/Jarrel/liburing \
    -e CUDA_HOME=/opt/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13 -v "${NQ_REPO:-/data/Jarrel/nestquant}":/nq:ro \
    -v /data/Jarrel/nq-build-container:/nqbuild -v /data/Jarrel/liburing:/data/Jarrel/liburing:ro $IMG \
    -c 'cd /nq/sm120 && /opt/vllm/.venv/bin/python -c "import build;build.get()" && cd ../streaming && /opt/vllm/.venv/bin/python -c "import stream_engine as S;S.mod()"'
  docker compose --profile glm5.3-hybrid-1m up -d
  echo "waiting for /v1/models (loading takes a while) ..."
  for i in $(seq 1 360); do
    curl -sf -H "Authorization: Bearer $(key)" localhost:8001/v1/models >/dev/null 2>&1 && { echo ready; exit 0; }
    docker ps --format '{{.Names}}' | grep -q '^glm53-nestquant$' || { echo "container exited"; docker logs --tail 80 glm53-nestquant; exit 1; }
    sleep 10
  done; echo "timeout"; exit 1 ;;
down) docker compose --profile glm5.3-hybrid-1m down ;;
logs) docker logs -f glm53-nestquant ;;
smoke)
  curl -s -H "Authorization: Bearer $(key)" -H 'Content-Type: application/json' localhost:8001/v1/completions \
    -d '{"model":"local","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
  docker logs glm53-nestquant 2>&1 | grep NestQuant | tail -8 ;;
esac
