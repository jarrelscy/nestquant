#!/bin/bash
# T33c hprobe GPU capture (run under: flock /tmp/nestquant/33-search/gpu.lock ./capture.sh).  FP8 reference forward
# (private copy of T18 nq_e2e.py with NQ_HP_* hooks), no candidates.  PRIVATE outputs under hprobe/private.
#   MODE=pool CORPORA=calib-fit,glm52-heldout ./capture.sh        -> private/pool   (16-token block means)
#   MODE=proj PROJ=DIR NQ_CORPUS_DIR=.. CORPORA=sm120tf ./capture.sh -> private/proj_CORPUS (per-token projections)
set -euo pipefail
HP=/tmp/nestquant/33-search/hprobe; P=$HP/private; C=/home/coder/git/nestquant/threads/33-search/hprobe
MODE=${MODE:-pool}; CORPORA=${CORPORA:-calib-fit,glm52-heldout}; TAG=${TAG:-hp_$MODE}
read -r -a GMAP <<< "${GPUS:-0 1 2 3 4 5 6 7}"
WORLD=${#GMAP[@]}
mkdir -p $P/logs $P/e2e
while IFS=, read -r i free; do
  [ "${free// /}" -ge 30000 ] || { echo "GPU $i only ${free}MiB free $(date -u)"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=/tmp/compat13/usr/local/cuda-13.2/compat:/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NQ_FP8=${NQ_FP8:-/tmp/nestquant/src/glm53-fp8}
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
export NQ_OUT=$P/e2e NQ_CORPUS_DIR=${NQ_CORPUS_DIR:-/tmp/nestquant/18-e2e/corpora} NQ_SHARD=contig NQ_VRAM_GB=${NQ_VRAM_GB:-40}
if [ "$MODE" = pool ]; then export NQ_HP_POOL=1 NQ_HP_DIR=${HPDIR:-$P/pool}
else export NQ_HP_PROJ=$PROJ NQ_HP_DIR=${HPDIR:-$P/proj_${CORPORA}}; fi
echo "take GPUs $TAG $(date -u)"
for r in $(seq 0 $((WORLD - 1))); do
  CUDA_VISIBLE_DEVICES=${GMAP[$r]} RANK=$r WORLD=$WORLD nice -n 5 \
    /home/coder/git/glm52/.venv/bin/python $C/nq_e2e_hp.py run --corpora "$CORPORA" --moe-chunk 16384 --tag "$TAG" \
    > "$P/logs/$TAG.r$r.log" 2>&1 &
done
wait
echo "release GPUs $TAG $(date -u)"
