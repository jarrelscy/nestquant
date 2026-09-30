#!/bin/bash
# GPU capture through the shared flock (8 ranks, contig shard = T32 trace windows).  PRIVATE outputs.
set -uo pipefail
H=/home/coder/git/nestquant/threads/33-search/draft
LG=/tmp/nestquant/33-search/draft/logs; mkdir -p $LG
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=/tmp/compat13/usr/local/cuda-13.2/compat:/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NQ_FP8=/tmp/nestquant/src/glm53-fp8 NQ_OUT=/tmp/nestquant/33-search/draft/private/e2e
export NQ_CORPUS_DIR=/tmp/nestquant/18-e2e/corpora NQ_SHARD=contig NQ_VRAM_GB=${NQ_VRAM_GB:-60}
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; unset NQ_TRACE_DIR
WORLD=${WORLD:-8}; RANKS=${RANKS:-$(seq 0 $((WORLD - 1)))}
for r in $RANKS; do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=$WORLD nice -n 5 /home/coder/git/glm52/.venv/bin/python $H/capture_draft.py \
    > $LG/cap${TAG:-}.r$r.log 2>&1 &
done
wait
echo done $(date -u)
