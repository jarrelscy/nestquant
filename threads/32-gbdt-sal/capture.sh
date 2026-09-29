#!/bin/bash
# T32 capture: FP8 reference forward (T18 nq_e2e.py ref path, no candidates) over the TRAIN corpora, dumping per sparse
# layer the ref routing (top-8 ids, gate weights incl. routed_scaling_factor, |x|^2 of the normalised MoE input).
# PRIVATE: traces stay under /tmp/nestquant/32-gbdt-sal/trace (never uploaded).
#   CORPORA=calib-fit,glm52-heldout GPUS="0 1 2 3 4 5 6 7" ./capture.sh
set -euo pipefail
OUT32=/tmp/nestquant/32-gbdt-sal
CORPORA=${CORPORA:-calib-fit,glm52-heldout}
TAG=${TAG:-t32cap}
read -r -a GMAP <<< "${GPUS:-0 1 2 3 4 5 6 7}"
WORLD=${#GMAP[@]}
# NQ_CORPUS_DIR: token-id corpora dir (default T18's); for arbitrary ids: prep_ids.py NAME SRC, then
#   NQ_CORPUS_DIR=$OUT32/corpora CORPORA=NAME NQ_TRACE_PROBS=1 NQ_TRACE_DIR=$OUT32/trace_NAME ./capture.sh
# NQ_OUT (results/refcache of the harness) is kept private under $OUT32/e2e.
export NQ_OUT=${NQ_OUT:-$OUT32/e2e} NQ_CORPUS_DIR=${NQ_CORPUS_DIR:-/tmp/nestquant/18-e2e/corpora}
export NQ_TRACE_DIR=${NQ_TRACE_DIR:-$OUT32/trace} NQ_SHARD=contig NQ_VRAM_GB=${NQ_VRAM_GB:-12} OMP_NUM_THREADS=2
for r in $(seq 0 $((WORLD - 1))); do
  CUDA_VISIBLE_DEVICES=${GMAP[$r]} RANK=$r WORLD=$WORLD nohup nice -n 5 \
    /home/coder/git/nestquant/threads/18-e2e-eval/run.sh run --corpora "$CORPORA" --moe-chunk 16384 --tag "$TAG" \
    > "$OUT32/logs/$TAG.r$r.log" 2>&1 &
done
wait
echo done
