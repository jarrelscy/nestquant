#!/bin/bash
# Wrapper: CUDA compat env + venv + thread caps; then nq_e2e.py "$@".
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
# libcudart.so.12 for exllamav3's Hadamard (nq_decode.dense_from_rotated)
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=8
export NQ_FP8=${NQ_FP8:-/tmp/nestquant/src/glm53-fp8} NQ_OUT=${NQ_OUT:-/tmp/nestquant/18-e2e}
HERE=$(cd "$(dirname "$0")" && pwd)
exec /home/coder/git/glm52/.venv/bin/python "$HERE/nq_e2e.py" "$@"
