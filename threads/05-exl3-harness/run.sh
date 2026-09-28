#!/bin/bash
# Wrapper: CUDA compat driver lib + assigned GPU + thread caps, then harness.py "$@"
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
HERE=$(cd "$(dirname "$0")" && pwd)
exec /home/coder/git/glm52/.venv/bin/python "$HERE/harness.py" "$@"
