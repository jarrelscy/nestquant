#!/bin/bash
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=/home/coder/git/nestquant/threads/05-exl3-harness:$PYTHONPATH
exec /home/coder/git/glm52/.venv/bin/python "$@"
