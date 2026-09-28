#!/bin/bash
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/coder/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /home/coder/git/glm52/.venv/bin/python "$@"
