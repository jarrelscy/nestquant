#!/bin/bash
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/mimo-a100/cuda12-runtime/nvidia/cuda_runtime/lib:/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=/home/coder/git/orbit-duet
exec /home/coder/git/glm52/.venv/bin/python "$@"
