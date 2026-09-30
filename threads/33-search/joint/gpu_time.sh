#!/bin/bash
# T33i step 3: GPU serve-path timing + GPU parity (one short hold, GPU 0 only)
cd /home/coder/git/nestquant/threads/33-search/joint
export CUDA_VISIBLE_DEVICES=0 LAYOUT=k0 PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=4
PY=/home/coder/git/prime-radiant/.venv/bin/python
echo "start $(date)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i 0
NOREF=1 DEV=cuda timeout 600 $PY parity_gpu.py j_tf_tw_ne 96 0,5 2>&1 | tail -3
DEV=cuda timeout 600 $PY parity_gpu.py j_tf_tw_ne 12 3 2>&1 | tail -3
echo "done $(date)"
