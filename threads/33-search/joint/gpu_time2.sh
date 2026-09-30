#!/bin/bash
# T33i step 3b: CUDA-graph + fp32-net timing and GPU parity (short hold, GPU 0 only)
cd /home/coder/git/nestquant/threads/33-search/joint
export CUDA_VISIBLE_DEVICES=0 LAYOUT=k0 PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=4
PY=/home/coder/git/prime-radiant/.venv/bin/python
echo "start $(date)"
for cfg in "1 1" "1 0" "0 0"; do set -- $cfg
  GRAPH=$1 BF16=$2 NOREF=1 DEV=cuda timeout 300 $PY parity_gpu.py j_tf_tw_ne 96 0,5 2>&1 | tail -3
done
GRAPH=1 BF16=0 DEV=cuda timeout 300 $PY parity_gpu.py j_tf_tw_ne 12 3 2>&1 | tail -3
GRAPH=1 BF16=1 DEV=cuda timeout 300 $PY parity_gpu.py j_tf_tw_ne 12 3 2>&1 | tail -3
echo "done $(date)"
