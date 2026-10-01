#!/bin/bash
# T35 Stage 1 pilot: 9 experts x {b20, b175, b15} + EXL3-1.5/1.75 anchors, one gpu.lock hold (<= 90 min).
export PYTHONPATH=${PYTHONPATH:-}
cd /home/coder/git/nestquant/threads/35-nq15
source /home/coder/git/nestquant/threads/12-reference-encoder/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NQ_GLM_SOURCE=/tmp/nestquant/src/glm53-fp8
LOG=/tmp/nestquant/35-nq15/logs
EXP=(16:36 16:92 16:165 49:36 49:92 49:165 66:36 66:92 66:165)
GPU=(0 1 2 3 4 5 6 7 0)
exec 9>/tmp/nestquant/33-search/gpu.lock
echo "waiting for gpu.lock $(date -u)"; flock 9; echo "take GPUs pilot15 $(date -u)"
pids=()
for i in "${!EXP[@]}"; do
  e=${EXP[$i]}
  CUDA_VISIBLE_DEVICES=${GPU[$i]} timeout 5000 nice -n 10 $PY pilot15.py $e > $LOG/pilot15_${e/:/_}.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
echo "release GPUs pilot15 rc=$rc $(date -u)"
