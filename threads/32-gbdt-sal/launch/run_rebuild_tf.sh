#!/bin/bash
# T32 rebuild: SM120 (b) teacher-forced FP8 recapture (PRIVATE) under the shared GPU lock, then sm120.py preptf.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; C=/home/coder/git/nestquant/threads/32-gbdt-sal
export LD_LIBRARY_PATH=/tmp/compat13/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
until grep -q "all captures done" $O/logs/rebuild_cap.log; do sleep 30; done
flock /tmp/nestquant/33-search/gpu.lock $C/launch/run_tfcap.sh
cd $C; PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=1 NPROC=16 nice -n 10 /home/coder/git/glm52/.venv/bin/python -W ignore sm120.py preptf
echo "TF done $(date -u)"
