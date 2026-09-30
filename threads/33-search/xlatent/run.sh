#!/bin/bash
# run.sh NAME ARGS... : one GPU training job under the shared lock
cd /home/coder/git/nestquant/threads/33-search/xlatent
export PYTHONPATH=/tmp/nestquant/33-search/xlatent/pylib OMP_NUM_THREADS=8
mkdir -p /tmp/nestquant/33-search/xlatent/logs
flock /tmp/nestquant/33-search/gpu.lock timeout 85m nice -n 10 /home/coder/.venv/bin/python train.py "$@" > /tmp/nestquant/33-search/xlatent/logs/$1.log 2>&1
