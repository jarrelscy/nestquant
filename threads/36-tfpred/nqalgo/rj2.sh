#!/bin/bash
# rj2.sh <joblist> <parallel>: lines "R CFG TASK" (nq-algo run.py, results/runs) or "X CFG TASK" (ext.py, nqalgo/runs); skips existing
J=$(readlink -f "$1"); cd /data/Jarrel/nq-tfpred
cat "$J" | xargs -P "${2:-4}" -L 1 bash -c 'export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; if [ "$0" = R ]; then nice -n 5 /data/Jarrel/nq-algo/venv/bin/python /data/Jarrel/nq-algo/run.py "$1" "$2"; else nice -n 5 /data/Jarrel/nq-algo/venv/bin/python /data/Jarrel/nq-tfpred/nqalgo/ext.py "$1" "$2"; fi 2>&1 | tail -1'
