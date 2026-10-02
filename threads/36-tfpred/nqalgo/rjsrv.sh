#!/bin/bash
# rjsrv.sh <joblist> <P>: lines "CFG TASK" via nq-algo tools/srvtap_run.py (prefill-resync wrapper; results -> nq-algo results/runs)
J=$(readlink -f "$1"); cd /data/Jarrel/nq-tfpred
cat "$J" | xargs -P "${2:-4}" -L 1 bash -c 'export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; [ -f /data/Jarrel/nq-algo/results/runs/$0.$1.json ] || nice -n 5 /data/Jarrel/nq-algo/venv/bin/python /data/Jarrel/nq-algo/tools/srvtap_run.py "$0" "$1" 2>&1 | tail -1'
