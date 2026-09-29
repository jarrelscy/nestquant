#!/bin/bash
# sims for the long-horizon batch: $1 = tag, rest = model names in models/ideas (+ old, v2 refs); lag 1 and 0, extras
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
S=/home/coder/git/nestquant/streaming; M=/tmp/nestquant/32-gbdt-sal/models/ideas
tag=$1; shift
ARGS="old=$S/gbdt_p64_s5.txt v2=$S/gbdt_v2sal_p64.txt"; for a in "$@"; do ARGS="$ARGS $a=$M/$a.txt"; done
for lag in ${LAGS:-1 0}; do
  T32_EXTRA=1 T32_BAND=all T32_LAG=$lag T32_TAG=_${tag}_lag$lag NPROC=16 nice -n 10 /home/coder/git/glm52/.venv/bin/python sim.py glm52-heldout $ARGS > /tmp/nestquant/32-gbdt-sal/logs/sim_${tag}_lag$lag.log 2>&1
done
