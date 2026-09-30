#!/bin/bash
# T33l GPU job (hold the lock via: flock /tmp/nestquant/33-search/gpu.lock ./run_gen.sh [--resume])
set -uo pipefail
export LD_LIBRARY_PATH=/tmp/compat13/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/tmp/nestquant/18-e2e/pylib
PY=/home/coder/git/glm52/.venv/bin/python
H=/home/coder/git/nestquant/threads/33-search/ceiling
O=/tmp/nestquant/33-search/ceiling
echo "lock taken $(date -u)"
if [ "${1:-}" != "--resume" ] && [ ! -f $O/smoke/gen.npz ]; then
  nice -n 10 $PY $H/gen.py --prefixes $O/prefixes.json --out $O/smoke --k 2 --steps 2 --n-layers 5 --max-prefixes 16 \
    --threads 16 --resident-layers 0 > $O/smoke.log 2>&1 || { echo "smoke failed"; tail -30 $O/smoke.log; exit 1; }
  echo "smoke ok $(date -u)"
fi
nice -n 10 $PY $H/gen.py --prefixes $O/${PREF:-prefixes.json} --out $O/${GOUT:-gen} --k 8 --steps 64 --threads 16 --reserve-gb 18 \
  --time-limit-min ${TLIM:-78} "$@" >> $O/${GOUT:-gen}.log 2>&1
rc=$?
echo "gen rc=$rc $(date -u)"
exit $rc
