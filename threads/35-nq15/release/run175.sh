#!/bin/bash
# T35 box-2 copy of threads/18-e2e-eval/run.sh: NO cuda-13.0 compat libcuda (driver 580 -> error 803), own lightgbm pylib.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NQ_FP8=${NQ_FP8:-/tmp/nestquant/src/glm53-fp8} NQ_OUT=${NQ_OUT:-/tmp/nestquant/35-nq15/e2e}
export LAYOUT=k0   # jlib/train import-time manifest only (inference unaffected; 28-serve-release out/ absent on box 2)
export PYTHONPATH=/tmp/nestquant/35-nq15/pylib${PYTHONPATH:+:$PYTHONPATH}
exec /home/coder/git/glm52/.venv/bin/python /home/coder/git/nestquant/threads/18-e2e-eval/nq_e2e.py "$@"
