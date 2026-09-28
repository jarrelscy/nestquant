#!/bin/bash
# Thread 19 wrapper: CUDA compat env + venv + caps; then python <script> "$@".
source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$HERE:/home/coder/git/nestquant/threads/05-exl3-harness:/home/coder/git/orbit-duet:/home/coder/git/nestquant/threads/18-e2e-eval:$PYTHONPATH
S=$1; shift
exec /home/coder/git/glm52/.venv/bin/python "$HERE/$S" "$@"
