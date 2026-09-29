#!/bin/bash
# run_hot.sh CORPUS name... (models/ideas/NAME.txt; v2 = streaming ref)
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
c=$1; shift; A="v2=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt"
for a in "$@"; do A="$A $a=/tmp/nestquant/32-gbdt-sal/models/ideas/$a.txt"; done
NPROC=${NPROC:-16} nice -n 10 /home/coder/git/glm52/.venv/bin/python hot_eval.py $c $A
