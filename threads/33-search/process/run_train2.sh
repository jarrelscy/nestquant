#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/process
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=20
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
for a in "v2hks v2,hks" "v2bou v2,bou" "v2proc v2,hks,hmm,bou,kf"; do
  set -- $a; [ -f /tmp/nestquant/33-search/process/models/$1.txt ] || $PY train.py $1 $2 60
done
M=/tmp/nestquant/33-search/process/models
for c in calib-val glm52-heldout; do HMS=0.3,0.4,0.5,0.7 NPROC=8 $PY evalp.py $c v2bou=$M/v2bou.txt v2proc=$M/v2proc.txt bous=sa:bou_srate 2>&1 | grep -v Warn; done
