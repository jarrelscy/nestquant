#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/process
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=20
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
for a in "v2r v2" "v2hk v2,hk" "v2hmm v2,hmm" "v2bo v2,bo" "v2kf v2,kf" "v2all v2,hk,hmm,bo,kf"; do
  set -- $a; [ -f /tmp/nestquant/33-search/process/models/$1.txt ] || $PY train.py $1 $2 60
done
