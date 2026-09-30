#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/seq
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
P="nice -n 10 /home/coder/git/glm52/.venv/bin/python binhist.py"
for n in bh1 v2only1; do
  $P train $n 0.15 && $P eval $n glm52-heldout && $P eval $n calib-fit
done
