#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/seq
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib NQ_LAYOUT=k0
P="nice -n 10 /home/coder/git/glm52/.venv/bin/python binhist.py"
for n in bh0_s v2only_s; do $P train $n 0.15 && $P eval $n glm52-heldout; done
$P eval bh0_s sm120tf; $P eval v2only_s sm120tf
