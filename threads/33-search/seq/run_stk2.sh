#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/seq
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib NQ_LAYOUT=k0
while kill -0 1998966 2>/dev/null; do sleep 20; done
P="nice -n 10 /home/coder/git/glm52/.venv/bin/python binhist.py"
$P train dyn_s 0.15
$P eval stk_s,dyn_s sm120tf
$P eval stk_s,dyn_s glm52-heldout
