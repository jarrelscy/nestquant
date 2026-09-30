#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/process
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib LAYOUT=k0
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
M=/tmp/nestquant/33-search/process/models
LSTEP=3 TAG=_k0_lstep3 PT=2 HMS=0.5,0.6,0.7 NPROC=9 FAMS=hks,hmm,bou,kf $PY evalp.py sm120tf v2_k0=v2 k0_v2r=$M/k0_v2r.txt k0_v2proc=$M/k0_v2proc.txt k0_v2bou=$M/k0_v2bou.txt 2>&1 | grep -v Warn
