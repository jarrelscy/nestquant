#!/bin/bash
set -e
cd /home/coder/git/nestquant/threads/33-search/hprobe
while pgrep -f "[r]un_lg.sh" > /dev/null; do sleep 10; done
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$PY make_feats.py hpr 8
TGT=log $PY make_feats.py lgpr 8
TGT=log $PY make_feats.py hpr 8
$PY exp2.py train g_h512 pr_sth512,pr_stlgh512
$PY exp2.py eval g_h512 glm52-heldout 0.3,0.4,0.5
$PY exp2.py train g_h512log pr_st_log,pr_sth512_log,pr_stlgh512_log
$PY exp2.py eval g_h512log glm52-heldout 0.3,0.4,0.5
