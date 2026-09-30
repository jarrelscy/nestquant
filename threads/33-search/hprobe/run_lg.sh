#!/bin/bash
set -e
cd /home/coder/git/nestquant/threads/33-search/hprobe
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib LGSRC=lg2
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$PY make_feats.py lgpr 8
$PY exp2.py train g_lgraw lg16,lg64,lgd
$PY exp2.py eval g_lgraw glm52-heldout 0.3,0.4,0.5
$PY exp2.py train g_prstlg pr_st,pr_stlg
$PY exp2.py eval g_prstlg glm52-heldout 0.3,0.4,0.5
