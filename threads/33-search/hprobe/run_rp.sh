#!/bin/bash
set -e
cd /home/coder/git/nestquant/threads/33-search/hprobe
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$PY make_dn.py -4,4
$PY exp2.py train g_dn lg16,lg64,lgd,dn-4_64,dn-4_d,dn+4_64,dn+4_d
$PY exp2.py eval g_dn glm52-heldout 0.3,0.4,0.5
TAGX=all $PY exp3.py 1000 h,lgh 8
$PY exp2.py train g_rp rp_h_l1000,rp_lgh_l1000
$PY exp2.py eval g_rp glm52-heldout 0.3,0.4,0.5
