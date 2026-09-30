#!/bin/bash
set -e
cd /home/coder/git/nestquant/threads/33-search/hprobe
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$PY exp2.py train g_dn_split lg16,lg64,lgd,dn-4_64,dn-4_d,dn+4_64,dn+4_d --split
$PY exp2.py eval g_dn_split calib-fit 0.3,0.4,0.5 --val
