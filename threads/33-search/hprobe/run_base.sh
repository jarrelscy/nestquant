#!/bin/bash
set -e
cd /home/coder/git/nestquant/threads/33-search/hprobe
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$PY exp2.py train v2re ""
$PY exp2.py eval v2re glm52-heldout 0.3,0.4,0.5
$PY exp2.py train v2re_split "" --split
$PY exp2.py eval v2re_split calib-fit 0.3,0.4,0.5 --val
