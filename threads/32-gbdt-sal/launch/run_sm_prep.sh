#!/bin/bash
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
for s in "$@"; do NPROC=${NPROC:-16} nice -n 10 $PY sm120.py prep $s; done
