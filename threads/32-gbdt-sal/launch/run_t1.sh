#!/bin/bash
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
nice -n 10 $PY build_t1.py glm52-heldout 16 > /tmp/nestquant/32-gbdt-sal/logs/build_t1_ho.log 2>&1
nice -n 10 $PY build_t1.py calib-fit 16 > /tmp/nestquant/32-gbdt-sal/logs/build_t1_cf.log 2>&1
echo done $(date -u) >> /tmp/nestquant/32-gbdt-sal/logs/build_t1_cf.log
