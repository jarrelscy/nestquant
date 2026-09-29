#!/bin/bash
# usage: run_joint.sh NAME MODE EPOCHS NLAYERS SUB
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
CUDA_VISIBLE_DEVICES= JT=${JT:-48} nice -n 10 /home/coder/git/glm52/.venv/bin/python joint.py train "$@" > /tmp/nestquant/32-gbdt-sal/logs/joint/$1.log 2>&1
