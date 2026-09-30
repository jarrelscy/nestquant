#!/bin/bash
# bg.sh NAME ARGS... : CPU training job, detached
cd /home/coder/git/nestquant/threads/33-search/xlatent
export PYTHONPATH=/tmp/nestquant/33-search/xlatent/pylib
NT=${NT:-8} nice -n 10 /home/coder/.venv/bin/python train.py "$@" --dev cpu > /tmp/nestquant/33-search/xlatent/logs/$1.log 2>&1
