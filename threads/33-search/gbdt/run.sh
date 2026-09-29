#!/bin/bash
# run.sh SCRIPT ARGS...  (nice, venv, pylib)
cd /home/coder/git/nestquant/threads/33-search/gbdt
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
exec nice -n 10 /home/coder/git/glm52/.venv/bin/python "$@"
