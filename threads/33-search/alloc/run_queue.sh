#!/bin/bash
# sequential CPU queue (<=18 procs at a time)
cd /home/coder/git/nestquant/threads/33-search/alloc
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python; O=/tmp/nestquant/33-search/alloc
while pgrep -f "[h]edge.py calib-fit" > /dev/null; do sleep 20; done
SRC=s5 nice -n 10 $PY ksweep.py calib-fit > $O/ksweep_calib_s5.log 2>&1
nice -n 10 $PY stack.py glm52-heldout > $O/stack_heldout.log 2>&1
nice -n 10 $PY stack.py calib-fit > $O/stack_calib.log 2>&1
echo queue done
