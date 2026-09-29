#!/bin/bash
cd /home/coder/git/nestquant/threads/32-gbdt-sal
S=/home/coder/git/nestquant/streaming; export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
nice -n 10 /home/coder/git/glm52/.venv/bin/python fixed_size.py glm52-heldout old=$S/gbdt_p64_s5.txt v2=$S/gbdt_v2sal_p64.txt > /tmp/nestquant/32-gbdt-sal/logs/fixed_size.log 2>&1
nice -n 10 /home/coder/git/glm52/.venv/bin/python fixed_size.py calib-fit old=$S/gbdt_p64_s5.txt v2=$S/gbdt_v2sal_p64.txt > /tmp/nestquant/32-gbdt-sal/logs/fixed_size_cf.log 2>&1
