#!/bin/bash
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
nice -n 10 $PY build_dx.py glm52-heldout 16 > /tmp/nestquant/32-gbdt-sal/logs/build_dx_ho.log 2>&1
nice -n 10 $PY build_dx.py calib-fit 16 > /tmp/nestquant/32-gbdt-sal/logs/build_dx_cf.log 2>&1
B=ema32,ema128,mem_cur_state,tok_since_hit,hits16,sema32,sema128,sal16,mps128
M=/tmp/nestquant/32-gbdt-sal/models/ideas
[ -f $M/dx.txt ] || nice -n 10 $PY train.py --band all --threads 12 --target sal --obj tweedie:1.5 --feats "$B,dx_lo,dx_hi,dx_self" --out $M/dx.txt > /tmp/nestquant/32-gbdt-sal/logs/ideas/train_dx.log 2>&1
echo done $(date -u)
