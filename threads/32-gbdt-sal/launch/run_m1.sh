#!/bin/bash
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
nice -n 10 $PY build_m1.py glm52-heldout 12 > /tmp/nestquant/32-gbdt-sal/logs/build_m1_ho.log 2>&1
nice -n 10 $PY build_m1.py calib-fit 12 > /tmp/nestquant/32-gbdt-sal/logs/build_m1_cf.log 2>&1
echo built $(date -u)
B=ema32,ema128,mem_cur_state,tok_since_hit,hits16,sema32,sema128,sal16,mps128
M=/tmp/nestquant/32-gbdt-sal/models/ideas; L=/tmp/nestquant/32-gbdt-sal/logs/ideas
run() { local a=$1; shift; [ -f $M/$a.txt ] || nice -n 10 $PY train.py --band all --threads 12 "$@" --out $M/$a.txt > $L/train_$a.log 2>&1 & }
run m1   --target sal --obj tweedie:1.5 --feats "$B,m1_topic,m1_joint"
run m1s  --target sal --obj tweedie:1.5 --feats "$B,m1_state"
run m2   --target sal --obj tweedie:1.5 --feats "$B,m2_ent16,m2_p1_16,m2_ent_last"
run m12  --target sal --obj tweedie:1.5 --feats "$B,m1_topic,m1_joint,m1_state,m2_ent16,m2_p1_16,m2_ent_last"
wait
echo trained $(date -u)
