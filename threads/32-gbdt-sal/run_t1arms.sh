#!/bin/bash
# T32 T1/T2 token-affinity arms (CPU): band-all v2 base + build_t1.py features, next-64 salience, tweedie 1.5
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
M=/tmp/nestquant/32-gbdt-sal/models/ideas; L=/tmp/nestquant/32-gbdt-sal/logs/ideas; mkdir -p $M $L
B=ema32,ema128,mem_cur_state,tok_since_hit,hits16,sema32,sema128,sal16,mps128
T1=t1_e16,t1_e64,t1b_e16,t1f_e16,t1f_last
T2=t2_n16,t2_n64
run() { local a=$1; shift; [ -f $M/$a.txt ] || nice -n 10 $PY train.py --band all --threads ${THR:-12} "$@" --out $M/$a.txt > $L/train_$a.log 2>&1 & }
run t1      --target sal --obj tweedie:1.5 --feats "$B,$T1"
run t2up    --target sal --obj tweedie:1.5 --feats "$B,$T1,$T2"
run t2only  --target sal --obj tweedie:1.5 --feats "$B,$T2"
wait
echo "trained $(date -u)"
