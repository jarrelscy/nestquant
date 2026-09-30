#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/process
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=20 LAYOUT=k0
PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
for a in "k0_v2r v2" "k0_v2proc v2,hks,hmm,bou,kf" "k0_v2bou v2,bou" "k0_v2hmm v2,hmm" "k0_v2hks v2,hks"; do
  set -- $a; [ -f /tmp/nestquant/33-search/process/models/$1.txt ] || $PY train.py $1 $2 60
done
M=/tmp/nestquant/33-search/process/models
A="v2_k0=v2"; for n in k0_v2r k0_v2proc k0_v2bou k0_v2hmm k0_v2hks; do A="$A $n=$M/$n.txt"; done
for c in glm52-heldout calib-val; do HMS=0.4,0.5,0.6,0.7,0.8 NPROC=8 $PY evalp.py $c $A k0_hks=sa:hks_pred*mps k0_bou=sa:bou_rate*mps k0_hmm=sa:hmm_pred*mps k0_ema=sa:sema128 2>&1 | grep -v Warn; done
