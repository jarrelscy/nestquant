#!/bin/bash
# T32 after the /tmp wipe: SM120 CPU chain (PRIVATE outputs under private/sm120), <= ~8 threads so it can overlap the
# KLD pass.  (1) sm120tf serve table with pretrained models; (2) prep + rows + count-only GBDT arms; (3) the evals
# that died in the wipe: sm120dec c_*/sA_*/sB_* and sm120tfx.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; C=/home/coder/git/nestquant/threads/32-gbdt-sal
until grep -q "TF done" $O/logs/rebuild_tf.log; do sleep 60; done
cd $O
echo "== sm120tf serve table $(date -u)"
NPROC=4 PT=2 $C/launch/run_sm_eval.sh sm120tf old old_mps v2
echo "== prep $(date -u)"
NPROC=8 $C/launch/run_sm_prep.sh sm120dec sm120all probemix calib-fit glm52-heldout
echo "== rows+train $(date -u)"
sed -e 's/NPROC=16 \$PY/NPROC=8 $PY/; s/NPROC=12 \$PY/NPROC=8 $PY/; s/THR=48/THR=8/' $C/launch/run_sm_rows.sh > $O/sm_rows.tmp.sh
bash $O/sm_rows.tmp.sh
echo "== evals $(date -u)"
NPROC=4 PT=2 $C/launch/run_sm_eval.sh sm120dec c_base c_long c_lh c_dom sA_base sA_long sA_lh sA_dom sB_base sB_long sB_lh sB_dom
NPROC=4 PT=2 $C/launch/run_sm_eval.sh sm120tfx old v2c c_base c_lh s_base s_lh
echo "sm rebuild done $(date -u)"
