#!/bin/bash
cd /tmp/nestquant/32-gbdt-sal
C="c_base c_long c_lh c_dom"; SA="sA_base sA_long sA_lh sA_dom"; SB="sB_base sB_long sB_lh sB_dom"; S="s_base s_long s_lh s_dom"
NPROC=25 PT=2 ./run_sm_eval.sh glm52-heldout $C $S
NPROC=16 PT=3 ./run_sm_eval.sh sm120dec $C $SA $SB
NPROC=25 PT=2 ./run_sm_eval.sh probemix old v2c $C $S
echo "done $(date -u)"
