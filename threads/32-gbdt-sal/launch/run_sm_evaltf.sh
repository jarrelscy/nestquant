#!/bin/bash
cd /tmp/nestquant/32-gbdt-sal
NPROC=16 PT=2 ./run_sm_eval.sh sm120tf old old_mps v2 c_base c_lh s_base s_lh
NPROC=16 PT=2 ./run_sm_eval.sh sm120tfx old v2c c_base c_lh s_base s_lh
echo "done $(date -u)"
