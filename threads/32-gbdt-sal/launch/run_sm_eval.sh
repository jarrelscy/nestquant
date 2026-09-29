#!/bin/bash
# run_sm_eval.sh STREAM name=model ...   (pretrained refs added: old native / band-all, v2 count-substituted)
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
S=/home/coder/git/nestquant/streaming; M=/tmp/nestquant/32-gbdt-sal/models/sm120
st=$1; shift; A=""
for a in "$@"; do case $a in
  old) A="$A old=$S/gbdt_p64_s5.txt@native";; old_ba) A="$A old_ba=$S/gbdt_p64_s5.txt";;
  v2c) A="$A v2c=$S/gbdt_v2sal_p64.txt";; v2) A="$A v2=$S/gbdt_v2sal_p64.txt";; old_mps) A="$A old_mps=$S/gbdt_p64_s5.txt@mps";; *) A="$A $a=$M/$a.txt";; esac; done
NPROC=${NPROC:-12} PT=${PT:-4} nice -n 10 /home/coder/git/glm52/.venv/bin/python sm120.py eval $st $A
