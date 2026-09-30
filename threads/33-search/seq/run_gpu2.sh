#!/bin/bash
# one lock hold, 6 arms in parallel on GPUs 0-5 (each ~10-15 min)
cd /home/coder/git/nestquant/threads/33-search/seq
PY=/home/coder/git/prime-radiant/.venv/bin/python; L=/tmp/nestquant/33-search/seq/logs
export NT=6 OMP_NUM_THREADS=6
echo "start $(date -u)" > $L/gpu2.txt
r() { g=$1; n=$2; shift 2; CUDA_VISIBLE_DEVICES=$g timeout 3000 $PY train.py $n --eemb 0 --noval 1 "$@" > $L/$n.log 2>&1; }
r 0 t64d7 --H 64 --dil 7 --ep 15 --lw 0.5 &
r 1 t64d9 --H 64 --dil 9 --ep 15 --lw 0.5 &
r 2 t64r8 --H 64 --dil 8 --ep 15 --lw 0.5 --v2 1 --resid 1 &
r 3 t96d8 --H 96 --dil 8 --ep 15 --lw 0.5 &
r 4 t64d8f --H 64 --dil 8 --ep 15 --lw 0.5 --full 1 &
r 5 t64d8l2 --H 64 --dil 8 --ep 15 --lw 2.0 &
wait
g=0; for n in t64d7 t64d9 t64r8; do CUDA_VISIBLE_DEVICES=$g timeout 900 $PY infer_sm.py $n sm120tf > $L/sm_$n.log 2>&1 & g=$((g+1)); done; wait
for n in t96d8 t64d8f t64d8l2; do CUDA_VISIBLE_DEVICES=$g timeout 900 $PY infer_sm.py $n sm120tf > $L/sm_$n.log 2>&1 & g=$((g+1)); done; wait
echo "end $(date -u)" >> $L/gpu2.txt
