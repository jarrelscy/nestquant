#!/bin/bash
# one GPU lock hold: 3 arms sequentially on GPU 0
cd /home/coder/git/nestquant/threads/33-search/seq
PY=/home/coder/git/prime-radiant/.venv/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 NT=8
L=/tmp/nestquant/33-search/seq/logs; mkdir -p $L
echo "start $(date -u)" > $L/gpu1.txt
timeout 1500 $PY train.py tcn_a --H 48 --dil 7 --ep 5 --noval 1 > $L/tcn_a.log 2>&1
timeout 1500 $PY train.py tcn_r --H 48 --dil 7 --ep 5 --v2 1 --resid 1 --noval 1 > $L/tcn_r.log 2>&1
timeout 1500 $PY train.py tcn_rl --H 48 --dil 7 --ep 5 --v2 1 --resid 1 --lw 0.5 --noval 1 > $L/tcn_rl.log 2>&1
echo "end $(date -u)" >> $L/gpu1.txt
