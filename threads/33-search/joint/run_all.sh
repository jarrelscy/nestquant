#!/bin/bash
# T33i production jF_all: calib chains (same split) + all 6 sm120tf tasks; one GPU, hard cap 20 min for the hold
cd /home/coder/git/nestquant/threads/33-search/joint
export LAYOUT=k0; LG=/tmp/nestquant/33-search/joint/logs
echo "start $(date)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 10 timeout 1140 /home/coder/git/prime-radiant/.venv/bin/python -u train.py jF_all \
  --arch tf --d 96 --nl 2 --tw 1 --noemb --budget 10 --valmin 2 \
  --tfadd embedding-drift-monitor,fin-saccr-rwa,formal-crypto,freight-dispatch-shift,pretrain-shard-corruption,sound-change-cascade \
  --score "" > $LG/jF_all.log 2>&1
echo "done rc=$? $(date)"
