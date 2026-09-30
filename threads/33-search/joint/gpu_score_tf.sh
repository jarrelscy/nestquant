#!/bin/bash
# short GPU hold: score j_tf_tw_ne on sm120tf (inputs pre-built on CPU in tmp_tfX)
cd /home/coder/git/nestquant/threads/33-search/joint
echo "got lock $(date)"
LAYOUT=k0 DEV=cuda KEEP=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 timeout 900 nice -n 10 \
  /home/coder/git/prime-radiant/.venv/bin/python -u score_stream.py sm120tf /tmp/nestquant/33-search/joint/tmp_tfX j_tf_tw_ne
echo "released $(date)"
