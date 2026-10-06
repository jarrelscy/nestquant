#!/bin/bash
# run_capo.sh [LAYERS] [EXTRA...]: trace-only routing capture on the online Flash-generated corpus (PRIVATE trace)
L=${1:-0-44}; shift
cd /home/coder/git/nestquant/threads/37-flash
exec flock /tmp/nestquant/33-search/gpu.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 nice -n 10 \
  /tmp/venv-t37g/bin/torchrun --nproc-per-node 8 --master-port 29638 capture37o.py --corpus online --out /tmp/nestquant/37-flash/cap-online --layers $L --fit-split fit "$@"
