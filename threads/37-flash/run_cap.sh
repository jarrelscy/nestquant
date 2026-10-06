#!/bin/bash
# run_cap.sh txt|mm [LAYERS]   full T37 capture on 8 A100s under the shared gpu.lock (PRIVATE trace for txt)
c=$1; L=${2:-0-44}
out=/tmp/nestquant/37-flash/cap-$c
tr=""; [ "$c" = txt ] && tr="--trace /tmp/nestquant/37-flash/private/trace"
cd /home/coder/git/nestquant/threads/37-flash
exec flock /tmp/nestquant/33-search/gpu.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 nice -n 10 \
  /tmp/venv-t37g/bin/torchrun --nproc-per-node 8 --master-port 29637 capture37g.py --corpus $c --out $out --layers $L $tr
