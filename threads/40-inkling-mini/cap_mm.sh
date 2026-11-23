#!/bin/bash
# image calibration capture (corpus_mm): forward then stats, one GPU
cd /tmp/nq_mini/code
export CUDA_VISIBLE_DEVICES=${GPUS:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True INK_MM=/tmp/nq_mini/corpus_mm
T=../venv/bin/torchrun
$T --nproc-per-node 1 --master-port 29751 cap_mini.py --phase fwd --out /tmp/nq_mini/cap-mm --nfit 1200,0 || exit 1
$T --nproc-per-node 1 --master-port 29752 cap_mini.py --phase stats --out /tmp/nq_mini/cap-mm --nfit 1200,0 || exit 1
echo CAP_MM_DONE
