#!/bin/bash
# full text capture: forward (MoE inputs + trace + reference top-64) then stats
cd /tmp/nq_mini/code
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,5} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=../venv/bin/torchrun
$T --nproc-per-node 4 --master-port 29741 cap_mini.py --phase fwd --out /tmp/nq_mini/cap-txt --trace /tmp/nq_mini/cap-txt/private/trace || exit 1
$T --nproc-per-node 4 --master-port 29742 cap_mini.py --phase stats --out /tmp/nq_mini/cap-txt || exit 1
echo CAP_TXT_DONE
