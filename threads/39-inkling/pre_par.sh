#!/bin/bash
# parallel predecode of the blended build, one layer stream per GPU
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/coder/git/nestquant/threads/05-exl3-harness:/home/coder/git/orbit-duet
export NQ35_BASE_K=1.75 TORCH_CUDA_ARCH_LIST=10.0
mkdir -p /tmp/nq_inkling/pre_mm
for g in 0 1 2 3 4 5 6 7; do
  ( for L in $(seq $g 8 47); do CUDA_VISIBLE_DEVICES=$g /home/coder/git/glm52/.venv/bin/python /tmp/nq_inkling/code/predecode_ink.py /tmp/nq_inkling/enc_b175mm /tmp/nq_inkling/pre_mm $L; done ) &
done
wait; echo PRE_DONE
