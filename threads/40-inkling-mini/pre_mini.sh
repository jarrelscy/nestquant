#!/bin/bash
# predecode each finalized layer to bf16 L2/L4 (eval weights), in layer order
cd /tmp/nq_mini/code
NQ=/home/coder/git/nestquant
export CUDA_VISIBLE_DEVICES=${GPU:-4} LD_LIBRARY_PATH=$NQ/threads/06-expert-objective/lib:$LD_LIBRARY_PATH PYTHONPATH=$NQ/threads/05-exl3-harness:/home/coder/git/orbit-duet NQ35_BASE_K=1.75 TORCH_CUDA_ARCH_LIST=10.0
for L in $(seq 0 23); do
  until grep -q '"rc": 0' /tmp/nq_mini/enc_b175mm/fin/L$L.json 2>/dev/null; do sleep 30; done
  /home/coder/git/glm52/.venv/bin/python predecode_ink.py /tmp/nq_mini/enc_b175mm /tmp/nq_mini/pre_mm $L || echo "PRE_FAIL L$L"
done
echo PRE_DONE
