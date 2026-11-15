#!/bin/bash
# predecode every finalized layer; runs on GPU 0 alongside encoders
source /dev/null
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/coder/git/nestquant/threads/05-exl3-harness:/home/coder/git/orbit-duet
export NQ35_BASE_K=1.75 TORCH_CUDA_ARCH_LIST=10.0 CUDA_VISIBLE_DEVICES=${GPU:-0}
for L in $(seq 0 47); do
  until python3 -c "import json,sys;sys.exit(0 if json.load(open('/tmp/nq_inkling/enc_b175/fin/L$L.json'))['rc']==0 else 1)" 2>/dev/null; do sleep 120; done
  nice -n 10 /home/coder/git/glm52/.venv/bin/python /tmp/nq_inkling/code/predecode_ink.py /tmp/nq_inkling/enc_b175 /tmp/nq_inkling/pre $L
done
