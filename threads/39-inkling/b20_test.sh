#!/bin/bash
until python3 -c "import json,sys;sys.exit(0 if json.load(open('/tmp/nq_inkling/enc_b20/fin/L5.json'))['rc']==0 else 1)" 2>/dev/null; do sleep 60; done
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/coder/git/nestquant/threads/05-exl3-harness:/home/coder/git/orbit-duet
export NQ35_BASE_K=2.0 TORCH_CUDA_ARCH_LIST=10.0 CUDA_VISIBLE_DEVICES=1
mkdir -p /tmp/nq_inkling/pre_b20 /tmp/nq_inkling/pre_l5
/home/coder/git/glm52/.venv/bin/python /tmp/nq_inkling/code/predecode_ink.py /tmp/nq_inkling/enc_b20 /tmp/nq_inkling/pre_b20 5
grep -h bpw /tmp/nq_inkling/enc_b20/fin/L5.json | head -2
ln -sf /tmp/nq_inkling/pre/L5.lv2.pt /tmp/nq_inkling/pre_l5/L5.lv2.pt; ln -sf /tmp/nq_inkling/pre/L5.lv4.pt /tmp/nq_inkling/pre_l5/L5.lv4.pt; cp /tmp/nq_inkling/pre/L5.relerr.json /tmp/nq_inkling/pre_l5/
cd /tmp/nq_inkling/code
for spec in "l5b175_2 pre_l5 2" "l5b175_hot64 pre_l5 hot:64" "l5b175_4 pre_l5 4" "l5b20_2 pre_b20 2"; do
  set -- $spec; O=/tmp/nq_inkling/kld/$1; rm -rf $O
  flock /tmp/nq_inkling/kld.lock /tmp/nq_inkling/venv/bin/torchrun --nproc-per-node 8 --master-port 29714 cap_ink.py --out $O --layers 0-47 --nfit 0,0 --no-stats \
    --wdir /tmp/nq_inkling/$2 --wsel $3 --wmax 6 --ref /tmp/nq_inkling/ref > $O.log 2>&1
done
/tmp/nq_inkling/venv/bin/python /tmp/nq_inkling/kld/tools/perkld.py l5b175_2 l5b175_hot64 l5b175_4 l5b20_2
