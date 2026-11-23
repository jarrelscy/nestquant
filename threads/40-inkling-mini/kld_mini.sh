#!/bin/bash
# full-model KLD vs native, text (W=4, same split as cap-txt) and captions (W=1, same as cap-mm)
cd /tmp/nq_mini/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=../venv/bin/torchrun
for W in ${ARMS:-2 4 hot:56 float}; do
  tag=${W/:/}
  O=/tmp/nq_mini/kld/t_$tag
  [ -f $O/final/kld.json ] || CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,5} $T --nproc-per-node 4 --master-port 29761 cap_mini.py --phase eval \
      --out $O --wdir /tmp/nq_mini/pre_mm --wsel $W --ref /tmp/nq_mini/cap-txt > $O.log 2>&1
  echo "text $W $(grep 'KLD vs ref' $O.log)"
  O=/tmp/nq_mini/kld/v_$tag
  [ -f $O/final/kld.json ] || INK_MM=/tmp/nq_mini/corpus_mm CUDA_VISIBLE_DEVICES=${MMGPU:-0} $T --nproc-per-node 1 --master-port 29762 cap_mini.py --phase eval \
      --out $O --nfit 1200,0 --wdir /tmp/nq_mini/pre_mm --wsel $W --ref /tmp/nq_mini/cap-mm > $O.log 2>&1
  echo "caption $W $(grep 'KLD vs ref' $O.log)"
done
echo KLD_DONE
