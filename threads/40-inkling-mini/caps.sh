#!/bin/bash
cd /tmp/nq_mini/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True INK_MM=/tmp/nq_mini/corpus_mm
i=0; GP=(3 4 1 2 0)
for W in native 2 4 hot:56 float; do
  tag=${W/:/}; O=/tmp/nq_mini/kld/v_$tag; wd=/tmp/nq_mini/pre_mm; ws=$W
  [ $W = native ] && { wd=/tmp/nq_mini/pre_native; ws=4; }
  rm -rf $O
  (CUDA_VISIBLE_DEVICES=${GP[$i]} ../venv/bin/torchrun --nproc-per-node 1 --master-port $((29780+i)) cap_mini.py --phase eval --out $O --nfit 1200,0 --wdir $wd --wsel $ws --ref /tmp/nq_mini/cap-mm > $O.log 2>&1; echo "caption $W $(grep "KLD vs ref" $O.log)") &
  i=$((i+1))
done
wait
echo CAPS_DONE
