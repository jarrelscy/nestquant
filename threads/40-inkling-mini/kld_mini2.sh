#!/bin/bash
# rerun with per-token KLD saved: text arms (W=4), then caption arms + caption floor in parallel (W=1 each)
cd /tmp/nq_mini/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=../venv/bin/torchrun
for W in 2 4 hot:56 float; do
  tag=${W/:/}; O=/tmp/nq_mini/kld/t_$tag; rm -rf $O
  CUDA_VISIBLE_DEVICES=1,2,3,4 $T --nproc-per-node 4 --master-port 29761 cap_mini.py --phase eval --out $O --wdir /tmp/nq_mini/pre_mm --wsel $W --ref /tmp/nq_mini/cap-txt > $O.log 2>&1
  echo "text $W $(grep 'KLD vs ref' $O.log)"
done
until ! pgrep -f "out /tmp/nq_mini/kld/v_float" > /dev/null; do sleep 30; done
i=0; GP=(1 2 3 4 0)
for W in native 2 4 hot:56 float; do
  tag=${W/:/}; O=/tmp/nq_mini/kld/v_$tag
  [ $W = float ] && [ -f $O/final/klv.r0.pt ] && continue
  wd=/tmp/nq_mini/pre_mm; ws=$W; [ $W = native ] && { wd=/tmp/nq_mini/pre_native; ws=4; }
  rm -rf $O
  (INK_MM=/tmp/nq_mini/corpus_mm CUDA_VISIBLE_DEVICES=${GP[$i]} $T --nproc-per-node 1 --master-port $((29780+i)) cap_mini.py --phase eval --out $O --nfit 1200,0 --wdir $wd --wsel $ws --ref /tmp/nq_mini/cap-mm > $O.log 2>&1; echo "caption $W $(grep 'KLD vs ref' $O.log)") &
  i=$((i+1))
done
wait
echo KLD2_DONE
