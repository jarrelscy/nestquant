#!/bin/bash
# overfitting diagnostics: (a) no-early-stopping curves; (b) training-row scaling on data-rich MiMo
cd /home/coder/git/nestquant/threads/09-post-tuning
run(){ ./run.sh tune.py "$@" 2>&1 | grep -v Warn | tail -2; }
for mb in "glm90 2" "glm90 4" "mimo 2"; do
  run $mb suv,rot,bias,lut,tile --lr 1e-3 --epochs 40 --patience 1000 --no-select --track-train --tag ovf
  run $mb none --rank 32 --lr 3e-4 --epochs 40 --patience 1000 --no-select --track-train --tag ovf
done
for bits in 2; do for n in 1638 4096 16384; do
  run mimo $bits bias --lr 1e-3 --rows $n
  run mimo $bits suv,rot,bias,lut,tile --lr 3e-4 --rows $n
  run mimo $bits none --rank 32 --lr 1e-4 --rows $n
done; done
for bits in 2 4; do for lr in 3e-4 1e-3; do
  run glms03 $bits suv --lr $lr --tag lr$lr
  run glms03 $bits bias --lr $lr --tag lr$lr
  run glms03 $bits suv,rot,bias,lut --lr $lr --tag lr$lr
done; run glms03 $bits none --rank 32 --lr 1e-4 --tag lr1e-4; done
