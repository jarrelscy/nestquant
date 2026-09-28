#!/bin/bash
cd /home/coder/git/nestquant/threads/09-post-tuning
for model in glm glm90 mimo; do for bits in 2 4; do
 for g in suv rot bias lut vlut tile suv,rot,bias,lut suv,rot,bias,lut,tile; do for lr in 3e-4 1e-3 3e-3; do
  [ -f results/${model}${bits}_${g//,/+}_lr$lr.json ] || ./run.sh tune.py $model $bits $g --lr $lr --tag lr$lr 2>&1 | grep -v Warn | tail -2
 done; done
 for r in 8 32; do for lr in 1e-4 3e-4 1e-3; do
  [ -f results/${model}${bits}_none_r${r}_lr$lr.json ] || ./run.sh tune.py $model $bits none --rank $r --lr $lr --tag lr$lr 2>&1 | grep -v Warn | tail -2
 done; done
done; done
