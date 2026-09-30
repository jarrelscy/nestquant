#!/bin/bash
# decode-fold training (T33j folds): calib chains 0-27 + 4 sm120tf train tasks, score the 2 held-out tasks
cd /home/coder/git/nestquant/threads/33-search/joint
export LAYOUT=k0
LG=/tmp/nestquant/33-search/joint/logs
while ! grep -q done $LG/tfprep.log; do sleep 10; done
echo "start $(date)"
E1=embedding-drift-monitor; FI=fin-saccr-rwa; FO=formal-crypto; SO=sound-change-cascade; FR=freight-dispatch-shift; PR=pretrain-shard-corruption
g=0
for spec in "1 $FO,$FR,$PR,$SO $E1,$FI" "2 $E1,$FI,$FR,$PR $FO,$SO" "3 $E1,$FI,$FO,$SO $FR,$PR"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nice -n 10 timeout 1500 /home/coder/git/prime-radiant/.venv/bin/python -u train.py jF$1 \
    --arch tf --d 96 --nl 2 --tw 1 --noemb --budget 10 --valmin 2 --tfadd $2 --scoretf $3 --score "" > $LG/jF$1.log 2>&1 &
  g=$((g+1)); sleep 20
done
wait
echo "done $(date)"
