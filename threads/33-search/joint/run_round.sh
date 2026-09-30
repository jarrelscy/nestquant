#!/bin/bash
# usage (inside flock): run_round.sh CFGFILE   -- each line "NAME args..." -> one GPU each (max 8), staggered starts
cd /home/coder/git/nestquant/threads/33-search/joint
LG=/tmp/nestquant/33-search/joint/logs
g=0
while read -r name args; do
  [ -z "$name" ] && continue
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nice -n 10 timeout 5100 /home/coder/git/prime-radiant/.venv/bin/python -u train.py $name $args > $LG/$name.log 2>&1 &
  g=$((g+1)); sleep 45
done < "$1"
wait
echo "round done $(date)"
