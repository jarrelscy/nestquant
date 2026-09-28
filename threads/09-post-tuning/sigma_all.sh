#!/bin/bash
cd /home/coder/git/nestquant/threads/09-post-tuning
for m in glm_16_92 glm_16_165 glm_49_36 glm_49_92 glm_49_165 glm_66_36 glm_66_92 glm_66_165; do
  echo "== $m"
  ./run.sh refit_sweep.py $m 90 2:0.03 2:0.3 4:0.03 4:0.3 2>&1 | python3 fmt.py
  ./run.sh refit_sweep.py $m full 2:0.03 2:0.3 4:0.03 4:0.3 2>&1 | python3 fmt.py
done
