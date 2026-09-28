source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
cd /home/coder/git/nestquant/threads/02-feedback-conflict
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=/tmp/nestquant/02-feedback-conflict
( for m in 0 2; do /home/coder/git/glm52/.venv/bin/python frontier.py glm $m > $L/fr_glm_$m.log 2>&1; done
  /home/coder/git/glm52/.venv/bin/python frontier.py mimogauss 0 > $L/fr_mimogauss_0.log 2>&1 ) &
( /home/coder/git/glm52/.venv/bin/python frontier.py glm 1 > $L/fr_glm_1.log 2>&1
  /home/coder/git/glm52/.venv/bin/python frontier.py mimogauss 2 > $L/fr_mimogauss_2.log 2>&1
  /home/coder/git/glm52/.venv/bin/python frontier.py glmgauss 0 > $L/fr_glmgauss_0.log 2>&1 ) &
wait
