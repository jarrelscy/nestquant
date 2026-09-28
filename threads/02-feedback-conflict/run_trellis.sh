source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
cd /home/coder/git/nestquant/threads/02-feedback-conflict
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=/tmp/nestquant/02-feedback-conflict
for a in "glm 0" "glm 1" "mimogauss 0" "mimogauss 2"; do set -- $a; /home/coder/git/glm52/.venv/bin/python trellis_frontier.py $1 $2 > $L/tr_$1_$2.log 2>&1; done
