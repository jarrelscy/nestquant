source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
T=/home/coder/git/nestquant/threads
export PYTHONPATH=$T/29-outlier-gap:$T/27-pv-tune:$T/05-exl3-harness:$T/12-reference-encoder:$T/19-full-capture:$T/25-campaign:$T/26-vision-calib:/home/coder/git/orbit-duet:$PYTHONPATH
PY=/home/coder/git/glm52/.venv/bin/python
