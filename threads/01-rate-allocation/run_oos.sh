source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=/home/coder/git/orbit-duet
P=/home/coder/git/glm52/.venv/bin/python; T=/tmp/nestquant/01-rate-allocation
for LE in "16 92" "16 165" "49 36" "49 92" "49 165" "66 36" "66 92" "66 165"; do
$P oos_innov.py glm $LE >> $T/oos.log 2>&1
done
