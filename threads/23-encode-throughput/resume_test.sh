#!/bin/bash
# Resume invariance: nq_layer_batch --group 4 on L3 E0:16 SIGKILLed mid 2nd group, resumed, vs the reference layer.
#   resume_test.sh T12_DIR PREFIX GPU   (needs ${PREFIX}lay_ref from gate.sh)
T12D=$1; P=$2; G=$3
cd /home/coder/git/nestquant/threads/23-encode-throughput
source ../12-reference-encoder/env.sh >/dev/null 2>&1
export NQ23_T12=$T12D CUDA_VISIBLE_DEVICES=$G
PY=/home/coder/git/glm52/.venv/bin/python
S=/tmp/nestquant/23-encode-throughput; LG=$S/logs/$P; mkdir -p $LG
out=$S/${P}lay_res; rm -rf $out
$PY nq_layer_batch.py --layer 3 --stats $S/capsnap --out $out --experts 0:16 --group 4 --no-finalize > $LG/resume1.log 2>&1 &
pid=$!
until grep -q "^\[L3 E0\.\.E3\]" $LG/resume1.log 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 2; done
sleep 30; kill -9 $pid; wait $pid 2>/dev/null
echo "SIGKILL at $(date -u +%H:%M:%S) mid 2nd group: $(ls $out/L3/experts | wc -l) files: $(ls $out/L3/experts | tr '\n' ' ')" | tee $LG/resume_kill.txt
$PY nq_layer_batch.py --layer 3 --stats $S/capsnap --out $out --experts 0:16 --group 4 > $LG/resume2.log 2>&1
grep "^\[L3" $LG/resume1.log $LG/resume2.log
$PY cmp_layer.py $S/${P}lay_ref $out 3 2>&1 | grep -v "Warn\|from_numpy" | tee $LG/resume_cmp.txt
