#!/bin/bash
# Thread 23 single-process re-gate (candidate nq_encode_batch_f23 via f23_run.py) of a batch-side change against EXISTING reference arms (gate.sh outputs).
#   gate1.sh T12_DIR REF_PREFIX PREFIX GPU
# Same 27 experts + layer paths as gate.sh, one process at a time on one GPU:
#   group 4 (ALL), group 3 reversed, group 1 (8 experts) vs ${REF_PREFIX}ref;  L3 E0:16 --group 4 vs ${REF_PREFIX}lay_ref;
#   --stats-mm self-blend L30 E166:170 vs ${REF_PREFIX}mm_ref
set -u
T12D=$1; R=$2; P=$3; G=$4
HERE=/home/coder/git/nestquant/threads/23-encode-throughput; cd $HERE
set +u; source ../12-reference-encoder/env.sh >/dev/null 2>&1; set -u
export NQ23_T12=$T12D CUDA_VISIBLE_DEVICES=$G
PY0=/home/coder/git/glm52/.venv/bin/python
PY="$PY0 f23_run.py"      # candidate encoder via the alias shim (pinned files untouched)
S=/tmp/nestquant/23-encode-throughput; LG=$S/logs/$P; mkdir -p $LG
EIG=30:169,3:2,3:60,43:180,76:108,56:96
FIX=3:8,20:9,43:13,56:42,66:20,77:12
ORD=3:0,16:36,16:92,16:165,40:7,49:36,66:165,76:200,77:11,30:128,8:17,25:201,35:50,52:3,70:222
ALL=$EIG,$FIX,$ORD
REV=$(echo $ALL | tr , '\n' | tac | paste -sd,)
G1=30:169,3:2,43:180,76:108,3:8,77:12,16:36,70:222
CAP=$S/capsnap
T0=$(date +%s)
$PY check_bitid.py batch --tag ${P}g4 --group 4 --experts $ALL > $LG/g4.log 2>&1
$PY nq_layer_batch.py --layer 3 --stats $CAP --out $S/${P}lay_bat --experts 0:16 --group 4 > $LG/lay_bat.log 2>&1
$PY nq_layer_batch.py --layer 30 --stats $CAP --stats-mm $CAP --out $S/${P}mm_bat --experts 166:170 --group 4 > $LG/mm_bat.log 2>&1
$PY check_bitid.py batch --tag ${P}g3r --group 3 --experts $REV > $LG/g3r.log 2>&1
$PY check_bitid.py batch --tag ${P}g1 --group 1 --experts $G1 > $LG/g1.log 2>&1
echo "gate1 runs done in $(( ($(date +%s)-T0)/60 )) min (refs: ${R})" | tee $LG/summary.txt
for t in g4:$ALL g3r:$ALL g1:$G1; do
  tag=${t%%:*}; ex=${t#*:}
  echo "== ($tag) $(grep 'batch total' $LG/$tag.log)" >> $LG/summary.txt
  $PY0 check_bitid.py cmp --ref-tag ${R}ref --tag ${P}$tag --experts $ex 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
done
echo "== layer path (group 4) $(grep amortised $LG/lay_bat.log | tail -n 1)" >> $LG/summary.txt
$PY0 cmp_layer.py $S/${R}lay_ref $S/${P}lay_bat 3 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
echo "== --stats-mm self-blend L30 E166:170 $(grep amortised $LG/mm_bat.log | tail -n 1)" >> $LG/summary.txt
$PY0 cmp_layer.py $S/${R}mm_ref $S/${P}mm_bat 30 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
grep "ALL BIT\|FAIL\|MISMATCH\|LAYER\|==" $LG/summary.txt
