#!/bin/bash
# Thread 23 widened bit-identity gate.  gate.sh T12_DIR PREFIX GPU_A GPU_B GPU_C
#   T12_DIR: pinned copy of thread 12's committed encoder (NQ23_T12);  PREFIX: scratch tag prefix
# (1) 27 experts (6 eigen-plane cases, 6 T22 fixed-set, 15 ordinary L3-L77) ref vs batch
# (2) group composition: group 3 (GPU B; group 8 exceeds the 12 GB cap), group 4 reversed (GPU C), group 2 shuffled (GPU B), group 1 (GPU A, 8 experts)
# (3) layer path: nq_layer (ref_layer.py) vs nq_layer_batch.py --group 4 on L3 E0:16: experts + tp0-7 + manifest
# (5) --stats-mm self-blend (T26 BlendCapture, text as vision) layer path L30 E166:170 (GPU D)
# (4) resume: nq_layer_batch --group 4 killed mid 2nd group, resumed -> vs (3) reference
set -u
T12D=$1; P=$2; GA=$3; GB=$4; GC=$5; GD=${6:-$5}
HERE=/home/coder/git/nestquant/threads/23-encode-throughput; cd $HERE
set +u; source ../12-reference-encoder/env.sh >/dev/null 2>&1; set -u
export NQ23_T12=$T12D
PY=/home/coder/git/glm52/.venv/bin/python
S=/tmp/nestquant/23-encode-throughput; LG=$S/logs/$P; mkdir -p $LG
EIG=30:169,3:2,3:60,43:180,76:108,56:96
FIX=3:8,20:9,43:13,56:42,66:20,77:12
ORD=3:0,16:36,16:92,16:165,40:7,49:36,66:165,76:200,77:11,30:128,8:17,25:201,35:50,52:3,70:222
ALL=$EIG,$FIX,$ORD
H1=$EIG,$FIX,3:0; H2=16:36,16:92,16:165,40:7,49:36,66:165,76:200,77:11,30:128,8:17,25:201,35:50,52:3,70:222
REV=$(echo $ALL | tr , '\n' | tac | paste -sd,)
SHUF=$(echo $ALL | tr , '\n' | shuf --random-source=<(yes 23) | paste -sd,)
G1=30:169,3:2,43:180,76:108,3:8,77:12,16:36,70:222
CAP=$S/capsnap
run() { local gpu=$1; shift; CUDA_VISIBLE_DEVICES=$gpu "$@"; }
resume_test() {
  local out=$S/${P}lay_res; rm -rf $out
  run $GC $PY nq_layer_batch.py --layer 3 --stats $CAP --out $out --experts 0:16 --group 4 --no-finalize > $LG/resume1.log 2>&1 &
  local pid=$!
  until grep -q "^\[L3 E0\.\.E3\]" $LG/resume1.log 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  sleep 30; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "killed mid-group: $(ls $out/L3/experts | wc -l) files present: $(ls $out/L3/experts | tr '\n' ' ')" | tee $LG/resume_kill.txt
  run $GC $PY nq_layer_batch.py --layer 3 --stats $CAP --out $out --experts 0:16 --group 4 > $LG/resume2.log 2>&1
}
T0=$(date +%s)
( run $GA $PY check_bitid.py ref --ref-tag ${P}ref --experts $H1 > $LG/refA.log 2>&1
  run $GA $PY ref_layer.py --layer 3 --stats $CAP --out $S/${P}lay_ref --experts 0:16 > $LG/lay_ref.log 2>&1
  run $GA $PY check_bitid.py batch --tag ${P}g1 --group 1 --experts $G1 > $LG/g1.log 2>&1 ) &
( run $GB $PY check_bitid.py ref --ref-tag ${P}ref --experts $H2 > $LG/refB.log 2>&1
  run $GB $PY check_bitid.py batch --tag ${P}g3 --group 3 --experts $ALL > $LG/g3.log 2>&1
  run $GB $PY check_bitid.py batch --tag ${P}g2s --group 2 --experts $SHUF > $LG/g2s.log 2>&1 ) &
( run $GC $PY nq_layer_batch.py --layer 3 --stats $CAP --out $S/${P}lay_bat --experts 0:16 --group 4 > $LG/lay_bat.log 2>&1
  resume_test
  run $GC $PY check_bitid.py batch --tag ${P}g4r --group 4 --experts $REV > $LG/g4r.log 2>&1 ) &
( run $GD $PY ref_layer.py --layer 30 --stats $CAP --stats-mm $CAP --out $S/${P}mm_ref --experts 166:170 > $LG/mm_ref.log 2>&1
  run $GD $PY nq_layer_batch.py --layer 30 --stats $CAP --stats-mm $CAP --out $S/${P}mm_bat --experts 166:170 --group 4 > $LG/mm_bat.log 2>&1 ) &
wait
echo "gate runs done in $(( ($(date +%s)-T0)/60 )) min" | tee $LG/summary.txt
for t in g3:$ALL g4r:$ALL g2s:$ALL g1:$G1; do
  tag=${t%%:*}; ex=${t#*:}
  echo "== ($tag) $(grep 'batch total' $LG/$tag.log)" >> $LG/summary.txt
  $PY check_bitid.py cmp --ref-tag ${P}ref --tag ${P}$tag --experts $ex 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
done
echo "== layer path (group 4) $(grep amortised $LG/lay_bat.log | tail -n 1)" >> $LG/summary.txt
$PY cmp_layer.py $S/${P}lay_ref $S/${P}lay_bat 3 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
echo "== resume: $(cat $LG/resume_kill.txt)" >> $LG/summary.txt
$PY cmp_layer.py $S/${P}lay_ref $S/${P}lay_res 3 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
echo "== --stats-mm self-blend L30 E166:170 $(grep amortised $LG/mm_bat.log | tail -n 1)" >> $LG/summary.txt
$PY cmp_layer.py $S/${P}mm_ref $S/${P}mm_bat 30 2>&1 | grep -v "Warn\|from_numpy" >> $LG/summary.txt
$PY - >> $LG/summary.txt <<PYEOF
import torch, glob
for f in sorted(glob.glob("$S/${P}ref/*.pt")):
    r = torch.load(f, weights_only=False)
    print("lr_rank", f.split("/")[-1][:-3], r["meta"].get("lr_rank"), "ref %.0f s" % r["_ref"]["seconds"])
PYEOF
grep -h "^ref L" $LG/refA.log $LG/refB.log | awk '{print $4}' | sort -n | awk '{a[NR]=$1} END {print "ref median s/expert:", a[int((NR+1)/2)]}' >> $LG/summary.txt
grep "ALL BIT\|FAIL\|LAYER\|==\|killed\|median" $LG/summary.txt
