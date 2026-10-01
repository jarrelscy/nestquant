#!/bin/bash
# T34 BF16-teacher panel (brandonmusic/GLM-5.3-BF16-full-logits, confirmation lane, 64 windows x 2048):
# KL(teacher || arm), full vocab, fp32 log-softmax (nq_e2e NQ_TEACHER mode; FP8 reference scored as arm "fp8").
# 64 windows / WORLD 8 contig = 8 windows per rank; chain=map with chain=arange(64): every dynamic predictor
# starts COLD per 2047-token window.  Two gpu.lock holds (each < 90 min, timeout 5300 s):
#   A passT34TA: fp8(ref) tr3 nq4 nq2 serve jF77_hm07 jF173 nqS173   (173 = 3.455 bpw = TR3's measured 3.4557)
#   B passT34TB: k0_hm10 jF77_hm10 jF26 jF51 jF128 k26_v2 jF182 nqS182
# Sweep arms (jF26/51/128/182) run at hm 0.7 (jF77 hold-A value); kB_manifest (no fixed, floating_default all
# 256; its first 77 == k0 floating_default set).
#   run_teacher.sh A|B|AB
set -euo pipefail
O=/tmp/nestquant/34-tr3; PD=/tmp/nestquant/src/predec; S=/home/coder/git/nestquant/streaming
G=/tmp/nestquant/32-gbdt-sal; J=/tmp/nestquant/33-search/joint; M26=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh
LO=$PD/farm/nq2; HI=$PD/farm/nq4
AD=adapt:lo=$LO,hi=$HI,chain=map,salstat=1
JF="$AD,predictor=gbdt,gmode=sync,joint=$J/models/jF_all.pt"
V2="$AD,predictor=gbdt,gbdt_model=$S/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256"
CA="--cand nq4=dir:$HI --cand nq2=dir:$LO"      # tr3 dropped (coordinator 2026-09-30: published numbers taken as-is)
CA="$CA --cand serve=$AD,manifest=$M26,predictor=gbdt,gbdt_scale=mps,gmode=sync"
CA="$CA --cand jF77_hm07=$JF,manifest=$G/k0_manifest.json,n_float=77,hm=0.7"
CA="$CA --cand jF173=$JF,manifest=$G/kB_manifest.json,n_float=173,hm=0.7"
CA="$CA --cand nqS173=mix:lo=$LO,hi=$HI,set=$O/nqS173.json"
CB="--cand k0_hm10=$V2,manifest=$G/k0_manifest.json,n_float=77,hm=1.0"
CB="$CB --cand jF77_hm10=$JF,manifest=$G/k0_manifest.json,n_float=77,hm=1.0"
for n in 26 51 128; do CB="$CB --cand jF$n=$JF,manifest=$G/kB_manifest.json,n_float=$n,hm=0.7"; done
CB="$CB --cand k26_v2=$V2,manifest=$M26"
CB="$CB --cand jF182=$JF,manifest=$G/kB_manifest.json,n_float=182,hm=0.7 --cand nqS182=mix:lo=$LO,hi=$HI,set=$O/nqS182.json"
mkdir -p $O/e2e/logs
for f in $O/src/model-layer-077.safetensors $O/corpora/bf16conf.npy $O/corpora/bf16conf.map.npz $O/nqS182.json $O/nqS173.json \
         $J/models/jF_all.pt $G/k0_manifest.json $G/kB_manifest.json $M26 \
         $O/teacher/reference-full-panel/logits/confirmation/confirmation-0063.safetensors; do test -e $f; done
hold() {  # TAG CANDS
  f=$O/kld_$1.tmp.sh
  { echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs $1 \$(date -u)\""
    echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 NQ_OUT=$O/e2e NQ_CORPUS_DIR=$O/corpora"
    echo "export NQ_TEACHER=bf16conf=$O/teacher/reference-full-panel/logits/confirmation"
    echo 'pids=()'
    echo 'for r in 0 1 2 3 4 5 6 7; do'
    echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup timeout 5300 $R run --corpora bf16conf --moe-chunk 16384 --tag $1 $2 > $O/e2e/logs/$1.r\$r.log 2>&1 &"
    echo '  pids+=($!)'
    echo 'done'
    echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
    echo "echo \"release GPUs $1 rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
  chmod +x $f
  flock /tmp/nestquant/33-search/gpu.lock $f
  NQ_OUT=$O/e2e $R merge --tag $1
}
case ${1:?A|B|AB} in
  A) hold passT34TA "$CA" ;;
  B) hold passT34TB "$CB" ;;
  AB) hold passT34TA "$CA"; hold passT34TB "$CB" ;;
  *) exit 2 ;;
esac
echo "run_teacher $1 done $(date -u)"
