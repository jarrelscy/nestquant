#!/bin/bash
# T32 fp8dec decode KLD pass (coordinator 2026-09-30; primary decode corpus = FP8 GLM-5.3's own generations, T33l
# run 1).  Waits for $O/private/fp8dec_go (touched when T33l confirms run 1 complete; contains DEC_DIR), preps the
# corpus on CPU (fp8dec_prep.py: dec2t32 --world 8, K=2, decode masks), symlinks sm120tfk8 in as the secondary, then
# ONE gpu.lock hold, tag passT32KF, corpora fp8dec-heldout,fp8dec-tb21,sm120tfk8, arms k26_v2 / k0_hm10 / kP64_hm10,
# chain=map (one predictor sequence per task).  PRIVATE: NQ_OUT under private/.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; PD=/tmp/nestquant/src/predec; T=/home/coder/git/nestquant/threads/32-gbdt-sal
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; PY=/home/coder/git/glm52/.venv/bin/python
F=$P/fp8dec_t32; TAG=${TAG:-passT32KF}; CORP=fp8dec-heldout,fp8dec-tb21,sm120tfk8
until [ -s $P/fp8dec_go ]; do sleep 30; done
DEC=$(head -1 $P/fp8dec_go)
echo "prep $DEC $(date -u)"
rm -rf $F.tmp; nice -n 10 $PY $T/fp8dec_prep.py $DEC $(dirname $DEC)/tasks_run1.json $F.tmp --k 2 --world 8
rm -rf $F; mv $F.tmp $F
ln -sf $P/corpora/sm120tfk8.npy $F/corpora/sm120tfk8.npy; ln -sf $P/corpora/sm120tfk8.map.npz $F/corpora/sm120tfk8.map.npz
A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=map,salstat=1,predictor=gbdt,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256
M26=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json
CANDS="--cand k26_v2=$A,manifest=$M26 --cand k0_hm10=$A,manifest=$O/k0_manifest.json,n_float=77,hm=1.0"
CANDS="$CANDS --cand kP64_hm10=$A,manifest=$O/kB_manifest.json,n_float=77,hm=1.0,nf_map=$O/alloc_P64.json"
mkdir -p $P/e2e_dec/logs
{ echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs $TAG \$(date -u)\""
  echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_OUT=$P/e2e_dec NQ_CORPUS_DIR=$F/corpora"
  echo 'pids=()'
  echo 'for r in 0 1 2 3 4 5 6 7; do'
  echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup timeout 5300 $R run --corpora $CORP --moe-chunk 16384 --tag $TAG $CANDS > $P/e2e_dec/logs/$TAG.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo 'done'
  echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
  echo "echo \"release GPUs $TAG rc=\$rc \$(date -u)\"; exit \$rc"; } > $O/kld_$TAG.tmp.sh
chmod +x $O/kld_$TAG.tmp.sh
flock /tmp/nestquant/33-search/gpu.lock $O/kld_$TAG.tmp.sh
NQ_OUT=$P/e2e_dec $R merge --tag $TAG
MM=""; for c in ${CORP//,/ }; do MM="$MM --mask-map $c=$F/corpora/$c.map.npz"; done
$PY $T/kld_report.py --results $P/e2e_dec/results/$TAG --ref k0_hm10 $MM --out $P/kld_$TAG.json
$PY $T/kld_report.py --results $P/e2e_dec/results/$TAG --ref k26_v2 $MM --out $P/kld_${TAG}_vs26.json
echo "kld_F done $(date -u)"
