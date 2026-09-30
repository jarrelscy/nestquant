#!/bin/bash
# T33i joint-predictor KLD pass (coordinator 2026-09-30), protocol = T32 passT32KP/KPd (launch/run_kld_P.sh):
# uniform 77 floating slots (k0), arms vs k0_hm10 (v2, in-hold ref; harness deterministic), matched swaps/1k by hm pair.
#   passT33iJ   nq-tail + wikitext, 32 windows, WORLD 8: joint=jF_all (calib + all 6 sm120tf tasks) at hm 0.7 / 1.0
#   passT33iJd  sm120tfk8 (8 task chains x 3 decode windows), PRIVATE NQ_OUT: joint_map = T33j fold model whose
#               held-out tasks contain the chain's task (out of sample) at hm 0.5 / 0.8
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; PD=/tmp/nestquant/src/predec; T=/home/coder/git/nestquant/threads/32-gbdt-sal
J=/tmp/nestquant/33-search/joint; R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; PY=/home/coder/git/glm52/.venv/bin/python
A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=1,salstat=1,predictor=gbdt,gmode=sync,manifest=$O/k0_manifest.json,n_float=77
REF="--cand k0_hm10=$A,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,grlo=0,grhi=256,hm=1.0"
test -f $J/models/jF_all.pt
CH="$REF --cand jA_hm07=$A,joint=$J/models/jF_all.pt,hm=0.7 --cand jA_hm10=$A,joint=$J/models/jF_all.pt,hm=1.0"
CD="$REF --cand jd_hm05=$A,joint_map=$J/jmap_sm120tfk8.json,hm=0.5 --cand jd_hm08=$A,joint_map=$J/jmap_sm120tfk8.json,hm=0.8"
mkdir -p $J/logs/kld $P/e2e_dec/logs
hold() {  # TAG CORPORA EXTRA_ENV LOGDIR MAXW CANDS
  f=$J/kld_$1.tmp.sh
  { echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs $1 \$(date -u)\""
    echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 $3"
    echo 'pids=()'
    echo 'for r in 0 1 2 3 4 5 6 7; do'
    echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup timeout 5100 $R run --corpora $2 $5 --moe-chunk 16384 --tag $1 $6 > $4/$1.r\$r.log 2>&1 &"
    echo '  pids+=($!)'
    echo 'done'
    echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
    echo "echo \"release GPUs $1 rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
  chmod +x $f
  flock /tmp/nestquant/33-search/gpu.lock $f
}
hold passT33iJd sm120tfk8 "NQ_OUT=$P/e2e_dec NQ_CORPUS_DIR=$P/corpora" $P/e2e_dec/logs "" "$CD"
NQ_OUT=$P/e2e_dec $R merge --tag passT33iJd
$PY $T/kld_report.py --results $P/e2e_dec/results/passT33iJd --ref k0_hm10 --mask-map $P/corpora/sm120tfk8.map.npz --out $J/kld_passT33iJd.json
hold passT33iJ nq-tail,wikitext "" $J/logs/kld "--max-windows 32" "$CH"
$R merge --tag passT33iJ
$PY $T/kld_report.py --results /tmp/nestquant/18-e2e/results/passT33iJ --ref k0_hm10 --out $J/kld_passT33iJ.json
echo "kld_J done $(date -u)"
