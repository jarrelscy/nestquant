#!/bin/bash
# T33i joint-predictor KLD pass (coordinator 2026-09-30), protocol = T32 fp8dec contract (commit 3ce409d):
# uniform 77 floating slots (k0), arms vs k0_hm10 (v2, in-hold ref; harness deterministic), matched swaps/1k by hm pair
# (HMD / HMH, bracket v2 hm1.0 churn; pick offline from the fp8dec trace before the hold).
#   dec  passT33iJd  fp8dec-heldout,fp8dec-tb21,sm120tfk8 (T32 passT32KF corpora list + order, run_kld_F.sh @9c5e159),
#        chain=map, PRIVATE NQ_OUT; joint_map (jmap_fp8dec.json), all out of sample:
#          fp8dec-heldout -> jF_all (no fp8dec task in any joint training set: calib-fit + sm120tf only)
#          tb:<6 sm120tf tasks> and sm120tfk8's 6 tasks -> the T33j fold model that held the task out; other TB2.1
#          tasks (incl. cad-model) -> jF_all (never trained on)
#        REF defaults to T32's passT32KF k0_hm10 (same spec/env/corpora; harness bitwise-deterministic across passes,
#        checked KBd vs Kd): no in-hold ref arm; window lists asserted equal, tokkl linked.  REF= (empty) keeps own ref.
#   hum  passT33iJ   nq-tail + wikitext, 32 windows, chain=1: joint=jF_all (lowest priority; optional)
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; PD=/tmp/nestquant/src/predec; T=/home/coder/git/nestquant/threads/32-gbdt-sal
F=$P/fp8dec_t32/corpora
J=/tmp/nestquant/33-search/joint; R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; PY=/home/coder/git/glm52/.venv/bin/python
HMD=${HMD:-0.5,0.8}; HMH=${HMH:-0.7,1.0}
arms() {  # CHAIN JOINTSPEC HMLIST PREFIX
  local A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=$1,salstat=1,predictor=gbdt,gmode=sync,manifest=$O/k0_manifest.json,n_float=77
  local s="--cand k0_hm10=$A,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,grlo=0,grhi=256,hm=1.0" h
  [ -n "${5:-}" ] && s=""     # REF set: reuse T32's k0_hm10 (harness bitwise-deterministic across passes)
  for h in ${3//,/ }; do s="$s --cand $4_hm${h/./}=$A,$2,hm=$h"; done
  echo "$s"
}
mkdir -p $J/logs/kld $P/e2e_dec/logs
hold() {  # TAG CORPORA EXTRA_ENV LOGDIR MAXW CANDS
  f=$J/kld_$1.tmp.sh
  { echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs $1 \$(date -u)\""
    echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 $3"
    echo 'pids=()'
    echo 'for r in 0 1 2 3 4 5 6 7; do'
    echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup timeout 5340 $R run --corpora $2 $5 --moe-chunk 16384 --tag $1 $6 > $4/$1.r\$r.log 2>&1 &"
    echo '  pids+=($!)'
    echo 'done'
    echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
    echo "echo \"release GPUs $1 rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
  chmod +x $f
  flock /tmp/nestquant/33-search/gpu.lock $f
}
DC=fp8dec-heldout,fp8dec-tb21,sm120tfk8
REF=${REF-$P/e2e_dec/results/passT32KF}
dec() {  # TAG
  for c in ${DC//,/ }; do test -f $F/$c.map.npz && test -f $F/$c.npy; done
  if [ -n "$REF" ]; then test -f $REF/tokkl_k0_hm10_r7.npy; fi
  hold $1 $DC "NQ_OUT=$P/e2e_dec NQ_CORPUS_DIR=$F" $P/e2e_dec/logs "" "$(arms map joint_map=$J/jmap_fp8dec.json $HMD jd "$REF")"
  NQ_OUT=$P/e2e_dec $R merge --tag $1
  if [ -n "$REF" ]; then   # pair against T32's ref: identical window lists per rank required, then link its tokkl
    $PY - $REF $P/e2e_dec/results/$1 <<'PYEOF'
import glob, json, sys
a, b = sys.argv[1:]
fs = sorted(glob.glob(b + "/r*.json")); assert len(fs) == 8, fs
for f in fs:
    p, q = json.load(open(f)), json.load(open(a + "/" + f.split("/")[-1]))
    assert p["windows"] == q["windows"] and p["groups_per_window"] == q["groups_per_window"] and p["corpora"] == q["corpora"], f
    assert "k0_hm10" in q["results"], "ref arm missing in " + a
print("ref windows match", a)
PYEOF
    for f in $REF/tokkl_k0_hm10_r*.npy; do ln -sf $f $P/e2e_dec/results/$1/; done
  fi
  $PY $T/kld_report.py --results $P/e2e_dec/results/$1 --ref k0_hm10 $(for c in ${DC//,/ }; do echo --mask-map $c=$F/$c.map.npz; done) \
    --out $P/kld_$1.json
}
PH=${1:?dec|hum}
for m in jF_all jF1 jF2 jF3; do test -f $J/models/$m.pt; done
case $PH in
  dec) dec passT33iJd ;;
  hum) hold passT33iJ nq-tail,wikitext "" $J/logs/kld "--max-windows 32" "$(arms 1 joint=$J/models/jF_all.pt $HMH jA)"
       $R merge --tag passT33iJ
       $PY $T/kld_report.py --results /tmp/nestquant/18-e2e/results/passT33iJ --ref k0_hm10 --out $J/kld_passT33iJ.json ;;
  *) echo "bad phase $PH"; exit 2 ;;
esac
echo "kld_J $PH done $(date -u)"
