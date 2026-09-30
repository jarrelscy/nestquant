#!/bin/bash
# T32 k0 + allocation-B KLD pass (coordinator 2026-09-30): arms k0_hm10 (ref), kB_hm10, kB_hm07; kB = no fixed set,
# per-layer n_float from T33k calib allocation B (alloc_B.json, sum 5775) via Adapt nf_map; kB_manifest floating_default
# covers all 256 so B>77 layers start full.  ONE gpu.lock hold, two sub-passes (separate diag -> per-corpus swaps/1k):
#   passT32KB   nq-tail + wikitext, 32 windows, WORLD 8, NQ_OUT default (18-e2e/results)      (same as passT32Kk)
#   passT32KBd  sm120tfk8 (8 chains x 3 all-decode windows = sm120tfk + 2 more task chains), WORLD 8, PRIVATE NQ_OUT
# Then checks every layer's served floating-slot count == B (diag nf / float_max / float_min) and reports vs k0_hm10.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; PD=/tmp/nestquant/src/predec; T=/home/coder/git/nestquant/threads/32-gbdt-sal
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; PY=/home/coder/git/glm52/.venv/bin/python
until grep -q "kld_dec done" $O/logs/kld_dec.log; do sleep 30; done
A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=1,salstat=1,predictor=gbdt,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256
KB="$A,manifest=$O/kB_manifest.json,n_float=77,nf_map=$O/alloc_B.json"
CANDS="--cand k0_hm10=$A,manifest=$O/k0_manifest.json,n_float=77,hm=1.0 --cand kB_hm10=$KB,hm=1.0 --cand kB_hm07=$KB,hm=0.7"
mkdir -p $P/e2e_dec/logs $O/logs/kB
sub() {  # TAG CORPORA EXTRA_ENV LOGDIR MAXW
  echo "echo \"sub $1 start \$(date -u)\""
  echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 $3"
  echo 'pids=()'
  echo 'for r in 0 1 2 3 4 5 6 7; do'
  echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup $R run --corpora $2 $5 --moe-chunk 16384 --tag $1 $CANDS > $4/$1.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo 'done'
  echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
}
{ echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs passT32KB \$(date -u)\""
  sub passT32KB nq-tail,wikitext "" $O/logs/kB "--max-windows 32"
  sub passT32KBd sm120tfk8 "NQ_OUT=$P/e2e_dec NQ_CORPUS_DIR=$P/corpora" $P/e2e_dec/logs ""
  echo "echo \"release GPUs passT32KB rc=\$rc \$(date -u)\"; exit \$rc"; } > $O/kld_B.tmp.sh
chmod +x $O/kld_B.tmp.sh
flock /tmp/nestquant/33-search/gpu.lock $O/kld_B.tmp.sh
$R merge --tag passT32KB
NQ_OUT=$P/e2e_dec $R merge --tag passT32KBd
$PY - <<PYEOF
import glob, json
B = {int(k): v for k, v in json.load(open("$O/alloc_B.json")).items()}
for d in ("/tmp/nestquant/18-e2e/results/passT32KB", "$P/e2e_dec/results/passT32KBd"):
    bad = 0
    for f in sorted(glob.glob(d + "/r*.json")):
        p = json.load(open(f))
        for arm in ("kB_hm10", "kB_hm07"):
            dg = p["results"][arm]["extra"]["diag"]
            assert set(int(L) for L in dg) == set(B), (f, arm, "layer set")
            for L, x in dg.items():
                if not (x["nf"] == x["float_max"] == x["float_min"] == B[int(L)]):
                    bad += 1; print("SLOT MISMATCH", f, arm, L, x.get("nf"), x.get("float_max"), x.get("float_min"), B[int(L)])
    print(d, "slot check vs alloc_B:", "OK" if not bad else f"{bad} mismatches", "sum", sum(B.values()))
PYEOF
$PY $T/kld_report.py --results /tmp/nestquant/18-e2e/results/passT32KB --ref k0_hm10 --out $O/kld_passT32KB.json
$PY $T/kld_report.py --results $P/e2e_dec/results/passT32KBd --ref k0_hm10 --mask-map $P/corpora/sm120tfk8.map.npz --out $P/kld_passT32KBd.json
echo "kld_B done $(date -u)"
