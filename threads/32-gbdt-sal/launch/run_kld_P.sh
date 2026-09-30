#!/bin/bash
# T32 k0 + pooled-fit allocations P64 / P48 (T33k, sum 5775) KLD pass (coordinator 2026-09-30), same protocol as
# passT32KB: arms k0_hm10 (ref), kP64_hm10, kP48_hm10 + band-attribution arms kP64lo / kP48lo (P slots on L3-40 only,
# n_float 77 on L41-77: dKLD(lo) = early-band share, full - lo = late-band share).  kB_manifest floating_default covers
# all 256 (P48 layers up to 160).  TWO gpu.lock holds (each < 90 min, the 5-arm human+decode total is ~100 min):
#   passT32KP   nq-tail + wikitext, 32 windows, WORLD 8, NQ_OUT default (18-e2e/results)
#   passT32KPd  sm120tfk8 (8 task chains x 3 all-decode windows), WORLD 8, PRIVATE NQ_OUT
# Then per-layer served-slot check vs each arm's nf_map (absent layers = 77) and kld_report vs k0_hm10.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; PD=/tmp/nestquant/src/predec; T=/home/coder/git/nestquant/threads/32-gbdt-sal
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; PY=/home/coder/git/glm52/.venv/bin/python
A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=1,salstat=1,predictor=gbdt,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256
KM="$A,manifest=$O/kB_manifest.json,n_float=77,hm=1.0"
CANDS="--cand k0_hm10=$A,manifest=$O/k0_manifest.json,n_float=77,hm=1.0"
for n in P64 P48 P64lo P48lo; do CANDS="$CANDS --cand k${n}_hm10=$KM,nf_map=$O/alloc_$n.json"; done
mkdir -p $P/e2e_dec/logs $O/logs/kP
hold() {  # TAG CORPORA EXTRA_ENV LOGDIR MAXW
  f=$O/kld_$1.tmp.sh
  { echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs $1 \$(date -u)\""
    echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 $3"
    echo 'pids=()'
    echo 'for r in 0 1 2 3 4 5 6 7; do'
    echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup $R run --corpora $2 $5 --moe-chunk 16384 --tag $1 $CANDS > $4/$1.r\$r.log 2>&1 &"
    echo '  pids+=($!)'
    echo 'done'
    echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
    echo "echo \"release GPUs $1 rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
  chmod +x $f
  flock /tmp/nestquant/33-search/gpu.lock $f
}
hold passT32KP nq-tail,wikitext "" $O/logs/kP "--max-windows 32"
$R merge --tag passT32KP
hold passT32KPd sm120tfk8 "NQ_OUT=$P/e2e_dec NQ_CORPUS_DIR=$P/corpora" $P/e2e_dec/logs ""
NQ_OUT=$P/e2e_dec $R merge --tag passT32KPd
$PY - <<PYEOF
import glob, json
M = {f"k{n}_hm10": {int(k): v for k, v in json.load(open(f"$O/alloc_{n}.json")).items()} for n in ("P64", "P48", "P64lo", "P48lo")}
for d in ("/tmp/nestquant/18-e2e/results/passT32KP", "$P/e2e_dec/results/passT32KPd"):
    bad = 0
    for f in sorted(glob.glob(d + "/r*.json")):
        p = json.load(open(f))
        for arm, B in M.items():
            for L, x in p["results"][arm]["extra"]["diag"].items():
                e = B.get(int(L), 77)
                if not (x["nf"] == x["float_max"] == x["float_min"] == e):
                    bad += 1; print("SLOT MISMATCH", f, arm, L, x.get("nf"), x.get("float_max"), x.get("float_min"), e)
    print(d, "slot check vs nf_maps:", "OK" if not bad else f"{bad} mismatches", {a: sum(B.get(L, 77) for L in range(3, 78)) for a, B in M.items()})
PYEOF
$PY $T/kld_report.py --results /tmp/nestquant/18-e2e/results/passT32KP --ref k0_hm10 --out $O/kld_passT32KP.json
$PY $T/kld_report.py --results $P/e2e_dec/results/passT32KPd --ref k0_hm10 --mask-map $P/corpora/sm120tfk8.map.npz --out $P/kld_passT32KPd.json
echo "kld_P done $(date -u)"
