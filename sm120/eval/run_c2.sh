#!/bin/bash
# C2 one-command eval: score an NQ artifact (+ the ARVQ row) against the FP8 reference (eval_fp8.py).
#   run_c2.sh ARTIFACT_DIR [TAG] [extra eval_fp8.py args, e.g. --n-layers 10 --windows id=4,wikitext=2,code=2]
# Env: NQ_REPACK (serving repack of ARTIFACT; default /home/jarrelscy/nq-p4rec/<artifact key>, ~366 GB, built if missing),
#      NQ_ARVQ, NQ_EVAL_OUT (default /data/Jarrel/nq-eval), NQ_VRAM_GB (default 80), LEASE_MIN (default 120; 0 = no lease),
#      NQ_EBATCH (default 64).
# The artifact key = sha256 over the per-layer manifest.json files; the repack dir carries it in artifact_stamp.json, and
# a stale or foreign repack is refused (a new artifact version needs only a repack: that's done here).
# Needs the whole box (4 GPUs, up to NQ_VRAM_GB each): takes the box lease non-blockingly, retrying every minute.
set -e
A=${1:?artifact dir};TAG=${2:-c2-$(date -u +%m%d-%H%M)};shift $(( $# >= 2 ? 2 : 1 ))
H=$(cd "$(dirname "$0")" && pwd);PY=/data/Jarrel/nqenv/bin/python
KEY=$(cd "$A/layers" && for d in $(ls -d L* | sort -V); do echo "$d $(sha256sum $d/manifest.json | cut -c1-64)"; done | sha256sum | cut -c1-16)
R=${NQ_REPACK:-/home/jarrelscy/nq-p4rec/$KEY}
if [ -f "$R/artifact_stamp.json" ]; then
  grep -q "\"key\": \"$KEY\"" "$R/artifact_stamp.json" || { echo "repack $R was built from another artifact version (want $KEY): $(cat $R/artifact_stamp.json)"; exit 2; }
elif [ -f "$R/rank0.json" ]; then echo "repack $R has no artifact_stamp.json: cannot tell which artifact it holds"; exit 2
else echo "repacking $A -> $R (key $KEY)"; $PY /data/Jarrel/nestquant/streaming/repack.py "$A" "$R" 4 3-77 2560000
  printf '{"artifact": "%s", "key": "%s", "time_utc": "%s"}\n' "$A" "$KEY" "$(date -u +%FT%TZ)" > "$R/artifact_stamp.json"; fi
LM=${LEASE_MIN:-120}
if [ "$LM" != 0 ]; then until /data/Jarrel/coord/boxlease.sh try nestquant $LM; do sleep 60; done; trap '/data/Jarrel/coord/boxlease.sh release nestquant' EXIT; fi
cd "$H";NQ_VRAM_GB=${NQ_VRAM_GB:-80} OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True \
  /data/Jarrel/nqenv/bin/torchrun --nproc_per_node 4 eval_fp8.py --repack "$R" --artifact "$A" \
  ${NQ_ARVQ:+--arvq $NQ_ARVQ} --out ${NQ_EVAL_OUT:-/data/Jarrel/nq-eval} --tag "$TAG" --ebatch ${NQ_EBATCH:-64} "$@"
