#!/bin/bash
# T35 b175 release finalize: gates -> stamp -> upload bins -> upload rank*.json + artifact_stamp.json LAST -> gate (c).
set -uo pipefail
R=/tmp/nestquant/35-nq15/release; S=$R/stage; REPO=jarrelscy/GLM-5.3-NestQuant-1.75-4bit; T=/tmp/nestquant/35-nq15
PY=/home/coder/git/glm52/.venv/bin/python; export HF_XET_HIGH_PERFORMANCE=1
log(){ echo "$(date -u +%FT%TZ) $*"; }
die(){ log "STOP: $*"; exit 2; }
until [ -e $R/done/ALL_DONE ]; do sleep 60; done; log "ALL_DONE"
# gate (b) on the last layers (L3-6 Had512, plus late layers)
( cd $T/gate_b && nice -n 10 $PY gate_b.py 3:0:0 4:85:1 5:170:2 6:255:3 58:3:0 75:77:1 77:200:2 3:131:3 > run3.log 2>&1 )
grep -q "^GATE_B PASS" $T/gate_b/run3.log || die "gate b run3"; log "gate b run3 PASS"
# gate (a) L3 (v1 old/new/shipped)
until grep -q DONE $T/gate_a2/shipped.L3.log; do sleep 30; done
( cd $T/gate_a2 && $PY cmp_L.py 3 > cmp_L3.log 2>&1 ); grep -q "GATE_A L3 PASS" $T/gate_a2/cmp_L3.log || die "gate a L3"; log "gate a L3 PASS"
nice -n 10 $PY $R/recheck.py $R/repo > $R/logs/recheck_repo.log 2>&1; grep -q "RECHECK .* PASS" $R/logs/recheck_repo.log || die "recheck repo"
$PY $T/wt_main/sm120/serve/tools/verify_repack.py $R/repo > $R/logs/verify_repo.log 2>&1; grep -q "MISSING none" $R/logs/verify_repo.log || die "verify_repack repo"
log "local recheck + verify_repack PASS"
# artifact stamp (run_c2.sh key over the source manifests)
KEY=$(cd $T/enc_b175 && for d in $(ls -d L[0-9]* | sort -V); do echo "$d $(sha256sum $d/manifest.json | cut -c1-64)"; done | sha256sum | cut -c1-16)
n=$(cd $T/enc_b175 && ls -d L[0-9]* | wc -l); [ "$n" = 75 ] || die "manifest dirs $n"
printf '{"artifact": "%s", "key": "%s", "time_utc": "%s", "note": "%s"}\n' "nq-glm53-b175" "$KEY" "$(date -u +%FT%TZ)" \
  "b175: base K 1.75 (pattern 0xEEEE, code 1) + residual 2.25/2.25/2.5625 (down code 9), nq-res-v2; L3-6 in_had_down 512" > $R/repo/artifact_stamp.json
log "stamp $(cat $R/repo/artifact_stamp.json)"
# wait for the res uploader to finish its last batch
while pgrep -f "uploade[r].sh" >/dev/null; do sleep 30; done
up(){ nice -n 10 hf upload-large-folder $REPO $S --repo-type model --num-workers 6 --no-bars "$@" >> $R/logs/upload_final.log 2>&1; }
for r in 0 1 2 3; do [ -e $S/rank$r.bin ] || ln $R/repo/rank$r.bin $S/rank$r.bin; done
t0=$(date +%s); log "bins upload start"
up --include 'res/*' --include 'README.md' --include 'NQ_RES_V2.md' --include 'rank*.bin' --exclude 'rank*.json' --exclude 'artifact_stamp.json' || die "bin upload rc"
t1=$(date +%s); log "bins upload done in $((t1-t0))s ($((4*54814310400/(t1-t0)/1000000)) MB/s)"
for r in 0 1 2 3; do [ -e $S/rank$r.json ] || cp $R/repo/rank$r.json $S/rank$r.json; done; cp $R/repo/artifact_stamp.json $S/artifact_stamp.json
up --include 'res/*' --include 'README.md' --include 'NQ_RES_V2.md' --include 'rank*.bin' --include 'rank*.json' --include 'artifact_stamp.json' || die "json upload rc"
log "rank*.json + artifact_stamp.json uploaded: REPO COMPLETE"
$PY - <<'P' > $R/logs/hf_listing.log 2>&1
from huggingface_hub import HfApi
fs = [f for f in HfApi().list_repo_tree('jarrelscy/GLM-5.3-NestQuant-1.75-4bit', recursive=True) if hasattr(f, 'size')]
names = sorted(f.path for f in fs); print(len(names), sum(f.size for f in fs)); print([n for n in names if not n.startswith('res/')])
P
log "hf listing: $(head -2 $R/logs/hf_listing.log | tr '\n' ' ')"
# gate (c): fresh download with the start.sh include set (repeated --include: see start.sh note), verify_repack + recheck vs markers + byte-equal json/stamp
G=$R/gate_c; mkdir -p $G; t0=$(date +%s)
nice -n 10 hf download $REPO --repo-type model --include 'rank*.json' --include 'rank*.bin' --include 'res/*' --include 'artifact_stamp.json' --local-dir $G > $R/logs/gate_c_dl.log 2>&1 || die "gate c download"
t1=$(date +%s); log "gate c download $((t1-t0))s"
$PY $T/wt_main/sm120/serve/tools/verify_repack.py $G > $R/logs/gate_c_verify.log 2>&1
nice -n 10 $PY $R/recheck.py $G $R/repo > $R/logs/gate_c_recheck.log 2>&1
grep -q "MISSING none" $R/logs/gate_c_verify.log && grep -q "RECHECK .* PASS" $R/logs/gate_c_recheck.log && log "GATE_C PASS" || log "GATE_C FAIL"
