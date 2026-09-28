#!/bin/bash
# NestQuant restart insurance: mirror small/medium /tmp/nestquant state to flashblade S3.
# Capture outputs (19-capture*) are backed up per completed layer by thread 19, not here.
# GLM FP8 source (src/) is not backed up: re-download with hf download zai-org/GLM-5.3-FP8.
# Usage: fb_backup.sh [--loop]   (loop = every 30 min, single owner via flock)
export PATH=$HOME/.local/bin:$PATH AWS_PROFILE=flashblade AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
EP=https://fb.harrisonai.io
DST=s3://annalise-shared-prod/jarrel/nestquant
SRC=/tmp/nestquant
LOG=$SRC/fb_backup.log
DIRS="corpus 21-traces trace-survey 02-feedback-conflict 04-decode-kernel 12-reference-encoder 13-moe-layer-kernel 14-level4-floor 15-level4-decode 16-bit-allocation 17-level2-margin 18-e2e glm53-fp8-experts"
once() {
  for d in $DIRS; do
    [ -d $SRC/$d ] || continue
    aws s3 sync --only-show-errors --endpoint-url $EP $SRC/$d $DST/$d >>$LOG 2>&1 \
      && echo "$(date -u +%FT%TZ) ok $d" >>$LOG || echo "$(date -u +%FT%TZ) FAIL $d" >>$LOG
  done
  # corpus tokens are small: also keep a copy on the home volume
  mkdir -p $HOME/nestquant-corpus-backup && rsync -a --delete $SRC/corpus/ $HOME/nestquant-corpus-backup/ >>$LOG 2>&1
  date -u +%FT%TZ > $SRC/fb_backup.last
  aws s3 cp --only-show-errors --endpoint-url $EP $SRC/fb_backup.last $DST/fb_backup.last >>$LOG 2>&1
}
if [ "$1" = --loop ]; then
  exec 9>$SRC/fb_backup.lock; flock -n 9 || { echo "another backup owner running"; exit 0; }
  while true; do once; sleep 1800; done
else once; fi
