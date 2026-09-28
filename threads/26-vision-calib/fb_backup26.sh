#!/bin/bash
# T26 flashblade backup / restore (fb_backup.sh conventions).
#   fb_backup26.sh backup  <dir-under-/tmp/nestquant> [extra aws s3 sync args]   e.g. calib-mm | 19-capture-mm
#   fb_backup26.sh restore <dir-under-/tmp/nestquant>
#   fb_backup26.sh budget                     total bytes under s3://annalise-shared-prod/jarrel/ (must stay < 4.8 TB)
export PATH=$HOME/.local/bin:$PATH AWS_PROFILE=flashblade AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
EP=https://fb.harrisonai.io
DST=s3://annalise-shared-prod/jarrel/nestquant
SRC=/tmp/nestquant
set -eu
mode=$1
if [ $mode = budget ]; then
  aws s3 ls --endpoint-url $EP --recursive --summarize s3://annalise-shared-prod/jarrel/ | tail -2
  exit 0
fi
d=$2; shift 2
if [ $mode = backup ]; then
  aws s3 sync --only-show-errors --endpoint-url $EP "$@" $SRC/$d $DST/$d
  # verify: a dry-run re-sync with the same filters must have nothing left to upload (size/mtime compare)
  left=$(aws s3 sync --dryrun --endpoint-url $EP "$@" $SRC/$d $DST/$d | wc -l)
  n=$(aws s3 ls --endpoint-url $EP --recursive --summarize $DST/$d/ | tail -2 | tr '\n' ' ')
  echo "$(date -u +%FT%TZ) $d: remote $n; still to upload: $left"
  [ $left = 0 ]
elif [ $mode = restore ]; then
  mkdir -p $SRC/$d
  aws s3 sync --only-show-errors --endpoint-url $EP "$@" $DST/$d $SRC/$d
  echo "restored $DST/$d -> $SRC/$d"
fi
