#!/bin/bash
# T32 k=0 KLD prerequisites after the /tmp wipe: NestQuant TP8 layer containers from HF (current = h512 L3-6) ->
# /tmp/nestquant/nq-encode-v1/L{L} symlinks; predecoded_H (h512 L3-6 nq2/nq4) from flashblade.
set -euo pipefail
export AWS_PROFILE=flashblade AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
~/.local/bin/aws --endpoint-url https://fb.harrisonai.io s3 sync --only-show-errors s3://annalise-shared-prod/jarrel/nestquant/18-e2e/predecoded_H /tmp/nestquant/18-e2e/predecoded_H &
for f in l4_complement.json l4_sweep_union.json defset.json l4sub.json defset_float0.json; do
  ~/.local/bin/aws --endpoint-url https://fb.harrisonai.io s3 cp --only-show-errors s3://annalise-shared-prod/jarrel/nestquant/18-e2e/$f /tmp/nestquant/18-e2e/$f; done
/home/coder/git/glm52/.venv/bin/hf download jarrelscy/GLM-5.3-NestQuant-2-4bit --revision 26c6d2126fb48844e7f0896a6d9eefedea1d7ba4 \
  --include "layers/*" --local-dir /tmp/nestquant/src/nq-hf --max-workers 16 > /dev/null 2>&1
for d in /tmp/nestquant/src/nq-hf/layers/L*; do ln -sfn $d /tmp/nestquant/nq-encode-v1/$(basename $d); done
wait
echo "fetch done $(date -u)"; du -sh /tmp/nestquant/src/nq-hf /tmp/nestquant/18-e2e/predecoded_H
