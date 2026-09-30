#!/bin/bash
# smoke (3 dense layers + MTP, rank 0, heldout) then the full capture; run under the gpu.lock.
set -uo pipefail
H=/home/coder/git/nestquant/threads/33-search/draft
DRAFT_NL=3 DRAFT_NSTEP=2 DRAFT_CORPORA=glm52-heldout DRAFT_OUT=/tmp/nestquant/33-search/draft/private/smoke RANKS=0 TAG=smoke \
  bash $H/run_cap.sh
grep -q "done {" /tmp/nestquant/33-search/draft/logs/capsmoke.r0.log || { echo "smoke failed"; exit 1; }
TAG=full bash $H/run_cap.sh
