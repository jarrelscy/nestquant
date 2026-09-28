#!/bin/bash
# T25 campaign resume (B2): one idempotent command. ./resume.sh --check  (report only)  |  ./resume.sh  (fix + start)
# See RESUME.md. Safe to rerun at any time: every step skips what is already in place; a running driver is left alone.
set -euo pipefail
export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:/home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/cuda-compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/home/coder/git/glm52/.venv/bin:$HOME/.local/bin:$PATH
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p /tmp/nestquant/12-reference-encoder/bin
[ -e /tmp/nestquant/12-reference-encoder/bin/ninja ] || ln -sfn /home/coder/git/glm52/.venv/bin/ninja /tmp/nestquant/12-reference-encoder/bin/ninja
HERE=$(dirname "$(readlink -f "$0")")
rc=0; /home/coder/git/glm52/.venv/bin/python "$HERE/nq25_resume.py" "$@" || rc=$?
# not --check / pin, and not called by the watchdog itself: make sure the watchdog runs (single instance via flock)
if [ -z "${NQ25_WATCHDOG:-}" ] && [[ " $* " != *" --check "* ]] && [[ " $* " != *" pin "* ]]; then
  setsid nohup "$HERE/watchdog.sh" >/dev/null 2>&1 </dev/null &
fi
exit $rc
