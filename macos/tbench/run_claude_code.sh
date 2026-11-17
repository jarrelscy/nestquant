#!/bin/zsh
# Terminal-Bench 2.1 with the Claude Code harness against the local server (Anthropic Messages API).
# Usage: run_claude_code.sh <job-name> [harbor task filters...]
# Expects Harbor in $TBENCH (default ~/src/tbench) with the dataset under datasets/terminal-bench-2-1.
export PATH=$HOME/.local/bin:$PATH
JOB=$1; shift
cd ${TBENCH:-$HOME/src/tbench}; export PYTHONPATH=${0:A:h}
ANTHROPIC_API_KEY=local .venv/bin/harbor run -p datasets/terminal-bench-2-1/terminal-bench-2-1 -a local_claude_code:LocalClaudeCode \
  -m anthropic/glm-nestquant --base-url http://host.docker.internal:8000 \
  --ae CLAUDE_CODE_ATTRIBUTION_HEADER=0 --ae CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  --ae API_TIMEOUT_MS=7200000 --ae CLAUDE_STREAM_IDLE_TIMEOUT_MS=3600000 --ae CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=3600000 --ae CLAUDE_CODE_MAX_CONTEXT_TOKENS=110000 --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536 \
  --agent-timeout-multiplier 6 "$@" -n 1 -o jobs --job-name $JOB -y -q
echo "harbor exit $? $(date)"
