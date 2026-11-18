#!/bin/zsh
# Start the GLM-5.3-Flash NestQuant server on 0.0.0.0:8000 (OpenAI /v1/chat/completions and Anthropic /v1/messages).
# Idempotent: does nothing if the server is already up. Refuses to start when another model process holds the GPU
# lock, or when there is not enough reclaimable memory (the server peaks near 90 GiB on a 128 GB Mac; starting it
# without that headroom can freeze macOS).
#
# Environment overrides: NQ_PORT (8000), NQ_HOST (0.0.0.0), NQ_MAX_CONTEXT (131072), NQ_THINKING (high: effort for
# Anthropic thinking requests; off|low|high|max), NQ_THINKING_BUDGET (16384: reasoning tokens before the server closes
# the thinking and the model answers; 0 = no limit), NQ_MIN_AVAILABLE_GIB (95),
# NQ_MODEL_ALIAS (local: second model name), NQ_CACHE_SLOTS (4: conversations cached besides the current
# one, ~25 MB per 1K tokens each), NQ_TEMPERATURE (1.0), NQ_TOP_P (0.95), NQ_API_KEY (unset = no key), NQ_REPO (this checkout, else ~/src/nestquant).
setopt ERR_RETURN NO_UNSET PIPE_FAIL

PORT=${NQ_PORT:-8000}
HOST=${NQ_HOST:-0.0.0.0}
MAX_CONTEXT=${NQ_MAX_CONTEXT:-131072}
MIN_AVAILABLE_GIB=${NQ_MIN_AVAILABLE_GIB:-95}
TEMPERATURE=${NQ_TEMPERATURE:-1.0}
TOP_P=${NQ_TOP_P:-0.95}
THINKING=${NQ_THINKING:-high}
THINKING_BUDGET=${NQ_THINKING_BUDGET:-16384}
MODEL_ALIAS=${NQ_MODEL_ALIAS:-local}
CACHE_SLOTS=${NQ_CACHE_SLOTS:-4}
# The checkout this script lives in (macos/scripts/startnq.sh), else ~/src/nestquant (for a copy elsewhere).
repo_here=${0:A:h:h:h}
[[ -f $repo_here/macos/dflash_server.py ]] || repo_here=$HOME/src/nestquant
REPO=${NQ_REPO:-$repo_here}
STATE=$HOME/.nestquant
PIDFILE=$STATE/server.pid
LOG=$STATE/server.log
LOCK=/tmp/nestquant-model.lock

mkdir -p "$STATE"

health() { curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": "ok"'; }

# 1. Already running?
if health; then
  echo "NestQuant server is already up on $HOST:$PORT (pid $(cat "$PIDFILE" 2>/dev/null || echo unknown))."
  exit 0
fi
if [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "NestQuant server (pid $(cat "$PIDFILE")) is still loading; tail -f $LOG to follow it."
  exit 0
fi
rm -f "$PIDFILE"

# 2. GPU lock: only one ~86 GiB model process may run at a time.
if [[ -d $LOCK ]]; then
  owner_pid=$(sed -nE 's/.* pid ([0-9]+).*/\1/p' "$LOCK/owner" 2>/dev/null || true)
  if [[ -n $owner_pid ]] && kill -0 "$owner_pid" 2>/dev/null; then
    echo "Refusing to start: another model process holds the GPU lock: $(cat "$LOCK/owner")" >&2
    echo "Stop it first (~/stopnq.sh stops the server; benchmarks hold the lock while they run)." >&2
    exit 1
  fi
  echo "Removing a stale GPU lock ($(cat "$LOCK/owner" 2>/dev/null || echo 'no owner'))."
  rm -rf "$LOCK"
fi
if pgrep -f 'macos/(dflash_server|native_dflash|native_generate)\.py' >/dev/null; then
  echo "Refusing to start: a NestQuant model process is already running:" >&2
  pgrep -fl 'macos/(dflash_server|native_dflash|native_generate)\.py' >&2
  exit 1
fi

# 3. Memory: free + inactive + speculative + purgeable pages can be reclaimed without swapping.
page=$(sysctl -n hw.pagesize)
available_gib=$(vm_stat | awk -v page="$page" '
  /Pages free/ {s+=$3} /Pages inactive/ {s+=$3} /Pages speculative/ {s+=$3} /Pages purgeable/ {s+=$3}
  END {printf "%d", s*page/1073741824}')
total_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if (( available_gib < MIN_AVAILABLE_GIB )); then
  echo "Refusing to start: only ${available_gib} GiB of ${total_gib} GiB is available; the server needs ${MIN_AVAILABLE_GIB} GiB." >&2
  echo "Close large applications (browsers, Docker VMs, other models) and try again." >&2
  echo "Largest processes:" >&2
  { ps -axo rss=,comm= | sort -rn | head -5 | awk '{printf "  %6.1f GiB  %s\n", $1/1048576, $2}' >&2; } || true
  exit 1
fi
echo "Memory check passed: ${available_gib} GiB available of ${total_gib} GiB."

# 4. Start under the GPU lock (released when the server exits).
args=(--host "$HOST" --port "$PORT" --max-context "$MAX_CONTEXT" --temperature "$TEMPERATURE" --top-p "$TOP_P" --anthropic-thinking "$THINKING" --thinking-budget "$THINKING_BUDGET" --model-alias "$MODEL_ALIAS" --cache-slots "$CACHE_SLOTS" --long-reply-dir "$STATE/long-replies")
[[ -n ${NQ_API_KEY:-} ]] && args+=(--api-key "$NQ_API_KEY")
cd "$REPO"
nohup /bin/zsh -c '
  until mkdir '"$LOCK"' 2>/dev/null; do sleep 2; done
  echo "server$0 $(date) pid $$" > '"$LOCK"'/owner
  trap "rm -rf '"$LOCK"'" EXIT INT TERM
  .venv-macos/bin/python macos/dflash_server.py "$@"
' "$PORT" "${args[@]}" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "Starting NestQuant server (pid $(cat "$PIDFILE")); loading takes about 2 minutes. Log: $LOG"

# 5. Wait until it answers.
for _ in {1..90}; do
  if health; then
    echo "NestQuant server is up on $HOST:$PORT."
    echo "  OpenAI:    http://$HOST:$PORT/v1/chat/completions   (model glm-nestquant or $MODEL_ALIAS)"
    echo "  Anthropic: http://$HOST:$PORT/v1/messages"
    echo "  Default sampling: temperature $TEMPERATURE, top_p $TOP_P (requests can override); Anthropic thinking: $THINKING, thinking budget $THINKING_BUDGET tokens"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "The server exited during start-up. Last log lines:" >&2
    tail -20 "$LOG" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
  sleep 5
done
echo "The server is still loading after 7.5 minutes; check $LOG." >&2
exit 1
