#!/bin/zsh
# Stop the NestQuant server started by ~/startnq.sh (or any dflash_server.py). Idempotent: does nothing if it is not
# running. Sends SIGTERM, and SIGKILL after NQ_STOP_TIMEOUT seconds (30) because a request in progress can delay a
# clean exit. Releases the GPU lock if the server held it.
setopt NO_UNSET

STATE=$HOME/.nestquant
PIDFILE=$STATE/server.pid
LOCK=/tmp/nestquant-model.lock
TIMEOUT=${NQ_STOP_TIMEOUT:-30}

servers=($(pgrep -f 'macos/dflash_server\.py'))
wrapper=$(cat "$PIDFILE" 2>/dev/null || true)
if (( ${#servers} == 0 )) && { [[ -z $wrapper ]] || ! kill -0 "$wrapper" 2>/dev/null; }; then
  echo "NestQuant server is not running."
  rm -f "$PIDFILE"
  exit 0
fi

if (( ${#servers} )); then
  echo "Stopping NestQuant server (pid ${servers[*]})."
  kill -TERM ${servers[@]} 2>/dev/null
  for _ in $(seq 1 "$TIMEOUT"); do
    pgrep -f 'macos/dflash_server\.py' >/dev/null || break
    sleep 1
  done
  if pgrep -f 'macos/dflash_server\.py' >/dev/null; then
    echo "Still running after ${TIMEOUT}s; forcing it to stop."
    pkill -KILL -f 'macos/dflash_server\.py'
    sleep 2
  fi
fi

# The start wrapper exits after the server and removes the lock; clean up if it did not.
[[ -n $wrapper ]] && kill -0 "$wrapper" 2>/dev/null && { sleep 2; kill -TERM "$wrapper" 2>/dev/null; }
if [[ -d $LOCK ]]; then
  owner_pid=$(sed -nE 's/.* pid ([0-9]+).*/\1/p' "$LOCK/owner" 2>/dev/null || true)
  if [[ -z $owner_pid ]] || ! kill -0 "$owner_pid" 2>/dev/null || grep -q '^server' "$LOCK/owner" 2>/dev/null; then
    rm -rf "$LOCK"
  fi
fi
rm -f "$PIDFILE"
echo "NestQuant server stopped."
