#!/bin/bash
# T25 driver watchdog: every 60 s, if the campaign driver (ROOT/driver.pid) is not alive, rerun resume.sh (which fixes
# whatever is missing and restarts the driver; running workers are adopted). Single instance (flock ROOT/watchdog.lock).
# Leaves the campaign alone while ROOT/STOPPED exists (written by `nq25_campaign.py stop`; a manual resume.sh clears it).
# > 5 restarts within an hour -> backs off 30 min and appends a watchdog alert to ROOT/ALERTS.jsonl.
# Started detached by resume.sh; manual: setsid nohup ./watchdog.sh >/dev/null 2>&1 </dev/null &
HERE=$(dirname "$(readlink -f "$0")")
PY=/home/coder/git/glm52/.venv/bin/python
ROOT=$($PY -c "import json;print(json.load(open('$HERE/campaign.json'))['root'])")
mkdir -p "$ROOT"
exec 9>"$ROOT/watchdog.lock"
flock -n 9 || { echo "watchdog already running"; exit 0; }
echo $$ > "$ROOT/watchdog.pid"
export TZ=Australia/Melbourne NQ25_WATCHDOG=1
LOG="$ROOT/watchdog.log"
restarts=()
alive() { $PY -c "import sys;sys.path.insert(0,'$HERE');import nq25_resume as R;sys.exit(0 if R.driver_running('$ROOT') else 1)" 2>/dev/null; }
echo "[$(date '+%a %d %b %H:%M:%S')] watchdog start pid $$" >> "$LOG"
while true; do
  if [ ! -e "$ROOT/STOPPED" ] && ! alive; then
    t=$(date +%s); restarts=($(for x in "${restarts[@]}"; do [ $((t - x)) -lt 3600 ] && echo $x; done) $t)
    if [ ${#restarts[@]} -gt 5 ]; then
      echo "{\"time\": \"$(date '+%a %d %b %H:%M %Z')\", \"kind\": \"watchdog\", \"layer\": null, \"msg\": \"driver died ${#restarts[@]}x within 1 h; backing off 30 min (see $LOG, $ROOT/driver.out)\"}" >> "$ROOT/ALERTS.jsonl"
      echo "[$(date '+%a %d %b %H:%M:%S')] too many restarts; sleeping 30 min" >> "$LOG"
      sleep 1800; restarts=(); continue
    fi
    echo "[$(date '+%a %d %b %H:%M:%S')] driver not running -> resume.sh" >> "$LOG"
    "$HERE/resume.sh" >> "$LOG" 2>&1
  fi
  sleep 60
done
