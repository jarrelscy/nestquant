#!/bin/bash
# mem_guard.sh: Inkling OOM guard. Every 0.3 s reads cgroup anon+shmem (unreclaimable). Above WARN_GB logs; above KILL_GB
# sends SIGTERM (then SIGKILL after 20 s) to the process groups whose leader pids are listed in PIDS (Inkling jobs only;
# nothing else is ever signalled). Register a job: echo <pid> >> $PIDS (its process group is signalled).
PIDS=/tmp/nq_inkling/guard.pids; LOG=/tmp/nq_inkling/logs/mem_guard.log
WARN_GB=${WARN_GB:-1700}; KILL_GB=${KILL_GB:-2050}
touch $PIDS
while true; do
  u=$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{printf "%d", s/1e9}' /sys/fs/cgroup/memory.stat)
  if [ "$u" -ge "$KILL_GB" ]; then
    echo "$(date -u +%FT%TZ) KILL unreclaimable ${u}GB >= ${KILL_GB}" >> $LOG
    for p in $(sort -u $PIDS); do
      g=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); [ -n "$g" ] && kill -TERM -- -$g && echo "  TERM pgid $g (pid $p)" >> $LOG
    done
    sleep 20
    for p in $(sort -u $PIDS); do
      g=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); [ -n "$g" ] && kill -KILL -- -$g && echo "  KILL pgid $g" >> $LOG
    done
  elif [ "$u" -ge "$WARN_GB" ]; then
    echo "$(date -u +%FT%TZ) WARN unreclaimable ${u}GB" >> $LOG
  fi
  sleep 0.3
done
