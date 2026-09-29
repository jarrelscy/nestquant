#!/bin/bash
# ab.sh "label ENV=.. ENV=.." ...  -> restart NQ serving per config, single-stream step bench, JSON lines in ab.jsonl
S=/data/Jarrel/nestquant/sm120/serve/serve_nq.sh
for cfg in "$@"; do
  set -- $cfg; lab=$1; shift
  $S down >/dev/null 2>&1
  if ! env "$@" $S up > /data/Jarrel/nq-serve/ab_up.log 2>&1; then echo "{\"label\":\"$lab\",\"error\":\"boot failed\"}" >> /data/Jarrel/nq-serve/ab.jsonl; continue; fi
  python3 /tmp/nq_step.py warm 64 >/dev/null
  python3 /tmp/nq_step.py "$lab $*" | tail -1 >> /data/Jarrel/nq-serve/ab.jsonl
  sleep 65; docker logs glm53-nestquant 2>&1 | grep -E "host loop|level-4 experts" | sed "s/^/$lab /" >> /data/Jarrel/nq-serve/ab_stats.log
done
