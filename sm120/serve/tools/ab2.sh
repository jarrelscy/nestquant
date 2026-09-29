#!/bin/bash
# ab2.sh "label ENV=.. ENV=.." ...  -> restart NQ serving per config, single-stream step bench, JSON lines in ab.jsonl
# (ab.sh + the hit-share / follower log lines and a coherence probe per config)
S=/data/Jarrel/nestquant/sm120/serve/serve_nq.sh;D=/data/Jarrel/nq-serve
for cfg in "$@"; do
  set -- $cfg; lab=$1; shift
  $S down >/dev/null 2>&1
  if ! env "$@" $S up > $D/ab_up.log 2>&1; then echo "{\"label\":\"$lab\",\"error\":\"boot failed\"}" >> $D/ab.jsonl; docker logs --tail 60 glm53-nestquant > $D/bootfail_$lab.log 2>&1; continue; fi
  python3 $D/coh.py "$lab" >> $D/coh.jsonl 2>&1
  python3 $D/nq_step.py warm 64 >/dev/null
  python3 $D/nq_step.py "$lab $*" ${NQ_BENCH_TOK:-1024} | tail -1 >> $D/ab.jsonl
  sleep 65; docker logs glm53-nestquant 2>&1 | grep -E "host loop|level-4 experts|hit share|follower" | sed "s/^/$lab /" >> $D/ab_stats.log
done
