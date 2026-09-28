#!/bin/bash
# After shard 0's stage 1 completes: three stage-1 lanes (shards 1..13) + a third stage-2 worker on GPU 7.
ROOT=/tmp/nestquant/19-capture; D=/home/coder/git/nestquant/threads/19-full-capture/driver.sh
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$ROOT/shards/s00/state/progress.json'))['next_layer']>77 else 1)" 2>/dev/null; do sleep 60; done
echo "$(date) shard 0 stage 1 complete; launching lanes"
nohup $D stage1 2 1 4 7 10 13 > $ROOT/logs/lane2.log 2>&1 &
nohup $D stage1 3 2 5 8 11 > $ROOT/logs/lane3.log 2>&1 &
nohup $D stage1 7 3 6 9 12 > $ROOT/logs/lane7b.log 2>&1 &
nohup $D stage2 7 > $ROOT/logs/stage2_g7a.log 2>&1 &
wait
