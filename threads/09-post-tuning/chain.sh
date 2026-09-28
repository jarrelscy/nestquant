#!/bin/bash
cd /home/coder/git/nestquant/threads/09-post-tuning
while pgrep -f grid.sh >/dev/null; do sleep 20; done
./followup.sh > followup.log 2>&1
