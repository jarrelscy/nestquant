source /home/coder/git/nestquant/threads/16-bit-allocation/env.sh
cd /home/coder/git/nestquant/threads/16-bit-allocation
for LE in "16 92" "16 165" "49 36" "49 92" "49 165" "66 36" "66 92" "66 165"; do
  set -- $LE
  $PY anchors.py $1 $2 2>&1 | grep -v Warn
  $PY run_alloc.py $1 $2 none 2>&1 | grep -v Warn
done
echo ALLDONE
