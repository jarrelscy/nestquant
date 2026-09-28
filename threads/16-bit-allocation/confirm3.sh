source /home/coder/git/nestquant/threads/16-bit-allocation/env.sh
cd /home/coder/git/nestquant/threads/16-bit-allocation

for LE in "66 36" "66 92" "66 165"; do
  set -- $LE
  $PY compose_pos.py $1 $2 short greedy,alternate,positional 2>&1 | grep -v Warn
  $PY evalw.py $1 $2 results/eval_L$1E$2.json EXL3_2 EXL3_4 uni_none gate=A4g_none,up=A4g_none,down=uni_none A4pm_none A2pm_none \
     gate=C1.75_greedy,up=C1.75_greedy,down=C2.5_uniform gate=C1.75_alternate,up=C1.75_alternate,down=C2.5_uniform gate=C1.75_positional,up=C1.75_positional,down=C2.5_uniform 2>&1 | grep -v Warn
done
echo ALLDONE2
