#!/bin/bash
# evaltf.sh NAME CKPT GPU : export TF window forecasts (satb + embedding), then sim rows on nq-algo's r163 setup
# (satb A=shB11.io.pf.r163 C=shB11.io.pf.x155rm8.r163 D=shB22.co.pf.x155rm8.r163; embedding shB6.io) via nq-algo tools/srvtap_run.py
N=$1; C=$2; G=${3:-0}; cd /data/Jarrel/nq-tfpred; PY=/data/Jarrel/nq-algo/venv/bin/python; L=/data/Jarrel/nq-tfpred/logs
[ -f /data/Jarrel/nq-algo/fc/embedding-drift-monitor.$N.npy ] || CUDA_VISIBLE_DEVICES=$G /data/Jarrel/coord/memjob.sh 4 $PY src/export_fc.py --ckpt $C --name $N --tasks satb,embedding-drift-monitor > $L/export_$N.log 2>&1 || exit 1
J=/data/Jarrel/nq-tfpred/nqalgo/tf.$N.txt; : > $J
for t in satb embedding-drift-monitor; do if [ $t = satb ]; then EV="shB11.io.pf.r163 shB11.io.pf.x155rm8.r163 shB22.co.pf.x155rm8.r163"; else EV="shB6.io"; fi
  for e in $EV; do for a in srvtap-win:${N}w-c1-H64-mla1 curwin-$N-a64-h256 srvtap-win:${N}w-c1-H256-mla1 curwin-$N-a0-h64; do echo "$a@$e $t" >> $J; done; done; done
/data/Jarrel/coord/memjob.sh 8 bash /data/Jarrel/nq-tfpred/nqalgo/rjsrv.sh $J 6 > $L/sim_$N.log 2>&1
