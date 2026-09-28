#!/bin/bash
# SSD read-path sweep: io_uring + O_DIRECT -> pinned bounce -> H2D. Writes JSON lines to $OUT.
B=${SSDBENCH:-/data/Jarrel/nq-build/ssdbench}
R=${R0:-/home/jarrelscy/nq-ssd/t8g.bin}     # nvme0 (Gen5 x4, /)
D=${R1:-/data/Jarrel/nq-gds/t8g.bin}        # nvme1 (Gen4 x4, /data)
OUT=${OUT:-results/ssd_sweep.jsonl}; S=${SECS:-4}
mkdir -p $(dirname $OUT)
run(){ echo -n "{\"tag\":\"$1\"," >> $OUT; shift; $B "$@" --secs $S | sed 's/^{//' >> $OUT; tail -1 $OUT; }
for rec in 1208320 2420736 4841472; do for qd in 1 2 4 8 16 32; do run nvme0 --files $R --rec $rec --qd $qd --gpus 0 --h2d 1; done; done
for qd in 1 2 4 8 16; do run nvme0_noh2d --files $R --rec 2420736 --qd $qd --gpus 0 --h2d 0; done
for qd in 1 2 4 8 16; do run nvme1 --files $D --rec 2420736 --qd $qd --gpus 0 --h2d 1; done
for qd in 1 2 4 8; do run nvme0_4gpu --files $R --rec 2420736 --qd $qd --gpus 0,1,2,3 --h2d 1; done
for qd in 1 2 4 8; do run stripe01_4gpu --files $R,$D --rec 2420736 --qd $qd --gpus 0,1,2,3 --h2d 1; done
for qd in 2 4 8; do run stripe01_1gpu --files $R,$D --rec 2420736 --qd $qd --gpus 0 --h2d 1; done
