#!/bin/bash
# nq-lalloc arm driver: boot worktree serve with the arm's NQ_NF_LAYERS + KLD hook, run 25 forced-decode contexts, score in background
O=/data/Jarrel/nq-lalloc;WT=/data/Jarrel/nq-lalloc-wt;HA=/home/jarrelscy/homeassistant;PY=/data/Jarrel/nqenv/bin/python
IMG=glm53-arvq-sm120:fixes12-mtp-buffer-rng-20260917
log(){ echo "$(date '+%Y-%m-%d %H:%M %Z') $*" | tee -a $O/PROGRESS.md; }
key(){ grep -oP 'VLLM_API_KEY=\K\S+' $HA/.env; }
for ARM in ${ARMS:-U R1 R1r R2 R2r S Sr U2}; do
  TAG=la_$ARM
  if grep -q "\"tag\": \"$TAG\", \"ctx\": 24" $O/scores.jsonl 2>/dev/null; then log "$ARM already scored, skip"; continue; fi
  F=$ARM; [ "$ARM" = U2 ] && F=U; NFL=""; [ "$F" != U ] && NFL=/nq/streaming/results/nf_layers/$F.json
  [ -n "$NFL" ] && [ ! -f "$WT/streaming/results/nf_layers/$F.json" ] && { log "arm $ARM: missing $F.json"; exit 1; }
  if [ -n "$SKIPBOOT" ]; then SKIPBOOT=; log "arm $ARM: boot already in progress, waiting for /v1/models"
    until curl -sf -H "Authorization: Bearer $(key)" localhost:8001/v1/models >/dev/null 2>&1; do sleep 15; done
  else
  log "arm $ARM boot (NQ_NF_LAYERS=${NFL:-unset})"
  ( cd $HA && NQ_REPO=$WT NQ_COMPOSE_EXTRA=$WT/sm120/serve/docker-compose.nq-lalloc.yaml NQ_NF_LAYERS=$NFL NQ_KLD_HOOK=1 NQ_IOSTATS=2 NQ_TAP_IO=0 \
      ./switch.sh glm5.3-nq-jf ) > $O/boot_$ARM.log 2>&1 || { log "arm $ARM BOOT FAILED (see boot_$ARM.log)"; exit 1; }
  fi
  docker logs glm53-nestquant 2>&1 | grep -E "nq-lalloc|floating_default|fixed set" | head -4 >> $O/boot_$ARM.log
  curl -s -H "Authorization: Bearer $(key)" -H 'Content-Type: application/json' localhost:8001/v1/completions \
    -d '{"model":"local","prompt":"The capital of France is","max_tokens":32,"temperature":0}' | $PY -c "import sys,json;print('warm:',json.load(sys.stdin)['choices'][0]['text'][:80])" >> $O/boot_$ARM.log
  log "arm $ARM up; running 25 contexts"
  TAG=$TAG $PY $O/run_arm.py > $O/run_$ARM.out 2>&1 || { log "arm $ARM RUN FAILED"; exit 1; }
  log "arm $ARM registry done; running 4 confirmation windows (run_dec.py $TAG 1)"
  $PY /data/Jarrel/nq-kld/run_dec.py $TAG 1 > $O/dec_$ARM.out 2>&1 || { log "arm $ARM CONFIRMATION RUN FAILED"; exit 1; }
  log "arm $ARM run done: mean tps $($PY -c "import json;r=[json.loads(l) for l in open('$O/$TAG.jsonl')];print(round(sum(x['tps'] for x in r)/len(r),1),len(r))")"
  ( /data/Jarrel/coord/memjob.sh 40 $PY $O/score_arm.py $TAG > $O/score_$ARM.out 2>&1 && \
    for w in 0 1 2 3; do /data/Jarrel/coord/memjob.sh 40 $PY /data/Jarrel/nq-kld/score_dec.py ${TAG}_w${w}_r0 $w >> $O/score_$ARM.out 2>&1; done && \
    docker run --rm --entrypoint bash -v /data/Jarrel/nq-serve/dbg:/dbg $IMG -c "rm -rf /dbg/kld/${TAG}_c* /dbg/kld/${TAG}_w*" && echo "scored+cleaned $ARM" >> $O/score_$ARM.out ) &
done
wait
log "driver: all arms done"
