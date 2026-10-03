#!/bin/bash
# Gate for the clean-d tree (the serve path is D only), for a GPU window. Do NOT run while tb4 / prod owns the box.
#   sm120/serve/tools/gate_d.sh            boot this tree (NQ_REPO = this checkout, own build dir), gate, take it down
#   UP=0 sm120/serve/tools/gate_d.sh       gate the glm53-nestquant serve that is already up (must have NQ_DEV_MODE=1)
#   KEEP_UP=1                              leave the serve up afterwards
# Gates (all must pass; rows -> $OUT, summary on stdout, exit 1 on any FAIL):
#   1 chk_env   the container's NQ_* env == docker-compose.nq.yaml defaults (= D) and no removed knob present
#   2 coherence capital-of-France + 300-token story with no loops (tools/coh.py)
#   3 86K needle (gate_ho part 2: 100K-token target at depth 0.3): retrieved cold and on the LMCache hit
#   4 950K cold needle with phase 2 KV offload engaged ("KV offload of" in the log during it), then reset_prefix_cache
#     + same prompt = LMCache hit: retrieved, and the hit is >= 3x faster than cold
#   5 decode tok/s: 3 reps of dec_only (4 prompts each) vs the prod D rows (labels D, D2) in nq-io/ab/work.jsonl: mean
#     >= 92% of prod (prod per-run spread is ~77-95 tok/s)
#   + no Traceback / EngineDead in the serve log
set -u
T=$(cd "$(dirname "$0")/../../.." && pwd); AB=/data/Jarrel/nq-io/ab; PY=/data/Jarrel/nq-algo/venv/bin/python
OUT=${OUT:-$AB/gate_d.jsonl}; LAB=${LAB:-clean-d-$(git -C "$T" rev-parse --short HEAD)}; FAIL=()
say(){ echo "$(date -u +%FT%TZ) $*"; }
if [ "${UP:-1}" = 1 ]; then
  docker ps --format '{{.Ports}}' | grep -q ':8001->' && { say "port 8001 in use (tb4 / prod?): refusing to boot; UP=0 gates the serve that is up"; exit 1; }
  rm -f /dev/shm/nq_la_ctl /dev/shm/nq_pb_off /dev/shm/nq_pb_kv_off /dev/shm/nq_pf_off /dev/shm/nq_sr_ctl
  # same host-side env as switch.sh glm5.3-nq-jf; every NQ_* serve knob left to the D defaults
  export NQ_REPO=$T NQ_BUILD_DIR=${NQ_BUILD_DIR:-/data/Jarrel/nq-clean-build} NQ_DEV_MODE=1 \
    NQ_REPACK_DIR=${NQ_REPACK_DIR:-/home/jarrelscy/nq-p4rec/hf} NQ_LAYERS=3-77 NUM_SPEC=3 ARVQ_CAPTURE_SIZES='[1,2,3,4,5,6,8,16,32]' \
    NQ_MAXLEN=1048576 NQ_UTIL=0.92 NQ_MAX_NUM_SEQS=1 NQ_LMCACHE=1
  say "boot $T ($LAB)"; bash "$T/sm120/serve/serve_nq.sh" up || { say "boot FAILED"; docker logs --tail 80 glm53-nestquant; exit 1; }
fi
T0=$(date -u +%FT%TZ)
# 1 env
REMOVED='NQ_HOSTLOOP|NQ_SCHED|NQ_PREDICTOR|NQ_STREAM|NQ_LEADER|NQ_CAP_GBPS|NQ_TOK_PER_S|NQ_IO_MODE|NQ_IO_SPLIT_RANKS|NQ_FOLLOW_COALESCE|NQ_PB_PROTECT|NQ_JF_LEGACY|NQ_KVOFF_DRY|NQ_KVOFF_SYNC|NQ_PB_FREE_CAP|NQ_GBDT_[A-Z_]+|NQ_TF_[A-Z_]+'
python3 $AB/chk_env.py "$T/sm120/serve/docker-compose.nq.yaml" > /tmp/gate_d_env.log 2>&1 || FAIL+=(chk_env)
docker exec glm53-nestquant env | grep -E "^($REMOVED)=" && FAIL+=(removed_env_present)
grep "ENV CHECK" /tmp/gate_d_env.log
docker exec glm53-nestquant /opt/vllm/.venv/bin/python -c "import sys;sys.path[:0]=['/nq/streaming'];import scheduler_tap as TS;assert TS.TapScheduler.__mro__[1] is object" \
  || FAIL+=(tree_not_clean_d)                          # the mounted /nq is this cleaned tree (standalone tap scheduler)
docker logs glm53-nestquant 2>&1 | grep -E "NestQuant.*(predictor|nq-io|prefill-borrow: .*want)" | head -8
# 2 coherence
$PY "$T/sm120/serve/tools/coh.py" "$LAB" | tee /tmp/gate_d_coh.log; grep -q '"pass_": true' /tmp/gate_d_coh.log || FAIL+=(coherence)
# 3 + 4 needles (gate_ho: 950K cold, LMCache hit, then the 86K needle cold / hit)
N0=$(wc -l < "$OUT" 2>/dev/null || echo 0)
$PY $AB/gate_ho.py "$LAB" "$OUT" ${NEEDLE_TOKENS:-950000} || FAIL+=(gate_ho_crashed)
python3 - "$OUT" "$N0" <<'EOF' || FAIL+=(needles)
import json,sys
R=[json.loads(l) for l in open(sys.argv[1]).readlines()[int(sys.argv[2]):]]
nd={r['path']:r for r in R if r['kind']=='needle'};n1={r['path']:r for r in R if r['kind']=='needle100k'}
ok=True
for k in ('cold','lmcache_hit'):
    r=nd.get(k);print(f'950K {k}: tokens {r and r["prompt_tokens"]} retrieved {r and r["retrieved"]} {r and r["secs"]}s')
    ok&=bool(r and r['retrieved'])
if ok:
    sp=nd['cold']['secs']/max(nd['lmcache_hit']['secs'],1e-3);print(f'LMCache hit speedup {sp:.1f}x');ok&=sp>=3 and nd['lmcache_hit'].get('reset',False)
for k in ('cold','hit'):
    r=n1.get(k);print(f'86K {k}: tokens {r and r["prompt_tokens"]} retrieved {r and r["retrieved"]} match_cold {r and r["match_cold"]}')
    ok&=bool(r and r['retrieved'])
sys.exit(0 if ok else 1)
EOF
NOFF=$(docker logs --since "$T0" glm53-nestquant 2>&1 | grep -c "prefill-borrow: epoch .*KV offload of")
say "phase 2 KV offload engagements during the gate: $NOFF"; [ "$NOFF" -ge 1 ] || FAIL+=(kv_offload_not_engaged)
docker logs --since "$T0" glm53-nestquant 2>&1 | grep "prefill-borrow: epoch" | tail -6
# 5 decode tok/s vs prod D
for r in 1 2 3; do $PY $AB/dec_only.py "$LAB" "$OUT" $r > /dev/null || FAIL+=(dec_only_crashed); done
python3 - "$OUT" "$LAB" <<'EOF' || FAIL+=(decode_tps)
import json,sys,statistics as st
R=[json.loads(l) for l in open(sys.argv[1])];P=[json.loads(l) for l in open('/data/Jarrel/nq-io/ab/work.jsonl')]
x=[r['decode_tps'] for r in R if r.get('kind')=='decode' and r['label']==sys.argv[2] and r.get('decode_tps')]
b=[r['decode_tps'] for r in P if r.get('kind')=='decode' and r.get('label') in ('D','D2') and r.get('decode_tps')]
m,mb=st.mean(x),st.mean(b);print(f'decode tok/s {m:.1f} (n {len(x)}, {min(x):.1f}-{max(x):.1f}) vs prod D {mb:.1f} (n {len(b)}): {m/mb:.3f}')
sys.exit(0 if m>=0.92*mb else 1)
EOF
NERR=$(docker logs --since "$T0" glm53-nestquant 2>&1 | grep -cE "Traceback|EngineDead|RuntimeError")
[ "$NERR" = 0 ] || { FAIL+=(serve_errors); docker logs --since "$T0" glm53-nestquant 2>&1 | grep -E -A3 "Traceback|EngineDead|RuntimeError" | head -30; }
[ "${UP:-1}" = 1 ] && [ "${KEEP_UP:-0}" != 1 ] && bash "$T/sm120/serve/serve_nq.sh" down >/dev/null 2>&1
if [ ${#FAIL[@]} = 0 ]; then say "GATE D ($LAB): PASS"; else say "GATE D ($LAB): FAIL: ${FAIL[*]}"; exit 1; fi
