"""Secret-safe summary of resolved Compose settings (stdin); no GPU/server access."""
import json
import sys

KEYS = (
    'NQ_PREDICTOR', 'NQ_JOINT_FIXED', 'NQ_SLOTS_PER_LAYER', 'PARALLEL',
    'NUM_SPEC', 'MAXLEN', 'MAX_NUM_SEQS', 'UTIL', 'NQ_SCHED', 'NQ_TAP_QREAL',
    'NQ_TAP_TODO_FIX', 'NQ_LMPF', 'NQ_LMPF_CG', 'NQ_LMPF_BORROW',
    'NQ_PREFILL_BORROW', 'NQ_PREFILL_SLOTS', 'NQ_PREFILL_KV_OFFLOAD',
    'NQ_PF_BLOCK', 'NQ_DEC_ASYNC', 'NQ_DEC_BLOCK_FIRSTN',
    'NQ_FOLLOW_COALESCE', 'ENABLE_LMCACHE', 'LMCACHE_MAX_LOCAL_CPU_SIZE',
    'NQ_IO_MODE', 'NQ_KLD_HOOK',
)
if __name__ == '__main__':
    env = json.load(sys.stdin)['services']['glm53-nestquant']['environment']
    print('Resolved NestQuant serving settings (explicit overrides are preserved):')
    for key in KEYS:
        print(f'  {key}={env.get(key, "<unset>")}')
    if str(env.get('NQ_JOINT_FIXED')) != '1' or str(env.get('NQ_SLOTS_PER_LAYER')) != '56':
        print('NOTE: expert allocation differs from the latest TB4 26-fixed/51-floating configuration.')
