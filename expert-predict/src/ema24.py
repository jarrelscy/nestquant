import sys, json; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
t = sys.argv[1]; d = load(t); ex = d['ex']; Cb = block_counts(ex)
for nm, R, hl, hm in (('ema512_R64', 64, 512, 0.0), ('ema256_R16_hm0.1', 16, 256, 0.1), ('ema128_R16_hm0.1', 16, 128, 0.1)):
    o = sim(ex, ema_scores(Cb, R, hl), R=R, hm=hm, cap_gbps=24); print(json.dumps(dict(task=t, pred=nm, cap=24, share=o['share'], gbps=o['gbps'])), flush=True)
