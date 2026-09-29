"""Gate 1 (lossless restore), statistical form. NQ decode/prefill is not bitwise repeatable (K-split atomics + time-varying
expert levels), so exact temp-0 token match is impossible even for two GPU-prefix-cache hits on the SAME KV bytes.
Instead: one long shared prefix P, then Q short probe suffixes (P+q_i, max_tokens=1, top-20 logprobs).
  A: probes on the ORIGINAL KV (GPU prefix hit after the cold store)       B: same again (noise floor, same KV bytes)
  reset GPU prefix cache -> R: probes on the LMCache-RESTORED KV (first probe restores, the rest hit the restored pages)
  R2: again (noise floor on restored KV)
Lossless <=> KLD(A,R) is statistically indistinguishable from KLD(A,B) and argmax agreement matches.
A corrupted restore shows up as KLD >> floor (checked by the negative control: probes against a DIFFERENT prefix).
Usage: gate_lossless.py <tokens> <salt> [n_probes]"""
import json, statistics as S, sys, time
from lmc_common import *
N, salt = int(sys.argv[1]), sys.argv[2]; Q = int(sys.argv[3]) if len(sys.argv) > 3 else 24
P = "Audit log follows.\n\n" + filler(N, salt)
qs = [f"\n\nQuestion {i}: how many crate transfers did Line {(i * 7919) % max(1, N // 40):07d} [{salt}] log, and in which shift? Answer: Line" for i in range(Q)]
def probe(p): r = completion(p, max_tokens=1, logprobs=20); return r
def rnd(tag, prefix=P):
    m0 = metrics(); out = [probe(prefix + q) for q in qs]; m1 = metrics()
    print(f"{tag}: {sum(r['secs'] for r in out):.1f}s total, first {out[0]['secs']:.1f}s, gpu_hit {m1['prefix_cache_hits_total']-m0['prefix_cache_hits_total']:.0f} ext_hit {m1['external_prefix_cache_hits_total']-m0['external_prefix_cache_hits_total']:.0f}", flush=True)
    return out
t = time.time(); cold = probe(P + "\n\nSummary:"); print(f"cold store {cold['prompt_tokens']} tok {cold['secs']:.1f}s", flush=True)
A = rnd('A(orig)'); B = rnd('B(orig)')
reset_gpu_prefix(); time.sleep(2)
R = rnd('R(restored)'); R2 = rnd('R2(restored)')
NC = rnd('neg-ctrl(other prefix)', "Audit log follows.\n\n" + filler(min(N, 4000), salt + 'X'))
def st(x, y):
    k = [kld_top(a['top'][0], b['top'][0]) for a, b in zip(x, y)]
    return dict(kld_mean=round(S.mean(k), 5), kld_median=round(S.median(k), 5), kld_max=round(max(k), 5), argmax_agree=f"{sum(a['toks'][0]==b['toks'][0] for a,b in zip(x,y))}/{len(x)}")
res = dict(tokens=cold['prompt_tokens'], probes=Q, AB_noise_orig=st(A, B), RR2_noise_restored=st(R, R2), AR_orig_vs_restored=st(A, R), BR2=st(B, R2), A_vs_negctrl=st(A, NC))
json.dump(dict(res=res, A=A, B=B, R=R, R2=R2), open(f'/data/Jarrel/nq-serve/lmc/lossless_{salt}.json', 'w'))
print(json.dumps(res, indent=1))
