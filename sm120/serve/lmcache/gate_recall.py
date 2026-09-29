"""Local-content integrity probe for a long cached prefix (complements the first-token KLD gate, which is
insensitive to a few corrupted chunks). The prefix is the same deterministic filler as gate_lossless.py, so the
exact text of every line is known: ask the model to reproduce Q lines spread uniformly over the context
(prompt ends with 'Line NNNNNNN [salt]: warehouse', greedy continuation must match the real line).
Run it on whatever KV currently backs the prefix (GPU original, or LMCache-restored after /reset_prefix_cache).
Usage: gate_recall.py <tokens> <salt> [n_probes] [tag]"""
import json, sys
from lmc_common import *
N, salt = int(sys.argv[1]), sys.argv[2]; Q = int(sys.argv[3]) if len(sys.argv) > 3 else 16; tag = sys.argv[4] if len(sys.argv) > 4 else 'run'
P = "Audit log follows.\n\n" + filler(N, salt); L = P.split("\n")[2:]; n = len(L)
ok = 0; rows = []
for j in range(Q):
    i = int((j + 0.5) * n / Q); line = L[i]; head = f"Line {i:07d} [{salt}]: warehouse"
    r = completion(P + f"\n\nRepeat line {i:07d} of the audit log exactly.\n{head}", max_tokens=40, logprobs=1)
    want = line[len(head):]; got = r['text'].split("\n")[0]
    hit = got.strip() == want.strip(); ok += hit; rows.append(dict(i=i, hit=hit, want=want, got=got, secs=round(r['secs'], 1)))
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps(dict(tag=tag, tokens=r['prompt_tokens'], recall=f"{ok}/{Q}")))
json.dump(rows, open(f'/data/Jarrel/nq-serve/lmc/recall_{salt}_{tag}.json', 'w'), indent=1)
