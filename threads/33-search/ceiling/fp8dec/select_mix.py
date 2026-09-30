"""Pick a ~150-task decode mix from pool.jsonl (train split only; one prompt per task),
balanced over group quotas x turn position x prompt-length bucket. Writes mix150.json.
Decode budget: n_prompts x --decode (default 2048) tokens."""
import argparse, collections, json, random
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="/tmp/nestquant/33-search/ceiling/fp8dec_src/pool.jsonl")
ap.add_argument("--out", default="/tmp/nestquant/33-search/ceiling/fp8dec_src/mix150.json")
ap.add_argument("--decode", type=int, default=2048)
ap.add_argument("--split", default="train")
a = ap.parse_args()
Q = {"deepswe_r2e": 45, "openhands": 25, "sweagent_smith": 20, "tb_traj": 40, "tb_t1": 8, "r2e_t1": 0, "swebv_t1": 4, "math500": 8}
B = ["<=4k", "4k-8k", "8k-16k", "16k-30k"]
def bk(n): return B[0] if n <= 4096 else B[1] if n <= 8192 else B[2] if n <= 16384 else B[3]
rng = random.Random(0)
P = [json.loads(l) for l in open(a.pool)]
for p in P: p.pop("input_ids")
used, sel = set(), []
for g, q in Q.items():
    L = [p for p in P if p["group"] == g and p["split"] == a.split]
    rng.shuffle(L)
    cells = collections.defaultdict(list)
    for p in L: cells[(p["position"], bk(p["prompt_len"]))].append(p)
    keys = sorted(cells, key=lambda k: (B.index(k[1]), k[0]))
    n = 0
    while n < q and any(cells[k] for k in keys):
        for k in keys:
            while cells[k] and cells[k][-1]["task_id"] in used: cells[k].pop()
            if cells[k] and n < q:
                p = cells[k].pop(); used.add(p["task_id"]); sel.append(p); n += 1
H = collections.Counter((p["group"], p["position"], bk(p["prompt_len"])) for p in sel)
out = dict(n=len(sel), decode_tokens=len(sel) * a.decode, prompt_tokens=sum(p["prompt_len"] for p in sel),
           by_group=collections.Counter(p["group"] for p in sel), by_position=collections.Counter(p["position"] for p in sel),
           by_bucket=collections.Counter(bk(p["prompt_len"]) for p in sel),
           cells={"|".join(k): v for k, v in sorted(H.items())}, pool_idx=[p["pool_idx"] for p in sel])
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("pool_idx", "cells")}, indent=1))
