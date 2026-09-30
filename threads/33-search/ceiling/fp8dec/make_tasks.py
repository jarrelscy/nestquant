"""T33l fp8dec: build the dec.py task list for run 1 + the corpus manifest.  PRIVATE: stays on this box.
  train   = mix150 (approved) + top-up (same group x position x length stratification, disjoint tasks) so that the
            natural-stop decode budget reaches ~300k real tokens
  heldout = heldout-split tasks (sha1(task_id) % 10 == 0), same stratification, scaled
  tb21    = the 14 TB2.1-name turn-1 prompts (test only)
One prompt per task; no task appears in two corpora.
  make_tasks.py [--topup 90] [--heldout 32] --out DIR"""
import argparse, collections, hashlib, json, os, random
ap = argparse.ArgumentParser()
SRC = "/tmp/nestquant/33-search/ceiling/fp8dec_src"
ap.add_argument("--pool", default=f"{SRC}/pool.jsonl")
ap.add_argument("--mix", default=f"{SRC}/mix150.json")
ap.add_argument("--topup", type=int, default=90)
ap.add_argument("--heldout", type=int, default=32)
ap.add_argument("--out", required=True)
a = ap.parse_args()
Q = {"deepswe_r2e": 45, "openhands": 25, "sweagent_smith": 20, "tb_traj": 40, "tb_t1": 8, "r2e_t1": 0, "swebv_t1": 4,
     "math500": 8}
B = ["<=4k", "4k-8k", "8k-16k", "16k-30k"]


def bk(n):
    return B[0] if n <= 4096 else B[1] if n <= 8192 else B[2] if n <= 16384 else B[3]


P = [json.loads(line) for line in open(a.pool)]
meta = [{k: v for k, v in p.items() if k != "input_ids"} for p in P]
TB21 = {"cad-model", "embedding-drift-monitor", "fin-saccr-rwa", "formal-crypto", "freight-dispatch-shift",
        "ks-solver-cpp", "lake-temp-glm", "layout-config-recreation2", "photonic-waveguide-routing",
        "pretrain-shard-corruption", "react-lead-form", "satb-audio-transcription", "sound-change-cascade",
        "takens-embedding-lean"}


def is_tb21(m):
    return any(m["task_id"].endswith(":" + n) or m["task_id"].endswith("/" + n) or m["task_id"] == n for n in TB21)


def pick(split, n_total, used, seed):
    rng = random.Random(seed)
    sc = n_total / 150.0
    q = {g: int(round(v * sc)) for g, v in Q.items()}
    d = n_total - sum(q.values()); q["deepswe_r2e"] += d
    sel = []
    for g, qq in q.items():
        L = [m for m in meta if m["group"] == g and m["split"] == split and m["task_id"] not in used and not is_tb21(m)]
        rng.shuffle(L)
        cells = collections.defaultdict(list)
        for m in L:
            cells[(m["position"], bk(m["prompt_len"]))].append(m)
        keys = sorted(cells, key=lambda k: (B.index(k[1]), k[0]))
        n = 0
        while n < qq and any(cells[k] for k in keys):
            for k in keys:
                while cells[k] and cells[k][-1]["task_id"] in used:
                    cells[k].pop()
                if cells[k] and n < qq:
                    m = cells[k].pop(); used.add(m["task_id"]); sel.append(m); n += 1
    return sel


mix = json.load(open(a.mix))
prim = [meta[i] for i in mix["pool_idx"]]
assert all(m["split"] == "train" and not is_tb21(m) for m in prim)
used = {m["task_id"] for m in prim}
top = pick("train", a.topup, used, 1)
held = pick("heldout", a.heldout, set(), 2)
tb = [m for m in meta if m["split"] == "tb21"]
assert len(tb) == 14 and all(is_tb21(m) for m in tb), len(tb)
tr_ids = {m["task_id"] for m in prim + top}
assert not tr_ids & {m["task_id"] for m in held + tb}
assert not any(is_tb21(m) for m in prim + top + held)
hrule = sum(int(hashlib.sha1(m["task_id"].encode()).hexdigest(), 16) % 10 == 0 for m in held)
print(f"heldout tasks satisfying sha1 % 10 == 0 on task_id: {hrule}/{len(held)}")
tasks = []
for corpus, tier, L in (("fp8dec-train", "mix150", prim), ("fp8dec-train", "topup", top),
                        ("fp8dec-heldout", "heldout", held), ("fp8dec-tb21", "tb21", tb)):
    for m in L:
        tasks.append(dict(id=f"{corpus}/{m['pool_idx']}", corpus=corpus, tier=tier, pool_idx=m["pool_idx"],
                          source=m["source"], group=m["group"], task_id=m["task_id"], traj_id=m["traj_id"],
                          position=m["position"], turn_index=m["turn_index"], prompt_len=m["prompt_len"],
                          prompt=P[m["pool_idx"]]["input_ids"]))
os.makedirs(a.out, exist_ok=True)
json.dump(tasks, open(f"{a.out}/tasks_run1.json", "w"))


def summ(L):
    return dict(n=len(L), prompt_tokens=sum(m["prompt_len"] for m in L),
                by_group=dict(collections.Counter(m["group"] for m in L)),
                by_position=dict(collections.Counter(m["position"] for m in L)),
                by_bucket=dict(collections.Counter(bk(m["prompt_len"]) for m in L)),
                by_source=dict(collections.Counter(m["source"] for m in L)))


man = dict(note="PRIVATE - never leaves this box. GLM-5.3 FP8 decode corpus (T33l dec.py, DSA indexer, KV-carry).",
           sources=json.load(open(f"{SRC}/sources.json")),
           split_rule="task-level: heldout = sha1(task_id) % 10 == 0 (build_prompts.py); tb21 = the 14 TB2.1 names "
                      "(turn-1 only, test only); train = rest. One prompt per task per corpus; corpora task-disjoint.",
           prompt_render="GLM-5.3 chat template, Reasoning Effort: Max, <tools> block, native tool calls, "
                         "trajectory cut before assistant turn at quantiles 0.15/0.5/0.85 (early/middle/late) or t1; "
                         "prompt ends '<|assistant|><think>'",
           pool=dict(path=f"{SRC}/pool.jsonl", n=len(P),
                     by_split=dict(collections.Counter(m["split"] for m in meta))),
           corpora={c: summ([m for t, m in zip(tasks, prim + top + held + tb) if t["corpus"] == c])
                    for c in ("fp8dec-train", "fp8dec-heldout", "fp8dec-tb21")},
           tiers={"mix150": summ(prim), "topup": summ(top), "heldout": summ(held), "tb21": summ(tb)},
           tasks=[{k: v for k, v in t.items() if k != "prompt"} for t in tasks])
json.dump(man, open(f"{a.out}/manifest_run1.json", "w"), indent=1)
print(json.dumps({"corpora": man["corpora"], "tiers": {k: (v["n"], v["prompt_tokens"]) for k, v in man["tiers"].items()}},
                 indent=1))
