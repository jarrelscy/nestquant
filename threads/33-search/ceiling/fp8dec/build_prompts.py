"""T33l fp8dec: build the candidate decode-prompt pool from public agent trajectories.

For each trajectory, cut at assistant-turn boundaries at quantiles (default 0.15/0.5/0.85
of the assistant-turn index) and render prompt = system + tools + messages-before-that-turn
with the GLM-5.3 chat template (add_generation_prompt=True -> ends in "<|assistant|><think>",
thinking on, reasoning_effort default "max", clear_thinking default False = past reasoning kept).
Task-statement-only sources render turn 1. Split by task id (sha1 % 10 == 0 -> heldout);
TB test names -> split "tb21" only. Drops prompts > --max-len tokens.

Outputs (in --out dir):
  candidates.jsonl   metadata for every rendered candidate (no ids)
  pool.jsonl         selected pool, one JSON per prompt incl. "input_ids" list
  stats.json         per-source counts/turn/token stats + pool histogram
Usage: python build_prompts.py [--pool 2000] [--workers 6]
"""
import os
for _v in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pyarrow as _pa
_pa.set_cpu_count(1); _pa.set_io_thread_count(1)
import numpy as np
import argparse, collections, hashlib, json, os, random, statistics, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as C

TOK = "/tmp/nestquant/src/glm53-fp8"
OUT = "/tmp/nestquant/33-search/ceiling/fp8dec_src"
TB21_TEST = set("""cad-model embedding-drift-monitor fin-saccr-rwa formal-crypto freight-dispatch-shift ks-solver-cpp
lake-temp-glm layout-config-recreation2 photonic-waveguide-routing pretrain-shard-corruption react-lead-form
satb-audio-transcription sound-change-cascade takens-embedding-lean""".split())
QS = (0.15, 0.5, 0.85)
POS = ("early", "middle", "late")
# per-source trajectory caps (random sample, fixed seed) to bound render time
CAPS = {"r2egym_sft_traj": 1500, "deepswe_kimik2_traj": 1500, "swegym_oh_sft": 491, "nebius_sweagent": 1200,
        "swesmith_traj": 1200, "nebius_rebench_oh": 1500, "tb2_gpt5_traj": None, "tb2_sonnet45_traj": None,
        "tb2_glm47_traj": None, "tb2_kimik25_traj": None}
# pool mix (fraction of --pool); math/general capped at <=10%
MIX = {"deepswe_r2e": 0.30, "openhands": 0.18, "sweagent_smith": 0.14, "tb_traj": 0.22, "tb_t1": 0.04,
       "r2e_t1": 0.03, "swebv_t1": 0.02, "math500": 0.07}
GROUP = {"r2egym_sft_traj": "deepswe_r2e", "deepswe_kimik2_traj": "deepswe_r2e", "swegym_oh_sft": "openhands",
         "nebius_rebench_oh": "openhands", "nebius_sweagent": "sweagent_smith", "swesmith_traj": "sweagent_smith",
         "tb2_gpt5_traj": "tb_traj", "tb2_sonnet45_traj": "tb_traj", "tb2_glm47_traj": "tb_traj", "tb2_kimik25_traj": "tb_traj",
         "tb21_registry_t1": "tb_t1", "tb_harbor_gh_t1": "tb_t1", "r2egym_subset_t1": "r2e_t1", "swebv_t1": "swebv_t1", "math500": "math500"}
BUCKETS = [(0, 4096, "<=4k"), (4096, 8192, "4k-8k"), (8192, 16384, "8k-16k"), (16384, 30720, "16k-30k")]

_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK)
    return _tok


def render(msgs, tools, gen=True):
    s = tok().apply_chat_template(msgs, tools=tools or None, add_generation_prompt=gen, tokenize=False)
    return tok()(s, add_special_tokens=False)["input_ids"]


def split_of(task_id):
    name = task_id[3:] if task_id.startswith("tb:") else None
    if name in TB21_TEST:
        return "tb21"
    return "heldout" if int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 10 == 0 else "train"


def bucket(n):
    for lo, hi, b in BUCKETS:
        if lo < n <= hi or (lo == 0 and n <= hi):
            return b
    return ">30k"


def work(args):
    """Render cuts for one canonical record. Returns (meta_list, full_len)."""
    rec, full, max_len = args
    msgs, tools = rec["messages"], rec["tools"]
    aidx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    out, full_len = [], None
    try:
        if full and aidx:
            full_len = len(render(msgs, tools, gen=False))
        if not aidx:  # task-statement-only: turn 1
            cuts = [("t1", 0, len(msgs))]
        else:
            cuts, seen = [], set()
            for q, pos in zip(QS, POS):
                k = round(q * (len(aidx) - 1))
                if k not in seen:
                    seen.add(k)
                    cuts.append((pos, k, aidx[k]))
        for pos, k, i in cuts:
            if i == 0:
                continue
            ids = render(msgs[:i], tools, gen=True)
            if len(ids) > max_len:
                continue
            out.append(dict(source=rec["source"], group=GROUP[rec["source"]], task_id=rec["task_id"], traj_id=rec["traj_id"],
                            turn_index=k, n_turns=len(aidx), position=pos, prompt_len=len(ids),
                            split=split_of(rec["task_id"]), ids=np.asarray(ids, dtype=np.int32)))
    except Exception as e:
        print("render-fail", rec["source"], rec["task_id"], type(e).__name__, str(e)[:200], file=sys.stderr)
    return out, full_len, rec["source"], len(aidx), rec["task_id"]


def gen_records(swebv_excl, seed):
    rng = random.Random(seed)
    for s, it in C.TRAJ_SOURCES.items():
        if s not in CAPS:
            continue
        recs = list(it(limit=CAPS[s]) if s == "nebius_rebench_oh" else it(limit=None))
        if CAPS[s] and len(recs) > CAPS[s]:
            recs = rng.sample(recs, CAPS[s])
        for j, r in enumerate(recs):
            yield r, j < 250  # full-traj token stats on first 250 of the sample
    for s, it in C.T1_SOURCES.items():
        for r in it():
            if s == "swebv_t1" and r["task_id"] in swebv_excl:
                continue
            yield r, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=30720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    # SWE-bench Verified exclusion: same instance_id as any train-source trajectory
    import pyarrow.parquet as pq
    swebv = set(pq.read_table(f"{C.SRC}/swebv/data/test-00000-of-00001.parquet", columns=["instance_id"]).column(0).to_pylist())
    traj_ids = set()
    for f in [f"{C.SRC}/nebius_rebench_oh/trajectories.parquet"] + sorted(__import__("glob").glob(f"{C.SRC}/nebius_sweagent/data/*.parquet")) \
            + sorted(__import__("glob").glob(f"{C.SRC}/swesmith_traj/data/tool-*.parquet")):
        traj_ids |= set(pq.read_table(f, columns=["instance_id"]).column(0).to_pylist())
    import convert
    traj_ids |= set(convert.load_swegym_ps_map().values())  # SWE-Gym OH-SFT ids (mapped via problem statement)
    swebv_excl = swebv & traj_ids
    print(f"SWE-bench Verified instances overlapping trajectory instance_ids: {len(swebv_excl)}", flush=True)

    cands, stats = [], collections.defaultdict(lambda: {"trajs": 0, "tasks": set(), "turns": [], "full_len": []})
    with Pool(a.workers) as P:
        for out, full_len, src, nt, tid in P.imap_unordered(work, ((r, f, a.max_len) for r, f in gen_records(swebv_excl, a.seed)), chunksize=4):
            st = stats[src]
            st["trajs"] += 1
            st["tasks"].add(tid)
            st["turns"].append(nt)
            if full_len:
                st["full_len"].append(full_len)
            cands.extend(out)
            if sum(v["trajs"] for v in stats.values()) % 500 == 0:
                print("progress", {k: v["trajs"] for k, v in stats.items()}, len(cands), flush=True)
    print("candidates", len(cands), flush=True)

    # ---- pool selection: per group quota, spread over positions and tasks; tb21 split always kept whole
    rng = random.Random(a.seed)
    pool = [c for c in cands if c["split"] == "tb21"]
    rest = [c for c in cands if c["split"] != "tb21"]
    by = collections.defaultdict(list)
    for c in rest:
        by[c["group"]].append(c)
    for g, frac in MIX.items():
        L = by.get(g, [])
        rng.shuffle(L)
        # round-robin over (position) so early/middle/late are balanced, one prompt per traj per pass
        per_pos = collections.defaultdict(list)
        for c in L:
            per_pos[c["position"]].append(c)
        q, sel, keys = int(frac * a.pool), [], sorted(per_pos)
        while len(sel) < q and any(per_pos.values()):
            for k in keys:
                if per_pos[k] and len(sel) < q:
                    sel.append(per_pos[k].pop())
        pool.extend(sel)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if k != "ids"}) + "\n")
    with open(f"{a.out}/pool.jsonl", "w") as f:
        for i, c in enumerate(pool):
            f.write(json.dumps({"pool_idx": i, **{k: v for k, v in c.items() if k != "ids"}, "input_ids": c["ids"].tolist()}) + "\n")

    def hist(L):
        h = collections.Counter(bucket(c["prompt_len"]) for c in L)
        return {b: h.get(b, 0) for _, _, b in BUCKETS}
    S = {}
    for s, st in stats.items():
        t, fl = st["turns"], st["full_len"]
        cs = [c for c in cands if c["source"] == s]
        S[s] = dict(trajectories_used=st["trajs"], tasks_used=len(st["tasks"]),
                    mean_turns=round(statistics.mean(t), 1) if t else None, median_turns=statistics.median(t) if t else None,
                    full_traj_tokens_mean=round(statistics.mean(fl)) if fl else None,
                    full_traj_tokens_median=statistics.median(fl) if fl else None, full_traj_sample=len(fl),
                    full_traj_frac_over_30k=round(sum(x > 30720 for x in fl) / len(fl), 3) if fl else None,
                    candidates=len(cs), cand_prompt_len_mean=round(statistics.mean([c["prompt_len"] for c in cs])) if cs else None,
                    cand_hist=hist(cs), splits=dict(collections.Counter(c["split"] for c in cs)))
    P_ = dict(n=len(pool), by_group=dict(collections.Counter(c["group"] for c in pool)),
              by_split=dict(collections.Counter(c["split"] for c in pool)), by_position=dict(collections.Counter(c["position"] for c in pool)),
              hist=hist(pool), hist_by_group={g: hist([c for c in pool if c["group"] == g]) for g in MIX},
              prompt_tokens_total=sum(c["prompt_len"] for c in pool))
    json.dump(dict(sources=S, pool=P_, swebv_excluded=sorted(swebv_excl), tb21_test_names=sorted(TB21_TEST),
                   tb21_in_pool=sorted({c["task_id"] for c in pool if c["split"] == "tb21"})),
              open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(P_, indent=1))


if __name__ == "__main__":
    main()
