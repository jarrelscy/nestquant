"""Decontamination for the converted corpus.
1. Docs with any piece in orbit windows >= 28784 (thread 18's nq-tail eval) are excluded; docs in orbit windows
   28656..28783 (thread 19's held-out val) are forced into our val split.
2. 13-gram token overlap against thread 18's eval texts (vllm-docs heldout-xl, heldout, wikitext, github) and
   GPQA-diamond questions: docs with >= 5% of their 13-grams in an eval set are excluded.
Writes /tmp/nestquant/21-traces/decontam.json {exclude: {source_id: reason}, force_val: [source_id]}."""
import csv, json, sys
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

N = 13
EV = "/tmp/nestquant/18-e2e/evalsets"
tok = G.tokenizer()


def grams(ids):
    a = np.asarray(ids, np.int64)
    if len(a) < N:
        return np.zeros(0, np.int64)
    h = np.zeros(len(a) - N + 1, np.int64)
    for k in range(N):
        h = h * 1000003 + a[k:len(a) - N + 1 + k]
    return h


evals = {}
for name, f in [("vllm-docs", "glm52-heldout-xl.txt"), ("heldout", "glm52-heldout.txt"),
                ("wikitext", "glm52-neutral-wikitext.txt"), ("github", "glm52-neutral-github.txt")]:
    evals[name] = set(grams(tok.encode(open(f"{EV}/{f}").read(), add_special_tokens=False)).tolist())
gq = []
for r in csv.DictReader(open("/home/coder/git/glm52/gpqa/gpqa_diamond.csv")):
    q = r.get("Question") or r.get("question") or ""
    gq.extend(grams(tok.encode(q, add_special_tokens=False)).tolist())
evals["gpqa"] = set(gq)
print({k: len(v) for k, v in evals.items()})

TAIL, VAL0 = 28784, 28656
exclude, force_val = {}, []
for l in open("/tmp/nestquant/21-traces/orig_docs.jsonl"):
    d = json.loads(l)
    if max(d["windows"]) >= TAIL:
        exclude[d["source_id"]] = "nq-tail"; continue
    if max(d["windows"]) >= VAL0:
        force_val.append(d["source_id"])
    g = grams(d["ids"])
    if len(g) == 0:
        continue
    for k, s in evals.items():
        frac = np.mean([x in s for x in g.tolist()])
        if frac >= 0.05:
            exclude[d["source_id"]] = f"{k}:{frac:.2f}"; break
json.dump(dict(exclude=exclude, force_val=force_val), open("/tmp/nestquant/21-traces/decontam.json", "w"))
import collections
print("exclude", collections.Counter(v.split(":")[0] for v in exclude.values()), "force_val", len(force_val))
