"""13-gram dedup of trace docs against eval sets and the converted corpus.

Eval sets (drop a trace doc if >= 5% of its 13-grams hit): GPQA-diamond questions + answers, Terminal-Bench 4.0
instructions (tb_main/tasks, the streaming tb4 eval), ICH-eval model outputs (glm52/ich_eval gen files), thread-18
eval texts (vllm-docs heldout-xl, heldout, wikitext, github).
GPQA traces (vs gpqa) and TB2.1 trajectories (vs tb4) are exempt by user/lead decision; the overlap is reported.
Corpus (drop if >= 50% of its 13-grams are already in conv_docs, i.e. a near-copy of a corpus doc).
Writes /tmp/nestquant/21-traces/trace_dedup.json {source_id: reason} and prints per-source drop counts.
"""
import csv, glob, json, sys, collections
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

N = 13
Wd = "/tmp/nestquant/21-traces"
tok = G.tokenizer()


def grams(ids):
    a = np.asarray(ids, np.int64)
    if len(a) < N:
        return np.zeros(0, np.int64)
    h = np.zeros(len(a) - N + 1, np.int64)
    for k in range(N):
        h = h * 1000003 + a[k:len(a) - N + 1 + k]
    return h


def enc(t):
    return tok.encode(t, add_special_tokens=False)


def eval_hashes():
    ev = {}
    g = []
    for r in csv.DictReader(open("/home/coder/git/glm52/gpqa/gpqa_diamond.csv")):
        for k in ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Explanation"):
            if r.get(k):
                g.append(grams(enc(r[k])))
    ev["gpqa"] = np.unique(np.concatenate(g))
    g = [grams(enc(open(f).read())) for f in glob.glob(f"{Wd}/tb_main/tasks/*/instruction.md")]
    ev["tb4"] = np.unique(np.concatenate(g))
    g = []
    for f in glob.glob("/home/coder/git/glm52/ich_eval/ich_*.jsonl"):
        if "jev" in f:
            continue
        for l in open(f):
            t = json.loads(l).get("txt") or ""
            if t:
                g.append(grams(enc(t)))
    ev["ich"] = np.unique(np.concatenate(g))
    g = [grams(enc(open(f"/tmp/nestquant/18-e2e/evalsets/{f}").read())) for f in
         ("glm52-heldout-xl.txt", "glm52-heldout.txt", "glm52-neutral-wikitext.txt", "glm52-neutral-github.txt")]
    ev["t18"] = np.unique(np.concatenate(g))
    return ev


def corpus_hashes():
    g = []
    for l in open(f"{Wd}/conv_docs.jsonl"):
        g.append(np.unique(grams(json.loads(l)["ids"])))
    return np.unique(np.concatenate(g))


# user overrides (2026-09-28): GPQA traces are kept (their question ids go in the manifest); all TB2.1 trajectories kept
EXEMPT = {("gpqa", "gpqa-glm52hybrid"): "GPQA traces kept by user override; ids in manifest",
          ("tb4", "tb21-glm53arvq"): "all TB2.1 trajectories kept by lead decision"}


def main(path=f"{Wd}/trace_docs.jsonl"):
    ev = eval_hashes(); print({k: len(v) for k, v in ev.items()}, flush=True)
    corp = corpus_hashes(); print("corpus grams", len(corp), flush=True)
    drop = {}; notes = {}; cnt = collections.Counter(); tot = collections.Counter()
    for l in open(path):
        d = json.loads(l); tot[d["source"]] += 1
        g = np.unique(grams(d["ids"]))
        if len(g) == 0:
            continue
        for k, s in ev.items():
            fr = float(np.isin(g, s).mean())
            if fr >= 0.05:
                if (k, d["source"]) in EXEMPT:  # lead/user decision: keep, but report the overlap
                    notes[d["source_id"]] = f"{k}:{fr:.3f} (kept: {EXEMPT[(k, d['source'])]})"; continue
                drop[d["source_id"]] = f"{k}:{fr:.3f}"; break
        else:
            fr = float(np.isin(g, corp).mean())
            if fr >= 0.5:
                drop[d["source_id"]] = f"corpus:{fr:.3f}"
        if d["source_id"] in drop:
            cnt[(d["source"], drop[d["source_id"]].split(":")[0])] += 1
    json.dump(dict(drop=drop, kept_overlaps=notes), open(f"{Wd}/trace_dedup.json", "w"), indent=0)
    print("kept overlaps", notes)
    print("docs", dict(tot)); print("dropped", {f"{a}/{b}": v for (a, b), v in cnt.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
