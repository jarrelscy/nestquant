"""Rebuild orbit glm53_training_15m_v2 documents (by source_id, token_offset order) -> docs.jsonl (read-only source)."""
import json, numpy as np, sys
from tokenizers import Tokenizer
SRC = "/home/coder/git/orbit-duet/runs/glm53_training_15m_v2"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nestquant/21-traces/orig_docs.jsonl"
tok = Tokenizer.from_file("/tmp/nestquant/src/glm53-fp8/tokenizer.json")
T = np.load(f"{SRC}/tokens.npy", mmap_mode="r")
docs = {}; order = []
with open(f"{SRC}/windows.jsonl") as f:
    for w, line in enumerate(f):
        for s in json.loads(line)["segments"]:
            sid = s["source_id"]
            if sid not in docs:
                docs[sid] = dict(category=s["category"], parts=[]); order.append(sid)
            docs[sid]["parts"].append((s["token_offset"], w, s["window_offset"], s["tokens"]))
n_bad = 0
with open(OUT, "w") as g:
    for sid in order:
        d = docs[sid]; parts = sorted(d["parts"]); ids = []
        for off, w, wo, n in parts:
            if off != len(ids): n_bad += 1
            ids.extend(int(x) for x in T[w, wo:wo + n])
        g.write(json.dumps(dict(source_id=sid, category=d["category"], first_window=parts[0][1],
                                windows=sorted({p[1] for p in parts}), ids=ids)) + "\n")
print("docs", len(order), "offset gaps", n_bad)
