"""Validate packed outputs against the converted docs, and write SAMPLES.txt."""
import json, sys, hashlib, collections
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0, "/tmp/nq_inkling/code/corpus")
import inkfmt as IF
INK = Tokenizer.from_file("/tmp/nq_inkling/inktok/tokenizer.json")
GLM = Tokenizer.from_file("/tmp/nq_inkling/glm_tok/tokenizer.json")
R = "/tmp/nq_inkling/corpus"
for g in ("c2048", "c2048_traces"):
    docs = {}
    for l in open(f"{R}/work/{g}_ink.jsonl"):
        d = json.loads(l); docs[d["source_id"]] = d
    tok = np.load(f"{R}/{g}/tokens.npy"); seg = np.load(f"{R}/{g}/segments.npy")
    bt = np.load(f"{R}/{g}/bnd_think_d.npy"); be = np.load(f"{R}/{g}/bnd_end_d.npy")
    sp = json.load(open(f"{R}/{g}/split.json"))
    got = collections.defaultdict(dict); role = {}
    for w, l in enumerate(open(f"{R}/{g}/windows.jsonl")):
        r = "fit" if w < sp["fit"][1] else "val"
        o = 0
        for s in json.loads(l)["segments"]:
            a, n = s["window_offset"], s["tokens"]; assert a == o; o += n
            assert (seg[w, a:a + n] == s["segment_id"]).all()
            assert role.setdefault(s["source_id"], r) == r, "doc crosses split"
            assert r == docs[s["source_id"]]["split"]
            got[s["source_id"]][s["token_offset"]] = tok[w, a:a + n].tolist()
        assert (seg[w, o:] == -1).all() and (tok[w, o:] == IF.PAD).all() and (bt[w, o:] == 0).all() and (be[w, o:] == 0).all()
    assert not ((bt > 0) & (be > 0)).any()
    bad = 0; rawspecial = 0
    for sid, d in docs.items():
        ids = [x for _, v in sorted(got[sid].items()) for x in v]
        ref = [x for f in d["fragments"] for x in f["ids"]]
        bad += ids != ref
        for f in d["fragments"]:
            if f["mode"] != "chat":
                rawspecial += sum(1 for x in f["ids"] if x >= 199999)
            assert hashlib.sha256(INK.decode(f["ids"], skip_special_tokens=False).encode()).hexdigest() == f["text_sha256"]
    print(g, "docs", len(docs), "mismatched docs", bad, "special ids in raw/fallback", rawspecial,
          "eos count", int((tok == IF.EOS).sum()))

# samples
def pick(g, pred):
    best = None
    for l in open(f"{R}/work/{g}_ink.jsonl"):
        d = json.loads(l)
        f = d["fragments"][0]
        if pred(d, f) and (best is None or len(f["ids"]) < len(best[1]["ids"])):
            best = (d, f)
    return best
chat = pick("c2048", lambda d, f: f["mode"] == "chat" and f["n_calls"] > 0 and f["reasoning"] and f["src_token_offset"] == 0)
raw = pick("c2048", lambda d, f: f["mode"] == "raw" and 300 < len(f["ids"]))
src = {json.loads(l)["source_id"]: json.loads(l) for l in open(f"{R}/work/c2048_docs.jsonl")}
with open(f"{R}/SAMPLES.txt", "w") as o:
    for name, (d, f) in (("REASONING CHAT WITH TOOL CALLS", chat), ("RAW", raw)):
        o.write(f"{'=' * 100}\n{name}: source_id={d['source_id']} category={d['category']} split={d['split']} "
                f"inkling_tokens={len(f['ids'])} think_bnd={f['think']} end_bnd={f['end']}\n{'-' * 40} INKLING (decoded ids)\n")
        o.write(INK.decode(f["ids"], skip_special_tokens=False) + "\n")
        o.write(f"{'-' * 40} GLM SOURCE (decoded ids)\n" + GLM.decode(src[d["source_id"]]["ids"], skip_special_tokens=False) + "\n")
print(open(f"{R}/SAMPLES.txt").read()[:6000])
