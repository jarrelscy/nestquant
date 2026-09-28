"""Part B: convert orbit glm53_training_15m_v2 documents (orig_docs.jsonl from rebuild_docs.py) to GLM-5.3 native
chat format. ChatML docs are parsed and re-rendered with chat_template.jinja; raw docs keep their original ids.
Output: conv_docs.jsonl (one line per doc: source_id, category, kind, ids, n_think, n_end, ...), conv_stats.json."""
import json, sys, collections
from multiprocessing import Pool
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

IN = "/tmp/nestquant/21-traces/orig_docs.jsonl"
OUT = "/tmp/nestquant/21-traces/conv_docs.jsonl"


def orig_real_think(ids):
    return sum(1 for i, x in enumerate(ids) if x == G.ETHINK and i > 0 and ids[i - 1] != G.THINK)


def work(line):
    d = json.loads(line)
    tok = G.tokenizer()
    text = tok.decode(d["ids"], skip_special_tokens=False)
    rec = dict(source_id=d["source_id"], category=d["category"], orig_tokens=len(d["ids"]),
               orig_real_think=orig_real_think(d["ids"]))
    if "<|im_start|>" not in text:
        rec.update(kind="raw", ids=d["ids"], n_think=0, n_end=0)
        return rec, None
    try:
        msgs, tools = G.parse_chatml(text)
    except G.ParseError as e:
        return None, (d["source_id"], str(e), text[:300])
    s, ids = G.render(msgs, tools)
    # checks: decode round-trip of the rendered string, no ChatML leftovers, specials are single ids
    err = []
    if tok.decode(ids[:len(ids) - (1 if msgs[-1]["role"] == "assistant" else 0)], skip_special_tokens=False) != s:
        err.append("roundtrip")
    if "<|im_" in s:
        err.append("im_leftover")
    th, en = G.boundaries(ids)
    n_reason = sum(1 for m in msgs if m["role"] == "assistant" and m["reasoning_content"].strip())
    if len(th) != n_reason:
        err.append(f"think {len(th)} vs {n_reason}")
    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    if len(en) != n_asst:
        err.append(f"end {len(en)} vs {n_asst}")
    if err:
        return None, (d["source_id"], ";".join(err), s[:300])
    rec.update(kind="chat", ids=ids, n_think=len(th), n_end=len(en), has_tools=tools is not None,
               n_msgs=len(msgs), roles="".join(m["role"][0] for m in msgs))
    return rec, None


if __name__ == "__main__":
    st = collections.Counter(); errs = []
    with open(IN) as f, open(OUT, "w") as g, Pool(16) as pool:
        for rec, err in pool.imap(work, f, chunksize=64):
            if err:
                errs.append(err); st["error"] += 1; continue
            g.write(json.dumps(rec) + "\n")
            k = rec["kind"]
            st[f"{k}_docs"] += 1; st[f"{k}_tokens"] += len(rec["ids"]); st[f"{k}_orig_tokens"] += rec["orig_tokens"]
            st["think_ends"] += rec["n_think"]; st["end_of_turn"] += rec["n_end"]
            st[f"orig_real_think_{k}"] += rec["orig_real_think"]
    st = dict(st); st["errors"] = errs[:50]
    json.dump(st, open("/tmp/nestquant/21-traces/conv_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "errors"}, indent=1))
    for e in errs[:10]:
        print(e)
