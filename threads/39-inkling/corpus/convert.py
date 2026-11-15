"""Convert reassembled GLM-5.3 documents (work/<group>_docs.jsonl from reassemble.py) to Inkling-2-Small native
format. Output work/<group>_ink.jsonl: one line per source document with its Inkling fragments
(one fragment per contiguous run of the source document), plus work/<group>_convstats.json.

Chats: glmparse inverts the GLM template; the GLM round trip (re-render with GLM template == decoded text) is checked
for every chat document; then rendered with the Inkling template (reasoning_effort 0.9 if any non-empty reasoning
else 0). Partial fragments are rendered via sentinels so the Inkling text starts/ends at the same logical place.
Raw: decoded and plainly re-tokenized. Parse failure -> raw fallback (role tokens -> blank line, other GLM specials
dropped)."""
import json, re, sys, os, collections, hashlib
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmparse as GP
import inkfmt as IF

GLM_DIR, INK_DIR = "/tmp/nq_inkling/glm_tok", "/tmp/nq_inkling/inktok"
WORK = "/tmp/nq_inkling/corpus/work"
SH, ST = "", ""  # head / tail sentinels (private use, verified absent from the corpus)
END_TXT = {GP.USER: "<|user|>", GP.OBSERVATION: "<|observation|>", GP.EOT: "<|endoftext|>"}
DANGLING_MODEL_TEXT = "<|message_model|><|content_text|>"
_g = {}


def init():
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    _g["glm_raw"] = Tokenizer.from_file(f"{GLM_DIR}/tokenizer.json")
    _g["glm"] = AutoTokenizer.from_pretrained(GLM_DIR)
    _g["ink"] = AutoTokenizer.from_pretrained(INK_DIR)
    _g["parser"] = GP.Parser(_g["glm_raw"])
    _g["glm_special"] = {i for i in range(154820, 154880)}


def _mark(P):
    """Copy messages with head/tail sentinels inserted into the partial fields."""
    msgs = [dict(m) for m in P.msgs]
    if P.head and P.head["field"]:
        m = msgs[0]; m[P.head["field"]] = SH + m[P.head["field"]].lstrip("�")
    t = P.tail
    if t:
        m = msgs[-1]
        if t["kind"] == "field":
            m[t["field"]] = m[t["field"]].rstrip("�") + ST
        elif t["kind"] == "turn" and not m.get("tool_calls"):
            m["content"] = m["content"] + ST  # content is "" here (cut right after </think>)
    return msgs


def _cut(s, P):
    if P.head and P.head["field"]:
        k = s.index(SH); s = s[k + 1:]
    if P.tail:
        if ST in s:
            s = s[:s.index(ST)]
        elif P.tail["kind"] == "turn":
            assert s.endswith("<|content_model_end_sampling|>") or s.endswith("</tool_call>"), s[-60:]
            if s.endswith("<|content_model_end_sampling|>"):
                s = s[:-len("<|content_model_end_sampling|>")]
    assert SH not in s and ST not in s
    return s


def glm_roundtrip(P, text, at_start):
    """Re-render the parsed doc with the GLM template and compare to the decoded source text (exact)."""
    msgs = _mark(P)
    kw = dict(reasoning_effort=P.effort) if P.effort else {}
    s = _cut(_g["glm"].apply_chat_template(msgs, tools=P.tools, tokenize=False, **kw), P)
    if not at_start and not (P.head and P.head["field"]):
        pre = "[gMASK]<sop><|system|>Reasoning Effort: Max"
        assert s.startswith(pre)
        s = s[len(pre):]
        if P.head:
            assert s.startswith(P.head["glm_prefix"])
            s = s[len(P.head["glm_prefix"]):]
    if P.end_tok:
        s += END_TXT[P.end_tok]
    # a cut fragment may start/end inside a multi-byte char (U+FFFD on decode); _mark drops those, compare likewise
    if P.head and P.head["field"]:
        text = text.lstrip("\ufffd")
    if P.tail and P.tail["kind"] == "field":
        text = text.rstrip("\ufffd")
    if P.tail and P.tail.get("dropped_partial_call"):
        return text.startswith(s)
    return s == text


def ink_render(P, at_start):
    msgs = _mark(P)
    reasoning = any((m.get("reasoning_content") or "").strip() for m in P.msgs if m["role"] == "assistant")
    s = _g["ink"].apply_chat_template(msgs, tools=P.tools, tokenize=False, add_generation_prompt=False,
                                      reasoning_effort=0.9 if reasoning else 0)
    s = _cut(s, P)
    if not at_start and not (P.head and P.head["field"]):  # fragment: drop the template's effort preamble
        eff = "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>" % ("0.9" if reasoning else "0")
        assert eff in s and not P.tools
        s = s.replace(eff, "", 1)
    if s.endswith(DANGLING_MODEL_TEXT) and P.tail:
        s = s[:-len(DANGLING_MODEL_TEXT)]
    return s, reasoning


def raw_fallback_text(ids):
    tok = _g["glm_raw"]; parts, cur = [], []
    for t in ids:
        if t in GP.ROLES:
            parts.append(tok.decode(cur, skip_special_tokens=True)); cur = []
        elif t not in _g["glm_special"]:
            cur.append(t)
    parts.append(tok.decode(cur, skip_special_tokens=True))
    return "\n\n".join(p for p in parts if p)


def head_init(P):
    if not P.head:
        return None
    r = {"assistant": IF.MODEL, "tool": IF.TOOL, "user": IF.USER, "system": IF.SYSTEM}[P.head["role"]]
    c = IF.THINKING if P.head["field"] == "reasoning_content" else IF.TEXT
    return (r, c)


def convert_fragment(ids, kind, at_start):
    """-> dict(ids, text, mode, init, ...)."""
    ink = _g["ink"]
    if kind == "raw":
        text = _g["glm_raw"].decode(ids, skip_special_tokens=False)
        return dict(mode="raw", text=text, ids=ink.encode(text, add_special_tokens=False, split_special_tokens=True), init=None)
    text = _g["glm_raw"].decode(ids, skip_special_tokens=False)
    if SH in text or ST in text:
        raise RuntimeError("sentinel in corpus")
    try:
        P = _g["parser"].parse(ids, at_start=at_start)
        rt = glm_roundtrip(P, text, at_start)
        s, reasoning = ink_render(P, at_start)
    except (GP.ParseError, AssertionError, ValueError, KeyError, TypeError) as e:
        text = raw_fallback_text(ids)
        return dict(mode="fallback", text=text, ids=ink.encode(text, add_special_tokens=False, split_special_tokens=True), init=None, err=f"{type(e).__name__}: {e}"[:200])
    out_ids = ink.encode(s, add_special_tokens=False)
    return dict(mode="chat", text=s, ids=out_ids, init=head_init(P), roundtrip=rt, reasoning=reasoning,
                head=P.head, tail=P.tail, end_tok=P.end_tok, tools=P.tools is not None,
                n_msgs=len(P.msgs), n_calls=sum(len(m.get("tool_calls") or []) for m in P.msgs),
                n_reason=sum(1 for m in P.msgs if m["role"] == "assistant" and m.get("reasoning_content")),
                notes=P.notes)


def work(line):
    d = json.loads(line)
    ids = d.pop("ids"); runs, a = [], 0
    for i, (off, n) in enumerate(d["pieces"]):  # contiguous runs of the source document
        if runs and off == runs[-1][0] + runs[-1][2]:
            runs[-1][2] += n
        else:
            runs.append([off, a, n])
        a += n
    frags = []
    for off, a, n in runs:
        f = convert_fragment(ids[a:a + n], d["kind"], at_start=off == 0)
        th, en = IF.boundaries(f["ids"], chat=f["mode"] == "chat", init=f["init"])
        f.update(src_token_offset=off, src_tokens=n, think=th, end=en,
                 text_sha256=hashlib.sha256(f.pop("text").encode()).hexdigest())
        frags.append(f)
    d["fragments"] = frags
    return d


def main(group):
    st = collections.Counter(); errs = []
    with open(f"{WORK}/{group}_docs.jsonl") as f, open(f"{WORK}/{group}_ink.jsonl", "w") as g, Pool(32, initializer=init) as pool:
        for d in pool.imap(work, f, chunksize=8):
            g.write(json.dumps(d) + "\n")
            for fr in d["fragments"]:
                k = fr["mode"]; st[f"frag_{k}"] += 1; st[f"tok_{k}"] += len(fr["ids"]); st[f"srctok_{k}"] += fr["src_tokens"]
                if k == "chat":
                    st["roundtrip_pass"] += bool(fr["roundtrip"]); st["head_partial"] += fr["head"] is not None
                    st["tail_" + (fr["tail"]["kind"] if fr["tail"] else "none")] += 1
                    st["dropped_partial_call"] += bool(fr["tail"] and fr["tail"].get("dropped_partial_call"))
                    st["head_dropped_partial_call"] += any(n.startswith("head_dropped") for n in fr["notes"])
                    st["think"] += len(fr["think"]); st["end"] += len(fr["end"])
                    st["frag_with_tools"] += fr["tools"]; st["tool_calls"] += fr["n_calls"]
                    st["think_end_coincide"] += len(set(fr["think"]) & set(fr["end"]))
                if k == "fallback":
                    errs.append((d["source_id"], d["category"], fr["err"]))
    st = dict(st); st["fallback_errors"] = errs
    json.dump(st, open(f"{WORK}/{group}_convstats.json", "w"), indent=1)
    print(group, json.dumps({k: v for k, v in st.items() if k != "fallback_errors"}, indent=1))
    for e in errs[:20]:
        print("  FALLBACK", e)


if __name__ == "__main__":
    for gname in sys.argv[1:] or ["c2048", "c2048_traces"]:
        main(gname)
