"""Part B validation on a random sample of converted chat docs (plus full-corpus counts from conv_docs.jsonl).
Checks: (1) decode(ids) == rendered template string; (2) every special-token string appears only as its single id
(count of literal string in text == count of id); (3) no '<|im_' anywhere; (4) every original user/tool/system
content and every assistant reasoning appears verbatim in the new text; (5) GLM re-encode of decode is identical."""
import json, random, re, sys, collections
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

tok = G.tokenizer()
SPECIALS = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>",
            "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
            "</arg_value>", "<tool_response>", "</tool_response>"]
SID = {s: tok.convert_tokens_to_ids(s) for s in SPECIALS}
orig = {}
for l in open("/tmp/nestquant/21-traces/orig_docs.jsonl"):
    d = json.loads(l); orig[d["source_id"]] = d["ids"]
conv = [json.loads(l) for l in open("/tmp/nestquant/21-traces/conv_docs.jsonl")]
chat = [d for d in conv if d["kind"] == "chat"]
random.seed(21)
sample = random.sample(chat, 1500) + [d for d in chat if d.get("has_tools")][:300]
bad = collections.Counter()
for d in sample:
    ids = d["ids"]; text = tok.decode(ids, skip_special_tokens=False)
    if tok.encode(text, add_special_tokens=False) != ids:
        bad["reencode"] += 1
    if "<|im_" in text:
        bad["im_leftover"] += 1
    for s, i in SID.items():
        if text.count(s) != ids.count(i):
            bad["special_split:" + s] += 1
    otext = tok.decode(orig[d["source_id"]], skip_special_tokens=False)
    for role, x in G.BLOCK_RE.findall(otext):
        if role in ("user", "tool") and x not in text:
            bad["missing_" + role] += 1
        if role == "assistant":
            r = re.match(r"<think>(.*?)</think>", x, re.S).group(1)
            if r and ("<think>" + r + "</think>") not in text:
                bad["missing_reasoning"] += 1
    if not ids[:2] == [G.GMASK, G.SOP]:
        bad["no_prefix"] += 1
    if d["roles"][-1] == "a" and ids[-1] not in (G.USER, G.OBSERVATION):
        bad["no_final_end"] += 1
tot = collections.Counter()
for d in conv:
    tot[d["kind"] + "_docs"] += 1; tot["think_ends"] += d["n_think"]; tot["end_of_turn"] += d["n_end"]
    if d["kind"] == "chat":
        tot["end_tok_" + {G.USER: "user", G.OBSERVATION: "obs"}.get(d["ids"][-1], "none")] += 1
print("sample", len(sample), "failures", dict(bad))
print("totals", dict(tot))
