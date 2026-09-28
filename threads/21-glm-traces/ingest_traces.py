"""Ingest existing on-box reasoning traces (any model) -> GLM-5.3 chat-format docs for the c2048_traces group.

Each adapter yields normalized conversations:
  dict(source, model, conv_id, category, messages=[OpenAI-style msgs; assistant msgs carry reasoning_content,
       content, tool_calls=[{type,function:{name,arguments(dict)}}]; tool msgs carry content], tools=[...] or None,
       truncated=bool)
Rendering (glmfmt): GLM-5.3 chat_template (history thinking kept), then the end token GLM emits after the final
assistant turn (<|observation|> after a tool call, else <|user|>). Truncated/runaway final turns get NO end token;
if the cut happened inside the reasoning, the final turn is '<|assistant|><think>' + reasoning with no </think>.
Conversations are cut after their last assistant turn (trailing user/tool turns carry no boundary).

Output /tmp/nestquant/21-traces/trace_docs.jsonl: source_id, category ("trace:<src>"), kind="trace", source,
model, truncated, ids, n_think, n_end, n_msgs.
"""
import argparse, glob, hashlib, json, os, re, sys
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

W = "/tmp/nestquant/21-traces"
OUT = f"{W}/trace_docs.jsonl"
MAX_DOC = 2048 * 24  # docs longer than this are cut at an assistant boundary later by pack.py chunking anyway


def synth_tools(msgs):
    """Tool schemas are not stored in most session logs: rebuild minimal ones from the observed calls."""
    seen = {}
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            f = tc["function"]; d = seen.setdefault(f["name"], {})
            for k, v in (f["arguments"] or {}).items():
                d.setdefault(k, "integer" if isinstance(v, int) and not isinstance(v, bool) else
                             "boolean" if isinstance(v, bool) else "array" if isinstance(v, list) else
                             "object" if isinstance(v, dict) else "string")
    return [dict(type="function", function=dict(name=n, description=f"Tool `{n}`.",
                 parameters=dict(type="object", properties={k: dict(type=t) for k, t in ps.items()},
                                 required=sorted(ps)))) for n, ps in seen.items()] or None


# ---------------------------------------------------------------- adapters
def prime_sessions():
    """companion prime harness (glm-5.3f) session jsonl: message records with role user/assistant/toolResult."""
    files = glob.glob("/home/coder/.companion/prime/projects/-home-coder/*.jsonl") + \
        glob.glob("/home/coder/.companion/prime/projects/session-artifacts/**/*.jsonl", recursive=True)
    for f in sorted(files):
        msgs = []; trunc = False
        for l in open(f):
            x = json.loads(l)
            if x.get("type") == "custom_message" and x.get("customType") == "agent_message":
                msgs.append(dict(role="user", content=x["content"])); continue
            if x.get("type") != "message":
                continue
            m = x["message"]; c = m.get("content")
            items = c if isinstance(c, list) else [dict(type="text", text=c or "")]
            if m["role"] == "user":
                msgs.append(dict(role="user", content="\n".join(i.get("text", "") for i in items if i.get("type") == "text")))
            elif m["role"] == "toolResult":
                msgs.append(dict(role="tool", tool_call_id=m.get("toolCallId"),
                                 content="\n".join(i.get("text", "") for i in items if i.get("type") == "text")))
            elif m["role"] == "assistant":
                if m.get("stopReason") == "error":
                    continue
                r = "".join(i.get("thinking", "") for i in items if i.get("type") == "thinking")
                t = "".join(i.get("text", "") for i in items if i.get("type") == "text")
                tcs = [dict(type="function", id=i.get("id"), function=dict(name=i["name"], arguments=i.get("arguments") or {}))
                       for i in items if i.get("type") == "toolCall"]
                a = dict(role="assistant", content=t, reasoning_content=r)
                if tcs:
                    a["tool_calls"] = tcs
                msgs.append(a)
                trunc = m.get("stopReason") == "length"
        yield dict(source="prime-glm53f", model="glm-5.3f", conv_id=os.path.basename(f)[:-6], category="agentic",
                   messages=msgs, tools=synth_tools(msgs), truncated=trunc)


def claude_sessions(root, source, models):
    """Claude-Code jsonl (one record per content block, grouped by message.id). Split into episodes at real user
    text turns; keep episodes with >=1 non-empty thinking block from `models`. Compaction replays (agent-acompact-*)
    re-contain earlier history (and are where plaintext thinking survived): an episode is skipped when all of its
    thinking message ids were already emitted from another file."""
    seen_mid = set()
    for f in sorted(glob.glob(f"{root}/**/*.jsonl", recursive=True)):
        msgs = []; cur = None; mids = {}
        for l in open(f):
            try:
                x = json.loads(l)
            except Exception:
                continue
            if x.get("type") not in ("user", "assistant") or x.get("isMeta"):
                continue
            m = x.get("message") or {}
            c = m.get("content"); items = c if isinstance(c, list) else [dict(type="text", text=c or "")]
            if m.get("role") == "assistant":
                mid = m.get("id")
                if cur is None or cur.get("_mid") != mid:
                    cur = dict(role="assistant", content="", reasoning_content="", _mid=mid, _model=m.get("model"),
                               _stop=m.get("stop_reason"))
                    msgs.append(cur)
                cur["_stop"] = m.get("stop_reason") or cur["_stop"]
                for i in items:
                    if i.get("type") == "thinking":
                        cur["reasoning_content"] += i.get("thinking") or ""
                    elif i.get("type") == "text":
                        cur["content"] += i.get("text") or ""
                    elif i.get("type") == "tool_use":
                        cur.setdefault("tool_calls", []).append(dict(type="function", id=i.get("id"),
                                                                     function=dict(name=i["name"], arguments=i.get("input") or {})))
            else:
                cur = None
                for i in items:
                    if i.get("type") == "tool_result":
                        rc = i.get("content")
                        if isinstance(rc, list):
                            rc = "\n".join(r.get("text", "") for r in rc if r.get("type") == "text")
                        msgs.append(dict(role="tool", tool_call_id=i.get("tool_use_id"), content=rc or ""))
                    elif i.get("type") == "text":
                        t = i.get("text") or ""
                        if t.startswith("<command-") or t.startswith("<local-command") or "[Request interrupted" in t:
                            continue
                        msgs.append(dict(role="user", content=t))
        # tool results must directly follow their calls; drop orphans
        eps = []; ep = []
        for m in msgs:
            if m["role"] == "user" and ep and any(e["role"] == "assistant" for e in ep):
                eps.append(ep); ep = []
            ep.append(m)
        if ep:
            eps.append(ep)
        for k, ep in enumerate(eps):
            asst = [m for m in ep if m["role"] == "assistant"]
            if not any(m["_model"] in models and m["reasoning_content"].strip() for m in asst):
                continue
            if any(m["_model"] not in models for m in asst):  # mixed-model episode: keep only if all turns are ours
                continue
            tm = {m["_mid"] for m in asst if m["reasoning_content"].strip()}
            if tm <= seen_mid:
                continue
            seen_mid |= tm
            trunc = asst[-1]["_stop"] == "max_tokens"
            clean_ep = [{k2: v for k2, v in m.items() if not k2.startswith("_")} for m in ep]
            yield dict(source=source, model=asst[0]["_model"], conv_id=f"{os.path.relpath(f, root)}#{k}",
                       category="agentic", messages=clean_ep, tools=synth_tools(clean_ep), truncated=trunc)


def mdmathena():
    yield from claude_sessions("/home/coder/.claude/projects/-home-coder-mdmathena", "claude-opus46", {"claude-opus-4-6"})


def deepseek_flash():
    yield from claude_sessions("/home/coder/.claude/projects/-home-coder-git-glm52", "dsv41flash", {"deepseek/v41flash-b12x"})


def tb21_arvq():
    """Terminal-Bench 2.1 ATIF trajectories of the GLM-5.3 ARVQ quant (terminus-2). Rendered as GLM tool calls
    (bash_command) with observations as tool responses; the first user step is terminus' full instruction prompt."""
    for f in sorted(glob.glob("/home/coder/git/glm52/artifacts/glm53-hf-audit/ARVQ/benchmarks/tb2.1/*/*/agent/trajectory.json")):
        j = json.load(open(f)); msgs = []
        for s in j["steps"]:
            if s["source"] == "user":
                msgs.append(dict(role="user", content=s["message"] if isinstance(s["message"], str) else json.dumps(s["message"])))
            elif s["source"] == "agent":
                tcs = [dict(type="function", id=t.get("tool_call_id"), function=dict(name=t["function_name"], arguments=t.get("arguments") or {}))
                       for t in (s.get("tool_calls") or [])]
                a = dict(role="assistant", content=s.get("message") or "", reasoning_content=s.get("reasoning_content") or "")
                if tcs:
                    a["tool_calls"] = tcs
                msgs.append(a)
                obs = (s.get("observation") or {}).get("results") or []
                for o in obs:
                    msgs.append(dict(role="tool", content=o.get("content") if isinstance(o.get("content"), str) else json.dumps(o.get("content"))))
                if not tcs and obs:
                    pass
        task = f.split("/")[-3].split("__")[0]
        yield dict(source="tb21-glm53arvq", model="glm-5.3-arvq", conv_id=f.split("tb2.1/")[1], category="agentic_tb",
                   messages=msgs, tools=synth_tools(msgs), truncated=False, task=task)


def ctap2_r13(n=1300, seed=13):
    """ctap2 GRPO round-13 rollouts (Qwen3.8-Flash-Next + LoRA). completion = reasoning '</think>' answer (no
    opening tag); images are not on this box, so the user turn is the text instruction only. All finish=length
    rollouts are kept (truncated, no end token) plus a seeded random sample of the rest."""
    import random
    rows = [json.loads(l) for l in open("/home/coder/ctap2_archive/round013/round013.jsonl")]
    rng = random.Random(seed)
    trunc = [i for i, r in enumerate(rows) if r["finish_reason"] == "length"]
    rest = [i for i, r in enumerate(rows) if r["finish_reason"] != "length"]
    pick = trunc + rng.sample(rest, max(0, n - len(trunc)))
    for i in sorted(pick):
        r = rows[i]; comp = r["completion"]; tr = r["finish_reason"] == "length"
        if "</think>" in comp:
            rs, ans = comp.split("</think>", 1)
        else:
            rs, ans = comp, ""
        yield dict(source="ctap2r13-qwen38", model="qwen3.8-flash-next+lora013", conv_id=f"r13:{i}:{r['block_id']}",
                   category=f"ct:{r['kind']}", truncated=tr,
                   messages=[dict(role="user", content=r["instruction"]),
                             dict(role="assistant", reasoning_content=rs.strip("\n"), content=ans.strip())])


RAD = "/home/coder/git/rad-agent"
TAIL = 2048  # radagent prompts are ~17k tokens of fixed rubric: keep only the last TAIL tokens (one full window)


def radagent_gptoss(n=750, seed=7):
    """rad-agent missed-finding validation (Groq batch, gpt-oss-120b, 2026-03-13). Prompts rebuilt from
    misses/*.csv + validate_misses.SYSTEM_PROMPT (current file; custom_id row/pair_index verified). One run per
    pair, seeded sample stratified by category."""
    import random, pandas as pd
    sys.path.insert(0, f"{RAD}/scripts")
    from validate_misses import SYSTEM_PROMPT
    CATS = ["fracture", "effusion", "mass", "nodule", "lymphadenopathy", "consolidation", "dislocation", "other"]
    df = pd.concat([pd.read_csv(f"{RAD}/misses/{c}.csv").assign(finding_category=c) for c in CATS], ignore_index=True)
    by = {}
    for f in sorted(glob.glob(f"{RAD}/batch_results/batch_00[0-4][0-9]_results.json")):
        for x in json.load(open(f)):
            b = (x.get("response") or {}).get("body") or {}
            if not b.get("choices"):
                continue
            cat, i, p, run = x["custom_id"].split("__")
            by.setdefault((cat, int(i)), []).append(x)
    rng = random.Random(seed)
    keys = sorted(by); rng.shuffle(keys)
    percat = {}
    for k in keys:
        percat.setdefault(k[0], []).append(k)
    pick = []; q = n // len(percat)
    for c, ks in percat.items():
        pick += ks[:q]
    for (cat, i) in sorted(pick):
        x = rng.choice(by[(cat, i)]); row = df.iloc[i]
        assert row["finding_category"] == cat and str(row.get("pair_index", i)) == x["custom_id"].split("__")[2]
        user_msg = (
            f"## First-pass finding category: {cat.upper()}\n"
            f"## First-pass explanation: {row.get('explanation', 'N/A')}\n"
            f"## Reported modalities: Report A = {row.get('report_a_modality', 'N/A')} "
            f"| Report B = {row.get('report_b_modality', 'N/A')}\n"
            f"## Index study description: {row.get('index_dicom_study_desc', 'N/A')}"
            f" | Body part: {row.get('index_body_part_examined', 'N/A')}\n\n---\n\n"
            f"**REPORT A (Index Report — under review):**\n"
            f"{row.get('index_report', 'N/A')}\n\n---\n\n"
            f"**REPORT B (Comparison Report — same day):**\n"
            f"{row.get('other_report', 'N/A')}\n")
        ch = x["response"]["body"]["choices"][0]; m = ch["message"]
        yield dict(source="radagent-gptoss120b", model="gpt-oss-120b", conv_id=x["custom_id"], category=f"rad:{cat}",
                   truncated=ch.get("finish_reason") == "length", tail=TAIL,
                   messages=[dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=user_msg),
                             dict(role="assistant", reasoning_content=m.get("reasoning") or "", content=m.get("content") or "")])


def glm_cc():
    for p in ("-home-coder-git-ctc-ai", "-home-coder-git-rad-agent"):
        yield from claude_sessions(f"/home/coder/.claude/projects/{p}", "cc-glm53f", {"glm-5.3f"})


def gpqa_glm52():
    """GLM-5.2 NVFP4+AQLM hybrid GPQA-diamond traces (glm52/gpqa/gpqa_trace.py): prompt rebuilt with the same
    seed-0 choice permutation and reasoning_effort=high; reasoning is inline in content (split at </think> if
    present, else all reasoning). fin=length runaways are kept as truncated (no </think>, no end token)."""
    import csv, random
    rows = list(csv.DictReader(open("/home/coder/git/glm52/gpqa/gpqa_diamond.csv")))
    rng = random.Random(0)
    for r in rows:
        r["permutation"] = rng.sample(range(4), 4)
    T = ("Answer the following multiple choice question. The last line of your response should be "
         "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
         "Think step by step before answering.\n\n{Q}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")
    for f in sorted(glob.glob("/home/coder/git/glm52/gpqa/traces_ds_mla/*.json")):
        j = json.load(open(f)); r = rows[j["i"]]
        ch = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        ch = [ch[k] for k in r["permutation"]]
        assert "ABCD"[ch.index(r["Correct Answer"])] == j["gold"], f
        prompt = T.format(Q=r["Question"], A=ch[0], B=ch[1], C=ch[2], D=ch[3])
        txt = (j.get("reasoning") or "") + (j.get("content") or "")
        tr = j["fin"] == "length"
        split = "tag"
        if "</think>" in txt:
            rs, ans = txt.split("</think>", 1); rs = rs.split("<think>")[-1]
        else:
            # the server detokenized with skip_special_tokens, which deleted </think> and left the answer glued to
            # the reasoning ("...concise reasoning.Formaldehyde ...", "...answer: D.**Step-by-step"): for finished
            # traces recover the split at the LAST such no-space join if the tail holds the final 'Answer:' line
            rs, ans, split = txt, "", "none"
            js = [m.start() + 2 for m in re.finditer(r"[A-Za-z)][.!?](?=[A-Z]|\*\*)", txt)]
            if not tr and js and re.search(r"Answer:\s*[A-D]\s*$", txt[js[-1]:]):
                rs, ans, split = txt[:js[-1]], txt[js[-1]:], "recovered"
        print("  gpqa", os.path.basename(f), "fin", j["fin"], "split", split, len(rs), len(ans), file=sys.stderr)
        yield dict(source="gpqa-glm52hybrid", model="glm-5.2-nvfp4-aqlm-hybrid", conv_id=os.path.basename(f),
                   category="gpqa", truncated=tr, effort="high", gpqa_index=j["i"],
                   gpqa_record_id=r.get("Record ID"),
                   messages=[dict(role="user", content=prompt), dict(role="assistant", reasoning_content=rs, content=ans)])


# user override (lead, 2026-09-28): Qwen (ctap2, ICH) and gpt-oss sources are EXCLUDED; GPQA traces are KEPT.
ADAPTERS = dict(tb21=tb21_arvq, prime=prime_sessions, claude=mdmathena, dsflash=deepseek_flash, glmcc=glm_cc,
                gpqa=gpqa_glm52)
EXCLUDED_ADAPTERS = dict(ctap2=ctap2_r13, radagent=radagent_gptoss)


# ---------------------------------------------------------------- rendering
def clean(msgs):
    """merge consecutive same-role user msgs, drop empty user msgs, cut after the last assistant turn."""
    out = []
    for m in msgs:
        if m["role"] == "assistant":  # GLM emits no whitespace padding inside <think>..</think>
            m = dict(m, reasoning_content=(m.get("reasoning_content") or "").strip(), content=(m.get("content") or "").strip())
        if m["role"] == "user" and not (m.get("content") or "").strip():
            continue
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            out[-1] = dict(out[-1], content=out[-1]["content"] + "\n\n" + m["content"]); continue
        out.append(m)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    # drop a leading run that isn't user/system (subagent logs that start mid-stream)
    while out and out[0]["role"] not in ("user", "system"):
        out.pop(0)
    return out


def render_conv(cv):
    msgs = clean(cv["messages"])
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    tok = G.tokenizer()
    if not cv.get("truncated"):
        s, ids = G.render(msgs, tools=cv.get("tools"), reasoning_effort=cv.get("effort"))
        return ids
    last = msgs[-1]
    if (last.get("content") or last.get("tool_calls")):  # cut after the reasoning finished: keep </think>, no end
        s, ids = G.render(msgs, tools=cv.get("tools"), reasoning_effort=cv.get("effort"), final_end=False)
        return ids
    kw = dict(reasoning_effort=cv["effort"]) if cv.get("effort") else {}
    s = tok.apply_chat_template(msgs[:-1], tools=cv.get("tools"), tokenize=False, add_generation_prompt=True, **kw)
    assert s.endswith("<|assistant|><think>"), s[-40:]
    return tok.encode(s, add_special_tokens=False) + tok.encode(last.get("reasoning_content") or "", add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    names = [a.only] if a.only else list(ADAPTERS)
    out_path = OUT if not a.only else f"{W}/trace_docs.{a.only}.jsonl"
    n = 0; stats = {}
    with open(out_path, "w") as f:
        for name in names:
            for cv in ADAPTERS[name]():
                ids = render_conv(cv)
                if ids is None:
                    continue
                if cv.get("tail") and len(ids) > cv["tail"]:
                    ids = ids[-cv["tail"]:]
                th, en = G.boundaries(ids)
                sid = hashlib.sha1(f"{cv['source']}:{cv['conv_id']}".encode()).hexdigest()[:16]
                rec = dict(source_id=f"trace:{sid}", category=f"trace:{cv['source']}", kind="trace", source=cv["source"],
                           model=cv["model"], conv_id=cv["conv_id"], sub=cv.get("category"), truncated=bool(cv.get("truncated")),
                           ids=ids, n_think=len(th), n_end=len(en), n_msgs=len(cv["messages"]), task=cv.get("task"),
                           gpqa_index=cv.get("gpqa_index"), gpqa_record_id=cv.get("gpqa_record_id"))
                f.write(json.dumps(rec) + "\n"); n += 1
                s = stats.setdefault(cv["source"], dict(docs=0, tokens=0, think=0, end=0, truncated=0))
                s["docs"] += 1; s["tokens"] += len(ids); s["think"] += len(th); s["end"] += len(en); s["truncated"] += rec["truncated"]
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
