"""Part A prompt set -> /tmp/nestquant/21-traces/prompts.jsonl (one line per request; samples expanded).

Sources:
  tb     Terminal-Bench 2.1 (harbor-framework/terminal-bench archive/ = 2.0 tasks with 2.1 revisions) + the 8
         TB 3.0 tasks that were removed in 4.0 (git tag v3.0.0). TB 4.0 tasks are an eval set (streaming tb4) ->
         every TB3 task still in 4.0 is excluded, and TB2 tasks with >=10% 13-gram overlap with any TB4 instruction.
  math   NuminaMath-1.5 olympiad/contest word problems (no amc_aime source: AIME is a common eval)
  cp     open-r1/codeforces problem statements
  debug / sysdesign / chat / writing_wc   WildChat-1M first user turns (keyword-classified), English + Chinese
  medical  FreedomIntelligence/medical-o1-reasoning-SFT questions
  science  camel-ai physics/chemistry/biology
  writing  euclaise/writingprompts
  short  trivia_qa (nocontext) questions, alpaca short instructions, simple arithmetic
All non-TB prompts are 13-gram deduped against GPQA-diamond and thread-18 eval texts.
"""
import ast, csv, hashlib, json, random, re, subprocess, sys, time, urllib.parse, urllib.request
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

W = "/tmp/nestquant/21-traces"
TB = f"{W}/tb_main"
rng = random.Random(2126)
tok = G.tokenizer()


def grams(text, n=13):
    ids = tok.encode(text, add_special_tokens=False)
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def rows(ds, config, split, n_blocks, length=100, total=None):
    base = "https://datasets-server.huggingface.co/rows?"
    if total is None:
        q = urllib.parse.urlencode(dict(dataset=ds, config=config, split=split, offset=0, length=1))
        total = json.load(urllib.request.urlopen(base + q, timeout=60))["num_rows_total"]
    out = []
    for off in rng.sample(range(0, max(1, total - length)), n_blocks):
        q = urllib.parse.urlencode(dict(dataset=ds, config=config, split=split, offset=off, length=length))
        for t in range(5):
            try:
                out += [r["row"] for r in json.load(urllib.request.urlopen(base + q, timeout=60))["rows"]]
                break
            except Exception as e:
                time.sleep(2 + 3 * t)
    return out


# ---------------- tools / system prompt for agentic turns ----------------
TOOLS = [
    dict(type="function", function=dict(name="bash", description="Run a bash command in the task container and return its stdout and stderr (combined, truncated to 16 KB). The working directory persists between calls.",
         parameters=dict(type="object", properties=dict(command=dict(type="string", description="The bash command to run."),
                                                        timeout=dict(type="integer", description="Timeout in seconds (default 120).")), required=["command"]))),
    dict(type="function", function=dict(name="read_file", description="Read a text file from the container, optionally a line range.",
         parameters=dict(type="object", properties=dict(path=dict(type="string", description="Absolute path of the file."),
                                                        start_line=dict(type="integer"), end_line=dict(type="integer")), required=["path"]))),
    dict(type="function", function=dict(name="write_file", description="Create or overwrite a text file in the container with the given content.",
         parameters=dict(type="object", properties=dict(path=dict(type="string", description="Absolute path of the file."),
                                                        content=dict(type="string", description="Full file content.")), required=["path", "content"]))),
]
TB_SYSTEM = ("You are an autonomous software agent working inside a Linux (Debian/Ubuntu) container. The task files are in "
             "the current working directory (usually /app). You have no internet access beyond what the task states. "
             "Use the provided tools to inspect the environment, run commands, and edit files until the task is fully "
             "complete; your work is checked by automated tests after you finish. Work step by step and call one tool "
             "at a time.")


def tb_prompts():
    tb4 = {p.name: (p / "instruction.md").read_text() for p in __import__("pathlib").Path(f"{TB}/tasks").iterdir()
           if (p / "instruction.md").exists()}
    tb4_grams = set().union(*[grams(t) for t in tb4.values()])
    out, excluded = [], []
    arch = __import__("pathlib").Path(f"{TB}/archive")
    for p in sorted(arch.iterdir()):
        f = p / "instruction.md"
        if not f.exists():
            continue
        txt = f.read_text(); g = grams(txt)
        ov = len(g & tb4_grams) / max(1, len(g))
        if p.name in tb4 or ov >= 0.10:
            excluded.append((p.name, round(ov, 3))); continue
        out.append(dict(src="tb2.1", task=p.name, instruction=txt))
    tb3 = subprocess.run(["git", "-C", TB, "ls-tree", "--name-only", "v3.0.0", "tasks/"], capture_output=True, text=True).stdout.split()
    for t in tb3:
        name = t.split("/", 1)[1]
        if name in tb4 or name in ("README.md", "dataset.toml") or any(o["task"] == name for o in out):
            continue
        txt = subprocess.run(["git", "-C", TB, "show", f"v3.0.0:tasks/{name}/instruction.md"], capture_output=True, text=True).stdout
        if txt.strip():
            out.append(dict(src="tb3.0", task=name, instruction=txt))
    return out, excluded


def main():
    P = []
    tb, tb_excl = tb_prompts()
    print("tb tasks", len(tb), "excluded", len(tb_excl), tb_excl)
    for t in tb:
        for k in range(5):
            P.append(dict(cat="agentic_tb", src=t["src"], item=t["task"], sample=k,
                          messages=[dict(role="system", content=TB_SYSTEM), dict(role="user", content=t["instruction"])],
                          tools=TOOLS))
    # eval dedup grams
    ev = set()
    for r in csv.DictReader(open("/home/coder/git/glm52/gpqa/gpqa_diamond.csv")):
        ev |= grams(r["Question"])
    for f in ["glm52-heldout-xl.txt", "glm52-heldout.txt", "glm52-neutral-wikitext.txt", "glm52-neutral-github.txt"]:
        ev |= grams(open(f"/tmp/nestquant/18-e2e/evalsets/{f}").read())

    def add(cat, src, item, text, n=1, system=None):
        g = grams(text)
        if g and len(g & ev) / len(g) >= 0.05:
            return False
        msgs = ([dict(role="system", content=system)] if system else []) + [dict(role="user", content=text)]
        for k in range(n):
            P.append(dict(cat=cat, src=src, item=item, sample=k, messages=msgs))
        return True

    # math
    rs = [r for r in rows("AI-MO/NuminaMath-1.5", "default", "train", 40)
          if r.get("source") in ("olympiads", "cn_contest", "number_theory", "inequalities", "olympiads_ref")
          and r.get("question_type") == "math-word-problem" and r.get("problem_is_valid") == "Yes"
          and 80 < len(r["problem"]) < 2000]
    rng.shuffle(rs); c = 0
    for r in rs:
        if c >= 200: break
        c += add("math", "numina1.5:" + r["source"], hashlib.sha1(r["problem"].encode()).hexdigest()[:12], r["problem"])
    # competitive programming
    rs = [r for r in rows("open-r1/codeforces", "default", "train", 12)
          if r.get("description") and r.get("rating") and int(r["rating"] or 0) >= 1600]
    rng.shuffle(rs); c = 0
    for r in rs:
        if c >= 150: break
        ex = r.get("examples") or []
        ex_txt = "".join(f"\n\nExample input:\n{e['input']}\nExample output:\n{e['output']}" for e in ex[:2]) if isinstance(ex, list) else ""
        text = (f"Solve this competitive programming problem in C++17 or Python 3. Explain the approach briefly, then give the full code.\n\n"
                f"# {r['title']}\nTime limit: {r['time_limit']} s, memory limit: {r['memory_limit']} MB\n\n{r['description']}\n\n"
                f"## Input\n{r['input_format']}\n\n## Output\n{r['output_format']}{ex_txt}")
        c += add("competitive_programming", "codeforces", r["id"], text)
    # medical
    rs = rows("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train", 6)
    rng.shuffle(rs)
    for r in rs[:150]:
        add("medical", "medical-o1", hashlib.sha1(r["Question"].encode()).hexdigest()[:12], r["Question"])
    # science
    for ds in ("camel-ai/physics", "camel-ai/chemistry", "camel-ai/biology"):
        rs = rows(ds, "default", "train", 3); rng.shuffle(rs)
        for r in rs[:50]:
            add("science", ds, hashlib.sha1(r["message_1"].encode()).hexdigest()[:12], r["message_1"])
    # writing prompts
    rs = rows("euclaise/writingprompts", "default", "train", 4); rng.shuffle(rs); c = 0
    for r in rs:
        if c >= 80: break
        pr = re.sub(r"^\[\s*\w+\s*\]\s*", "", r["prompt"]).strip()
        if len(pr) < 40: continue
        c += add("writing", "writingprompts", hashlib.sha1(pr.encode()).hexdigest()[:12],
                 f"Write a short story (roughly 800-1500 words) for this prompt:\n\n{pr}")
    # wildchat
    buckets = dict(debug=[], sysdesign=[], writing_wc=[], chat=[], chat_zh=[])
    for r in rows("allenai/WildChat-1M", "default", "train", 80):
        if str(r.get("toxic")) == "True" or str(r.get("redacted")) == "True" or int(r.get("turn") or 1) != 1:
            continue
        conv = r["conversation"]
        if isinstance(conv, str):
            try: conv = ast.literal_eval(conv)
            except Exception: continue
        u = conv[0]["content"] if conv and conv[0].get("role") == "user" else None
        if not u or not (25 <= len(u) <= 6000):
            continue
        lang = r.get("language")
        low = u.lower()
        if lang == "Chinese":
            buckets["chat_zh"].append(u); continue
        if lang != "English":
            continue
        if re.search(r"traceback|error|exception|bug|doesn't work|does not work|not working|segfault|fix (this|my|the) code", low) and ("```" in u or "\n" in u):
            buckets["debug"].append(u)
        elif re.search(r"\b(design|architect|architecture|scalab|microservice|distributed|system design|database schema)\b", low):
            buckets["sysdesign"].append(u)
        elif re.search(r"\b(write|story|essay|poem|article|blog|chapter)\b", low):
            buckets["writing_wc"].append(u)
        else:
            buckets["chat"].append(u)
    quota = dict(debug=120, sysdesign=60, writing_wc=60, chat=130, chat_zh=50)
    for k, lst in buckets.items():
        rng.shuffle(lst); c = 0
        for u in lst:
            if c >= quota[k]: break
            c += add(k, "wildchat", hashlib.sha1(u.encode()).hexdigest()[:12], u)
    print({k: len(v) for k, v in buckets.items()})
    # templated system design (fills the thin wildchat bucket)
    systems = ["a URL shortener", "a global rate limiter", "a real-time multiplayer game backend", "a distributed job scheduler",
               "a metrics/time-series database", "a collaborative text editor", "a ride-hailing dispatch service",
               "a payment ledger with exactly-once semantics", "a CDN edge cache", "a feature-flag service",
               "a large-scale LLM inference serving platform", "a search autocomplete service", "a notification fan-out system",
               "an object storage service", "a hospital PACS image archive", "a stock exchange matching engine",
               "a log ingestion pipeline", "a recommendation feed", "a vector database", "a chat app with end-to-end encryption"]
    scales = ["10k requests/s", "1M daily active users", "100M users across 3 regions", "a 5-person startup budget", "petabytes of data"]
    for i, s in enumerate(systems):
        for sc in rng.sample(scales, 2):
            add("sysdesign", "template", f"sd{i}", f"Design {s} for {sc}. Cover the API, data model, storage choices, "
                "scaling and failure modes, consistency trade-offs, and what you would build first. Be concrete.")
    # short questions (expected: brief think, quick stop)
    rs = rows("mandarjoshi/trivia_qa", "rc.nocontext", "train", 3); rng.shuffle(rs)
    for r in rs[:150]:
        add("short", "trivia_qa", r["question_id"], r["question"])
    rs = [r for r in rows("tatsu-lab/alpaca", "default", "train", 4) if not r["input"] and len(r["instruction"]) < 150]
    rng.shuffle(rs)
    for r in rs[:100]:
        add("short", "alpaca", hashlib.sha1(r["instruction"].encode()).hexdigest()[:12], r["instruction"])
    for i in range(50):
        a, b = rng.randint(12, 999), rng.randint(12, 999)
        q = rng.choice([f"What is {a} + {b}?", f"What is {a} times {b}?", f"Is {a} a prime number?",
                        f"Convert {a} km to miles.", f"What is {a} mod {b % 50 + 2}?", f"Spell the word 'necessary' backwards.",
                        f"What day of the week comes {a % 7 + 1} days after Monday?", "What is the capital of Australia?"])
        add("short", "synthetic", f"s{i}", q)
    # reasoning effort mix: default max 70%, high 20%, low 10% (short: 50/30/20 so low-effort stops are represented)
    for p in P:
        u = rng.random()
        if p["cat"] == "short":
            p["effort"] = "max" if u < 0.5 else ("high" if u < 0.8 else "low")
        else:
            p["effort"] = "max" if u < 0.7 else ("high" if u < 0.9 else "low")
        p["id"] = hashlib.sha1(json.dumps([p["cat"], p["src"], p["item"], p["sample"]]).encode()).hexdigest()[:16]
    ids = [p["id"] for p in P]
    assert len(ids) == len(set(ids))
    rng.shuffle(P)
    with open(f"{W}/prompts.jsonl", "w") as f:
        for p in P:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    import collections
    print("prompts", len(P), collections.Counter(p["cat"] for p in P), collections.Counter(p["effort"] for p in P))
    json.dump(dict(tb_excluded=tb_excl, tb_used=[(t["src"], t["task"]) for t in tb]), open(f"{W}/tb_selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
