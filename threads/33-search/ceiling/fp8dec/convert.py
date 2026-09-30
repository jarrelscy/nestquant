"""T33l fp8dec: convert public agent trajectories / task statements into a canonical
OpenAI-style form {source, task_id, traj_id, tools, messages} renderable by the GLM-5.3
chat template (tool_calls with dict arguments, role=tool results, reasoning_content).

Mappings (text-action sources -> native tool calls):
  * R2E-Gym / DeepSWE / OpenHands-SFT "<function=NAME><parameter=K>V</parameter></function>"
    -> tool_call NAME(K=V); tool schema parsed from the "BEGIN FUNCTION #i" blocks in the
    system prompt (R2E: file_editor, execute_bash(cmd), search, finish; OH: execute_bash,
    str_replace_editor, finish). "Execution output of [x]:" / "EXECUTION RESULT of [x]:"
    user turns -> role=tool (header stripped).
  * SWE-agent text (nebius): last ``` block of each assistant turn -> bash(command) (the
    SWE-agent special commands open/goto/edit/... are shell commands of that interface);
    "submit" -> submit(). Discussion text -> visible content.
  * Terminus-2 (TB2 traces): JSON {analysis, plan, commands[], task_complete} or GLM-4.7
    style <tool_call>{"name":"bash_command",...}</tool_call> -> bash_command(keystrokes,
    duration) per command + mark_task_complete(); "New Terminal Output" user turns -> tool.
"""
import json, re, hashlib, glob, os
import pyarrow.parquet as pq

SRC = "/tmp/nestquant/33-search/ceiling/fp8dec_src"


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

# ---------------------------------------------------------------- text tool-block parser
_BLK = re.compile(r"BEGIN FUNCTION #\d+: ([\w\-]+) [\-–]+\n(.*?)[\-–]+ END FUNCTION", re.S)
_PAR = re.compile(r"^\s*\d+\.\s*\(?(\w+)\)?\s*\((\w+), (required|optional)\)\s*:?(.*?)(?=^\s*\d+\.\s*\(?\w+\)?\s*\(\w+, (?:required|optional)\)|\Z)", re.S | re.M)
_TYPES = {"string": "string", "integer": "integer", "array": "array", "boolean": "boolean", "number": "number"}


def parse_text_tools(sys_prompt):
    tools = []
    for name, body in _BLK.findall(sys_prompt):
        d, _, p = body.partition("Parameters:")
        d = d.replace("Description:", "", 1).strip()
        props, req = {}, []
        for pn, pt, rq, pd in _PAR.findall(p):
            pd = pd.strip()
            e = {"type": _TYPES.get(pt, "string"), "description": pd}
            m = re.search(r"Allowed values: \[(.*?)\]", pd)
            if m:
                e["enum"] = [x.strip().strip("`'\"") for x in m.group(1).split(",")]
            if e["type"] == "array":
                e["items"] = {"type": "integer"}
            props[pn] = e
            if rq == "required":
                req.append(pn)
        tools.append(fn(name, d, props, req))
    return tools


def strip_text_tool_block(sys_prompt):
    """Remove the text function-calling instructions (tools go through tools= instead)."""
    i = sys_prompt.find("We have access to the following functions")
    if i < 0:
        i = sys_prompt.find("You have access to the following functions")
    return sys_prompt[:i].rstrip() if i >= 0 else sys_prompt


_FN = re.compile(r"<function=([\w\-]+)>(.*?)(?:</function>|$)", re.S)
_FP = re.compile(r"<parameter=([\w\-]+)>(.*?)</parameter>", re.S)


def coerce(v, spec):
    t = (spec or {}).get("type")
    try:
        if t == "integer":
            return int(v.strip())
        if t == "array":
            return json.loads(v)
        if t == "boolean":
            return v.strip().lower() == "true"
    except Exception:
        pass
    return v


def parse_fn_calls(text, tools):
    specs = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in tools}
    calls = []
    for name, body in _FN.findall(text):
        args = {}
        for k, v in _FP.findall(body):
            v = v[1:] if v.startswith("\n") else v
            v = v[:-1] if v.endswith("\n") else v
            args[k] = coerce(v, specs.get(name, {}).get(k))
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    m = text.find("<function=")
    content = text[:m].rstrip() if m >= 0 else text
    return content, calls


def split_think(text):
    if "</think>" in text:
        r = text.split("</think>")[0].split("<think>")[-1]
        return r.strip(), text.split("</think>", 1)[-1].strip()
    return None, text


def amsg(content, calls=None, reasoning=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def tid_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]

# ---------------------------------------------------------------- R2E / DeepSWE / OpenHands text format
_OBS = re.compile(r"^(Exit code: -?\d+\n)?(?:Execution output of|EXECUTION RESULT of) \[[\w\-]+\]:\n?", re.S)


def conv_fn_text(msgs, source, task_id, traj_id):
    sysm = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    tools = parse_text_tools(sysm)
    out = [{"role": "system", "content": strip_text_tool_block(sysm)}] if sysm else []
    for m in msgs[1:] if sysm else msgs:
        c = m["content"] or ""
        if m["role"] == "assistant":
            r, c = split_think(c)
            content, calls = parse_fn_calls(c, tools)
            out.append(amsg(content, calls, r))
        elif m["role"] == "user" and _OBS.match(c) and out and out[-1]["role"] == "assistant":
            out.append({"role": "tool", "content": _OBS.sub(lambda m: m.group(1) or "", c, count=1)})
        else:
            out.append({"role": "user", "content": c})
    return dict(source=source, task_id=task_id, traj_id=traj_id, tools=tools, messages=out)


def issue_key(msgs):
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    m = re.search(r"<(github_issue|pr_description|issue_description)>(.*?)</\1>", u, re.S)
    return (m.group(2) if m else u[:4000]).strip()


def load_r2e_ps_map():
    """problem_statement -> R2E-Gym-Subset task id (repo@commit)."""
    mp = {}
    for f in sorted(glob.glob(f"{SRC}/r2egym_subset/data/*.parquet")):
        for r in pq.read_table(f, columns=["repo_name", "commit_hash", "problem_statement"]).to_pylist():
            mp[norm_ps(r["problem_statement"])] = f"r2e:{r['repo_name']}@{r['commit_hash'][:12]}"
    return mp


def norm_ps(s):
    s = re.sub(r"\[/?ISSUE\]", "", s or "")
    return re.sub(r"\s+", " ", s).strip()[:2000]


def iter_r2e_sft(limit=None):
    mp = load_r2e_ps_map()
    rows = pq.read_table(f"{SRC}/r2egym_sft_traj/data/train-00000-of-00001.parquet").to_pylist()
    for i, r in enumerate(rows[:limit]):
        k = issue_key(r["messages"])
        tid = mp.get(norm_ps(k), "r2e:ps-" + tid_hash(norm_ps(k)))
        yield conv_fn_text(r["messages"], "r2egym_sft_traj", tid, f"r2esft-{i}")


def iter_deepswe_kimi(limit=None):
    mp = load_r2e_ps_map()
    f = glob.glob(f"{SRC}/deepswe_kimik2_traj/*.jsonl")[0]
    for i, l in enumerate(open(f)):
        if limit and i >= limit:
            break
        msgs = json.loads(l)["messages"]
        k = issue_key(msgs)
        tid = mp.get(norm_ps(k), "r2e:ps-" + tid_hash(norm_ps(k)))
        yield conv_fn_text(msgs, "deepswe_kimik2_traj", tid, f"kimik2-{i}")


def load_swegym_ps_map():
    rows = pq.read_table(f"{SRC}/swegym/data/train-00000-of-00001.parquet", columns=["instance_id", "problem_statement"]).to_pylist()
    return {norm_ps(r["problem_statement"]): r["instance_id"] for r in rows}


def iter_swegym_oh(limit=None):
    mp = load_swegym_ps_map()
    rows = pq.read_table(f"{SRC}/swegym_oh_sft/data/train.success.oss-00000-of-00001.parquet").to_pylist()
    for i, r in enumerate(rows[:limit]):
        k = issue_key(r["messages"])
        tid = mp.get(norm_ps(k), "swegym:ps-" + tid_hash(norm_ps(k)))
        yield conv_fn_text(r["messages"], "swegym_oh_sft", tid, f"swegymoh-{i}")

# ---------------------------------------------------------------- SWE-agent text (nebius)
SWEA_TOOLS = [
    fn("bash", "Run one command in the SWE-agent shell (bash plus the special interface commands described in the system prompt: open, goto, scroll_up/down, create, edit, search_dir, search_file, find_file).",
       {"command": {"type": "string", "description": "The command to run."}}, ["command"]),
    fn("submit", "Submit the current solution (the repository diff).", {}, []),
]
_CODE = re.compile(r"```(?:\w+)?\n(.*?)```", re.S)


def iter_nebius_sweagent(limit=None, only_target=False):
    n = 0
    for f in sorted(glob.glob(f"{SRC}/nebius_sweagent/data/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if only_target and not r["target"]:
                continue
            t = r["trajectory"]
            sysm = t[0]["system_prompt"] or t[0]["text"] or ""
            sysm = sysm.split("RESPONSE FORMAT:")[0].rstrip() + "\n\nUse the bash tool to run exactly one command per turn (the special commands above are available in the shell); call submit when done."
            out = [{"role": "system", "content": sysm}]
            for m in t[1:]:
                if m["role"] == "ai":
                    txt = m["text"] or ""
                    blocks = _CODE.findall(txt)
                    if blocks:
                        cmd = blocks[-1].strip()
                        content = txt[:txt.rfind("```", 0, txt.rfind("```"))].replace("DISCUSSION\n", "").strip()
                        call = {"type": "function", "function": {"name": "submit", "arguments": {}}} if cmd == "submit" else \
                               {"type": "function", "function": {"name": "bash", "arguments": {"command": cmd}}}
                        out.append(amsg(content, [call]))
                    else:
                        out.append(amsg(txt.strip()))
                else:
                    role = "tool" if (out[-1]["role"] == "assistant" and out[-1].get("tool_calls")) else "user"
                    out.append({"role": role, "content": m["text"] or ""})
            yield dict(source="nebius_sweagent", task_id=r["instance_id"], traj_id=f"neb-{r['instance_id']}-{n}",
                       tools=SWEA_TOOLS, messages=out)
            n += 1
            if limit and n >= limit:
                return

# ---------------------------------------------------------------- native tool_calls (SWE-smith tool split, nebius rebench OpenHands)
SWESMITH_TOOLS = [
    fn("bash", "Run commands in a bash shell.", {"command": {"type": "string", "description": "The bash command to run."}}, ["command"]),
    fn("str_replace_editor", "Custom editing tool for viewing, creating and editing files. Commands: view, create, str_replace, insert, undo_edit.",
       {"command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"], "description": "The command to run."},
        "path": {"type": "string", "description": "Absolute path to file or directory."},
        "file_text": {"type": "string", "description": "Content for create."},
        "old_str": {"type": "string", "description": "String to replace (str_replace)."},
        "new_str": {"type": "string", "description": "Replacement / inserted string."},
        "insert_line": {"type": "integer", "description": "Line after which to insert."},
        "view_range": {"type": "array", "items": {"type": "integer"}, "description": "Line range for view."}},
       ["command", "path"]),
    fn("submit", "Submits the current file changes.", {}, []),
]


def _text(c):
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def _calls(tcs):
    out = []
    for tc in tcs or []:
        f = tc["function"]
        a = f["arguments"]
        if isinstance(a, str):
            try:
                a = json.loads(a) if a.strip() else {}
            except Exception:
                a = {"raw": a}
        if not isinstance(a, dict):
            a = {"value": a}
        out.append({"type": "function", "id": tc.get("id"), "function": {"name": f["name"], "arguments": a}})
    return out


def native_msgs(msgs):
    out = []
    for m in msgs:
        role = m["role"]
        if role == "assistant":
            r, c = split_think(_text(m.get("content")))
            out.append(amsg(c, _calls(m.get("tool_calls")), r))
        elif role == "tool":
            ids = m.get("tool_call_ids") or [m.get("tool_call_id")]
            out.append({"role": "tool", "content": _text(m.get("content")), "tool_call_id": ids[0] if ids else None})
        else:
            out.append({"role": role, "content": _text(m.get("content"))})
    return out


def iter_swesmith(limit=None):
    n = 0
    for f in sorted(glob.glob(f"{SRC}/swesmith_traj/data/tool-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            yield dict(source="swesmith_traj", task_id=r["instance_id"], traj_id=r["traj_id"], tools=SWESMITH_TOOLS,
                       messages=native_msgs(json.loads(r["messages"])))
            n += 1
            if limit and n >= limit:
                return


def _clean_tools(tools):
    """Drop null-valued keys produced by parquet struct unification."""
    def cl(x):
        if isinstance(x, dict):
            return {k: cl(v) for k, v in x.items() if v is not None}
        if isinstance(x, list):
            return [cl(v) for v in x]
        return x
    return cl(tools)


def iter_rebench_oh(limit=None, per_instance=1, seed=0):
    import random
    pf = pq.ParquetFile(f"{SRC}/nebius_rebench_oh/trajectories.parquet")
    ids = pq.read_table(pf.reader if False else f"{SRC}/nebius_rebench_oh/trajectories.parquet", columns=["instance_id"]).column(0).to_pylist()
    rng = random.Random(seed)
    by = {}
    for i, x in enumerate(ids):
        by.setdefault(x, []).append(i)
    keep = set()
    for x, l in by.items():
        rng.shuffle(l)
        keep.update(l[:per_instance])
    keep = sorted(keep)
    if limit:
        keep = sorted(rng.sample(keep, min(limit, len(keep))))
    ks = set(keep)
    off = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["trajectory_id", "instance_id", "trajectory", "tools"])
        nr = t.num_rows
        sel = [i - off for i in keep if off <= i < off + nr]
        off += nr
        if not sel:
            continue
        for r in t.take(sel).to_pylist():
            yield dict(source="nebius_rebench_oh", task_id=r["instance_id"], traj_id=r["trajectory_id"],
                       tools=_clean_tools(r["tools"]), messages=native_msgs(r["trajectory"]))

# ---------------------------------------------------------------- Terminus-2 (Terminal-Bench 2 traces)
TERMINUS_TOOLS = [
    fn("bash_command", "Send keystrokes to the tmux terminal. Keystrokes are sent verbatim; end commands with \\n; tmux escapes like C-c are allowed.",
       {"keystrokes": {"type": "string", "description": "Exact keystrokes to send."},
        "duration": {"type": "number", "description": "Seconds to wait before returning the screen (default 1.0)."}}, ["keystrokes"]),
    fn("mark_task_complete", "Mark the task as complete (the final terminal state is graded).", {}, []),
]
TERMINUS_SYS = ("You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task "
                "description and the current terminal screen. Use the tools to send commands to the terminal; each response should "
                "analyse the current state, state a plan, and then call tools. Call mark_task_complete when the task is done.")


def terminus_user(first):
    i = first.find("Task Description:")
    return first[i:] if i >= 0 else first


_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def terminus_assistant(txt):
    r, c = split_think(txt)
    calls = []
    for j in _TC.findall(c):
        try:
            d = json.loads(j)
            calls.append({"type": "function", "function": {"name": d.get("name", "bash_command"), "arguments": d.get("arguments", {})}})
        except Exception:
            pass
    if calls:
        return amsg(c[:c.find("<tool_call>")].strip(), calls, r)
    s, e = c.find("{"), c.rfind("}")
    try:
        d = json.loads(c[s:e + 1])
        vis = "\n".join(f"{k.capitalize()}: {d[k]}" for k in ("analysis", "plan") if d.get(k))
        for cmd in d.get("commands") or []:
            a = {"keystrokes": cmd.get("keystrokes", "")}
            if "duration" in cmd:
                a["duration"] = cmd["duration"]
            calls.append({"type": "function", "function": {"name": "bash_command", "arguments": a}})
        if d.get("task_complete"):
            calls.append({"type": "function", "function": {"name": "mark_task_complete", "arguments": {}}})
        return amsg(vis, calls, r)
    except Exception:
        return amsg(c.strip(), None, r)


def iter_tb2_traces(sub):
    f = glob.glob(f"{SRC}/{sub}/data/*.parquet")[0]
    for r in pq.read_table(f).to_pylist():
        conv = r["conversations"]
        if r["agent"] == "swe-agent":  # DeepSWE-Preview on TB2 via R2E scaffold (system text in first user msg)
            first = conv[0]["content"]
            msgs = [{"role": "system", "content": first.split("<IMPORTANT>")[0] if "BEGIN FUNCTION" in first else first}]
            tools = parse_text_tools(first)
            j = first.find("</IMPORTANT>")
            msgs = [{"role": "system", "content": strip_text_tool_block(first)}]
            task = first[j + len("</IMPORTANT>"):].strip() if j >= 0 else ""
            if task:
                msgs.append({"role": "user", "content": task})
            for m in conv[1:]:
                if m["role"] == "assistant":
                    rr, c = split_think(m["content"] or "")
                    content, calls = parse_fn_calls(c, tools)
                    msgs.append(amsg(content, calls, rr))
                else:
                    role = "tool" if msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls") else "user"
                    msgs.append({"role": role, "content": _OBS.sub(lambda m: m.group(1) or "", m["content"] or "", count=1)})
            yield dict(source=sub, task_id="tb:" + r["task"], traj_id=r["trial_name"], tools=tools, messages=msgs)
            continue
        msgs = [{"role": "system", "content": TERMINUS_SYS}, {"role": "user", "content": terminus_user(conv[0]["content"])}]
        for m in conv[1:]:
            if m["role"] == "assistant":
                msgs.append(terminus_assistant(m["content"] or ""))
            else:
                role = "tool" if msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls") else "user"
                msgs.append({"role": role, "content": m["content"] or ""})
        yield dict(source=sub, task_id="tb:" + r["task"], traj_id=r["trial_name"], tools=TERMINUS_TOOLS, messages=msgs)

# ---------------------------------------------------------------- task-statement-only (turn 1)
R2E_SYS = ("You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve "
           "certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.")
R2E_USER = """Consider the following github issue:
<github_issue>
{issue}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required
7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files.
"""
_R2E_TOOLS = None


def r2e_tools():
    global _R2E_TOOLS
    if _R2E_TOOLS is None:
        m = pq.read_table(f"{SRC}/r2egym_sft_traj/data/train-00000-of-00001.parquet").slice(0, 1).to_pylist()[0]["messages"]
        _R2E_TOOLS = parse_text_tools(m[0]["content"])
    return _R2E_TOOLS


def iter_r2e_subset_t1():
    for f in sorted(glob.glob(f"{SRC}/r2egym_subset/data/*.parquet")):
        for r in pq.read_table(f, columns=["repo_name", "commit_hash", "problem_statement"]).to_pylist():
            ps = re.sub(r"\[/?ISSUE\]", "", r["problem_statement"]).strip()
            yield dict(source="r2egym_subset_t1", task_id=f"r2e:{r['repo_name']}@{r['commit_hash'][:12]}", traj_id=None, tools=r2e_tools(),
                       messages=[{"role": "system", "content": R2E_SYS}, {"role": "user", "content": R2E_USER.format(issue=ps)}])


def iter_swebv_t1():
    for r in pq.read_table(f"{SRC}/swebv/data/test-00000-of-00001.parquet", columns=["instance_id", "problem_statement"]).to_pylist():
        yield dict(source="swebv_t1", task_id=r["instance_id"], traj_id=None, tools=r2e_tools(),
                   messages=[{"role": "system", "content": R2E_SYS}, {"role": "user", "content": R2E_USER.format(issue=r["problem_statement"].strip())}])


def iter_tb_t1(which):
    """which: tb21_registry (harborframework/terminal-bench-2.1) or tb_harbor_gh (harbor-framework/terminal-bench tasks/)."""
    base = f"{SRC}/tb21_registry/tasks" if which == "tb21_registry" else f"{SRC}/tb_harbor_gh/tasks"
    for name in sorted(os.listdir(base)):
        p = f"{base}/{name}/instruction.md"
        if not os.path.exists(p):
            continue
        instr = re.sub(r"<!--\s*harbor-canary.*?-->\s*", "", open(p).read()).strip()  # drop canary comment
        user = f"Task Description:\n{instr}\n\nCurrent terminal state:\nCurrent Terminal Screen:\nroot@sandbox:/app# "
        yield dict(source=which + "_t1", task_id="tb:" + name, traj_id=None, tools=TERMINUS_TOOLS,
                   messages=[{"role": "system", "content": TERMINUS_SYS}, {"role": "user", "content": user}])


def iter_math500():
    for l in open(f"{SRC}/math500/test.jsonl"):
        r = json.loads(l)
        yield dict(source="math500", task_id="math500:" + r["unique_id"], traj_id=None, tools=None,
                   messages=[{"role": "user", "content": r["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}])


TRAJ_SOURCES = {
    "r2egym_sft_traj": iter_r2e_sft, "deepswe_kimik2_traj": iter_deepswe_kimi, "swegym_oh_sft": iter_swegym_oh,
    "nebius_sweagent": iter_nebius_sweagent, "swesmith_traj": iter_swesmith, "nebius_rebench_oh": iter_rebench_oh,
    **{s: (lambda limit=None, s=s: iter_tb2_traces(s)) for s in ["tb2_gpt5_traj", "tb2_sonnet45_traj", "tb2_glm47_traj", "tb2_kimik25_traj", "tb2_deepswe_traj"]},
}
T1_SOURCES = {"r2egym_subset_t1": iter_r2e_subset_t1, "swebv_t1": iter_swebv_t1,
              "tb21_registry_t1": lambda: iter_tb_t1("tb21_registry"), "tb_harbor_gh_t1": lambda: iter_tb_t1("tb_harbor_gh"),
              "math500": iter_math500}
