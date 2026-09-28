"""Shared GLM-5.3 native-format helpers (thread 21).

ChatML-text -> GLM chat-template conversion, boundary detection on GLM token ids.
"""
import json, re
import numpy as np

TOK_DIR = "/tmp/nestquant/src/glm53-fp8"
EOT, GMASK, SOP, SYSTEM, USER, ASSISTANT, OBSERVATION = 154820, 154822, 154824, 154826, 154827, 154828, 154829
THINK, ETHINK = 154841, 154842
ROLE_TOKS = {SYSTEM, USER, ASSISTANT, OBSERVATION, EOT}
END_TOKS = (USER, OBSERVATION, EOT)
NB = 32  # boundary horizon

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK_DIR)
    return _tok


BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)
TC_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)
TOOLS_SYS_RE = re.compile(r"^You are provided with the following tools:\s*<tools>\n(.*?)\n?</tools>\s*$", re.S)


class ParseError(Exception):
    pass


def _pval(v):
    # Qwen3-coder XML convention: one optional newline on either side of the value
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _loads_any(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        import ast
        try:
            return ast.literal_eval(s)
        except Exception as e:
            raise ParseError(f"bad json: {e}")


def parse_chatml(text):
    """ChatML text -> (messages, tools). Raises ParseError if anything is left unparsed."""
    blocks = BLOCK_RE.findall(text)
    if "".join(f"<|im_start|>{r}\n{x}<|im_end|>" for r, x in blocks) != text:
        raise ParseError("text outside ChatML blocks")
    msgs, tools = [], None
    for role, x in blocks:
        if role == "system":
            m = TOOLS_SYS_RE.match(x)
            if m:
                if tools is not None:
                    raise ParseError("two tool blocks")
                tools = [json.loads(l) for l in m.group(1).split("\n") if l.strip()]
                continue
            if "<tools>" in x:  # Hermes / function-composing style: tool list inline in the system text
                mt = re.search(r"<tools>(.*?)</tools>", x, re.S)
                if not mt or tools is not None:
                    raise ParseError("unrecognised tools system block")
                try:
                    tools = _loads_any(mt.group(1).strip())
                    if isinstance(tools, dict):
                        tools = [tools]
                    x = (x[:mt.start()] + x[mt.end():])
                    x = re.sub(r"\n{3,}", "\n\n", x).strip()
                except ParseError:  # unparseable inline schema: keep the system text verbatim
                    tools = None
            msgs.append(dict(role="system", content=x))
        elif role == "user":
            msgs.append(dict(role="user", content=x))
        elif role == "tool":
            msgs.append(dict(role="tool", content=x))
        elif role == "assistant":
            m = re.match(r"<think>(.*?)</think>", x, re.S)
            if not m:
                raise ParseError("assistant without leading think")
            reasoning, rest = m.group(1), x[m.end():]
            calls = []
            for name, body in TC_RE.findall(rest):
                args = {k: _pval(v) for k, v in PARAM_RE.findall(body)}
                if PARAM_RE.sub("", body).strip():
                    raise ParseError("junk inside function body")
                calls.append(dict(type="function", function=dict(name=name.strip(), arguments=args)))
            content = TC_RE.sub("", rest)
            if not calls and "<tool_call>" in content:  # JSON-style calls: <tool_call>[{name, arguments}, ...]</tool_call>
                for body in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.S):
                    v = _loads_any(body.strip())
                    for c in (v if isinstance(v, list) else [v]):
                        args = c["arguments"]
                        if isinstance(args, str):
                            args = _loads_any(args)
                        calls.append(dict(type="function", function=dict(name=c["name"], arguments=args)))
                content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S)
            if "<think>" in content or "</think>" in content:
                raise ParseError("extra think tags")
            if "<tool_call>" in content or "<function=" in content or "</tool_call>" in content:
                raise ParseError("unparsed tool call")
            msg = dict(role="assistant", content=content, reasoning_content=reasoning)
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        else:
            raise ParseError(f"role {role}")
    return msgs, tools


def render(msgs, tools=None, reasoning_effort=None, final_end=True):
    """GLM chat template render + tokenize. Appends the end-of-turn token GLM emits after a final assistant turn:
    <|observation|> after tool calls, else <|user|> (Together returns matched_stop=154827 for normal answers)."""
    tok = tokenizer()
    kw = {} if reasoning_effort is None else dict(reasoning_effort=reasoning_effort)
    s = tok.apply_chat_template(msgs, tools=tools, tokenize=False, **kw)
    ids = tok.encode(s, add_special_tokens=False)
    if final_end and msgs and msgs[-1]["role"] == "assistant":
        ids.append(OBSERVATION if msgs[-1].get("tool_calls") else USER)
    return s, ids


def boundaries(ids, chat=True):
    """Return (think_end_positions, end_of_turn_positions) as indices of the boundary TOKEN.
    Real think end: </think> inside an assistant turn whose reasoning is non-empty (prev token != <think>).
    End of turn: first role/end token (<|user|>, <|observation|>, <|endoftext|>) after an assistant turn."""
    th, en = [], []
    if not chat:
        return th, en
    in_asst = False
    open_think = False  # only the first </think> closing the turn-opening <think> counts
    for i, t in enumerate(ids):
        if t == ASSISTANT:
            in_asst = True
            open_think = i + 1 < len(ids) and ids[i + 1] == THINK
        elif t in ROLE_TOKS:
            if in_asst and t in END_TOKS:
                en.append(i)
            in_asst = open_think = False
        elif in_asst and open_think and t == ETHINK:
            if ids[i - 1] != THINK:
                th.append(i)
            open_think = False
    return th, en


def bnd_arrays(n, th, en):
    """Per-token int8 distance 1..32 to the next boundary token (position p has d = b - p, i.e. d=1 is the
    token immediately before the boundary; the boundary token itself gets d=0 unless it precedes another one).
    Nearer kind wins, ties -> end, so a row is never counted for both kinds."""
    p = np.arange(n)
    out = []
    for bs in (th, en):
        b = np.asarray(sorted(bs), np.int64)
        if len(b) == 0:
            out.append(np.zeros(n, np.int64)); continue
        k = np.searchsorted(b, p, side="right")  # first boundary strictly after p
        d = np.where(k < len(b), b[np.minimum(k, len(b) - 1)] - p, 0)
        d[d > NB] = 0
        out.append(d)
    dth, den = out
    both = (dth > 0) & (den > 0)
    dth[both & (den <= dth)] = 0
    den[both & (dth > 0)] = 0
    return dth.astype(np.int8), den.astype(np.int8)
