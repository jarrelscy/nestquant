"""Inverse of the GLM-5.3 chat template (zai-org/GLM-5.3 chat_template.jinja, sha256 0e2b6268...) on token ids.

parse(ids, at_start) -> Parsed. Works at the id level (role / think / tool tags are single special ids), decoding
only the text between special ids, so content can never be confused with markup.

Partial documents are supported because the packed corpus holds tail-cut traces (fragments that start mid-message)
and docs cut right after a boundary / truncated runaways (fragments that end mid-message):
  head: None, or dict(role, field) describing the leading partial message (msgs[0]).
  tail: None, or dict(kind) describing how the last message is cut:
        'field'  -> msgs[-1][field] was cut mid-text (field in reasoning_content / content)
        'turn'   -> last assistant message is complete up to its last part but the turn was not ended
                    (cut right after </think> or after a complete </tool_call>)
        'tool_complete' -> last tool response complete, no following role token (nothing to strip)
  end_tok: the doc ends with the GLM end-of-turn token (<|user|>/<|observation|>/<|endoftext|>) after an assistant.
"""
import json
from dataclasses import dataclass, field
from typing import Optional

EOT, GMASK, SOP, SYSTEM, USER, ASSISTANT, OBSERVATION = 154820, 154822, 154824, 154826, 154827, 154828, 154829
THINK, ETHINK = 154841, 154842
TC, ETC, TR, ETR, AK, EAK, AV, EAV = 154843, 154844, 154845, 154846, 154847, 154848, 154849, 154850
ROLES = {SYSTEM: "system", USER: "user", ASSISTANT: "assistant", OBSERVATION: "tool", EOT: "eot"}
END_ROLES = (USER, OBSERVATION, EOT)
TOOLS_HEAD = ("\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
TOOLS_TAIL = ("\n</tools>\n\nFor each function call, output the function name and arguments within the following XML "
              "format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
              "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")


class ParseError(Exception):
    pass


@dataclass
class Parsed:
    msgs: list
    tools: Optional[list] = None
    effort: Optional[str] = None          # GLM "Reasoning Effort: X" (lowercased), None if not at doc start
    head: Optional[dict] = None
    tail: Optional[dict] = None
    end_tok: Optional[int] = None
    notes: list = field(default_factory=list)


def _find(ids, x, start=0):
    try:
        return ids.index(x, start)
    except ValueError:
        return -1


class Parser:
    def __init__(self, tok):
        self.tok = tok  # tokenizers.Tokenizer (GLM)

    def dec(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False) if ids else ""

    # ------------------------------------------------------------------ assistant body (after <|assistant|>)
    def _arg_value(self, raw, ptype):
        if ptype == "string":
            return raw
        try:
            v = json.loads(raw)
        except Exception:
            return raw
        if isinstance(v, str) or json.dumps(v, ensure_ascii=False) != raw:
            return raw  # the template prints strings verbatim and non-strings via tojson; keep exact text
        return v

    def _tool_calls(self, ids, types):
        """ids = everything after </think> + content. Returns (content_ids, calls, partial_call_dropped)."""
        k = _find(ids, TC)
        content_ids = ids if k < 0 else ids[:k]
        calls, partial = [], False
        while k >= 0:
            e = _find(ids, ETC, k)
            if e < 0:
                partial = True
                break
            body = ids[k + 1:e]
            a = _find(body, AK)
            name = self.dec(body if a < 0 else body[:a])
            args = {}
            ptypes = types.get(name, {})
            while a >= 0:
                b = _find(body, EAK, a); c = _find(body, AV, b); d = _find(body, EAV, c)
                if min(b, c, d) < 0 or c != b + 1:
                    raise ParseError("bad arg structure")
                key = self.dec(body[a + 1:b])
                args[key] = self._arg_value(self.dec(body[c + 1:d]), ptypes.get(key, {}).get("type"))
                a = _find(body, AK, d)
                if (a if a >= 0 else len(body)) != d + 1:
                    raise ParseError("junk between args")
            calls.append(dict(type="function", function=dict(name=name, arguments=args)))
            k = e + 1
            if k < len(ids):
                if ids[k] != TC:
                    raise ParseError("text after tool call")
            else:
                k = -1
        for t in (THINK, ETHINK, TR, ETR, AK, EAK, AV, EAV, ETC):
            if t in content_ids:
                raise ParseError(f"stray tag {t} in assistant content")
        return content_ids, calls, partial

    def _assistant(self, body, last, types, head_reasoning=False):
        """Returns (msg, tail) for an assistant block body (ids after <|assistant|>)."""
        if head_reasoning:
            e = _find(body, ETHINK)
            r_ids, rest = body[:e], body[e + 1:]
        else:
            if not body or body[0] != THINK:
                raise ParseError("assistant without <think>")
            e = _find(body, ETHINK)
            if e < 0:  # cut inside reasoning
                if not last:
                    raise ParseError("unclosed think")
                if THINK in body[1:]:
                    raise ParseError("nested think")
                return dict(role="assistant", reasoning_content=self.dec(body[1:]), content=""), dict(kind="field", field="reasoning_content")
            r_ids, rest = body[1:e], body[e + 1:]
        if THINK in r_ids:
            raise ParseError("nested think")
        content_ids, calls, partial = self._tool_calls(rest, types)
        msg = dict(role="assistant", reasoning_content=self.dec(r_ids), content=self.dec(content_ids))
        if calls:
            msg["tool_calls"] = calls
        tail = None
        if partial:
            if not last:
                raise ParseError("unclosed tool call")
            tail = dict(kind="turn", dropped_partial_call=True) if calls else dict(kind="field", field="content", dropped_partial_call=True)
        elif last:
            tail = dict(kind="turn") if (calls or not content_ids) else dict(kind="field", field="content")
        return msg, tail

    def _observation(self, body, last, head=False):
        """Tool responses. Returns (msgs, tail, head_partial)."""
        out, i, head_partial = [], 0, False
        if head and body and body[0] != TR:
            e = _find(body, ETR)
            if e < 0:
                raise ParseError("head observation without </tool_response>")
            out.append(dict(role="tool", content=self.dec(body[:e]))); i = e + 1; head_partial = True
        tail = None
        while i < len(body):
            if body[i] != TR:
                raise ParseError("text outside <tool_response>")
            e = _find(body, ETR, i)
            if e < 0:
                if not last:
                    raise ParseError("unclosed tool_response")
                out.append(dict(role="tool", content=self.dec(body[i + 1:])))
                tail = dict(kind="field", field="content")
                break
            if TR in body[i + 1:e]:
                raise ParseError("nested tool_response")
            out.append(dict(role="tool", content=self.dec(body[i + 1:e]))); i = e + 1
        if last and tail is None and out:
            tail = dict(kind="tool_complete")
        if not out and not head:
            raise ParseError("empty observation")
        return out, tail, head_partial

    # ------------------------------------------------------------------ document
    def parse(self, ids, at_start=True):
        ids = list(ids)
        P = Parsed(msgs=[])
        if at_start:
            if ids[:2] != [GMASK, SOP]:
                raise ParseError("no [gMASK]<sop>")
            ids = ids[2:]
        # split into blocks at role ids
        blocks, cur = [], [None, []]
        for t in ids:
            if t in ROLES:
                blocks.append(cur); cur = [t, []]
            else:
                cur[1].append(t)
        blocks.append(cur)
        if blocks[0][0] is None and not blocks[0][1]:
            blocks.pop(0)
        if at_start:
            if not blocks or blocks[0][0] != SYSTEM:
                raise ParseError("no effort system block")
            s = self.dec(blocks[0][1])
            if not s.startswith("Reasoning Effort: "):
                raise ParseError("bad effort block")
            P.effort = s[len("Reasoning Effort: "):].lower()
            blocks.pop(0)
            if blocks and blocks[0][0] == SYSTEM:
                s = self.dec(blocks[0][1])
                if s.startswith(TOOLS_HEAD):
                    if not s.endswith(TOOLS_TAIL):
                        raise ParseError("bad tools block")
                    body = s[len(TOOLS_HEAD):-len(TOOLS_TAIL)]
                    P.tools = [json.loads(l) for l in body.split("\n") if l.strip()]
                    blocks.pop(0)
        types = {}
        for t in P.tools or []:
            f = t.get("function", t)
            types[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
        # end-of-turn token: an empty USER/OBSERVATION/EOT block right after an assistant block, at doc end
        if len(blocks) >= 2 and blocks[-1][0] in END_ROLES and not blocks[-1][1] and blocks[-2][0] == ASSISTANT:
            P.end_tok = blocks[-1][0]
            blocks.pop()
        elif blocks and blocks[-1][0] == EOT:
            raise ParseError("eot not after assistant")
        prev_calls = None
        for bi, (role, body) in enumerate(blocks):
            last = bi == len(blocks) - 1 and P.end_tok is None
            if role is None:  # leading partial message of a mid-doc fragment
                nxt = blocks[1][0] if len(blocks) > 1 else None
                # fragment starts exactly at a structural tag: whole message(s), only the role token is missing
                if body[0] in (THINK, ETHINK, TC):
                    if body[0] == THINK:
                        pre, b2 = "<|assistant|>", body
                    elif body[0] == ETHINK:
                        pre, b2 = "<|assistant|><think>", [THINK] + body
                    else:
                        pre, b2 = "<|assistant|><think></think>", [THINK, ETHINK] + body
                    m, tail = self._assistant(b2, last, types)
                    P.head = dict(role="assistant", field=None, glm_prefix=pre)
                    P.msgs.append(m); P.tail = tail if last else P.tail
                    prev_calls = m.get("tool_calls")
                    continue
                if body[0] == TR:
                    ms, tail, _ = self._observation(body, last)
                    P.head = dict(role="tool", field=None, glm_prefix="<|observation|>")
                    P.msgs.extend(ms); P.tail = tail if last else P.tail
                    continue
                if ETHINK in body:
                    m, tail = self._assistant(body, last, types, head_reasoning=True)
                    P.head = dict(role="assistant", field="reasoning_content")
                    P.msgs.append(m); P.tail = tail if last else P.tail
                    prev_calls = m.get("tool_calls")
                elif TR in body or ETR in body:
                    ms, tail, hp = self._observation(body, last, head=True)
                    P.head = dict(role="tool", field="content") if hp else None
                    P.msgs.extend(ms); P.tail = tail if last else P.tail
                elif TC in body or ETC in body or AK in body:
                    k = _find(body, ETC)
                    if k >= 0 and (TC not in body or body.index(TC) > k):
                        P.notes.append(f"head_dropped_partial_call:{k + 1}")
                        body = body[k + 1:]
                    m, tail = self._assistant([THINK, ETHINK] + body, last, types)
                    m["reasoning_content"] = ""
                    P.head = dict(role="assistant", field="content", no_reasoning=True)
                    P.msgs.append(m); P.tail = tail if last else P.tail
                    prev_calls = m.get("tool_calls")
                else:
                    if nxt in END_ROLES:
                        r = "assistant"
                    elif nxt is None:
                        raise ParseError("head-only fragment without markers")
                    else:
                        r = "user"
                    if r == "assistant":
                        m = dict(role="assistant", reasoning_content="", content=self.dec(body))
                        P.head = dict(role="assistant", field="content", no_reasoning=True)
                    else:
                        m = dict(role="user", content=self.dec(body)); P.head = dict(role="user", field="content")
                    P.msgs.append(m)
                continue
            if role == SYSTEM:
                P.msgs.append(dict(role="system", content=self.dec(body)))
                if last:
                    P.tail = dict(kind="field", field="content")
            elif role == USER:
                P.msgs.append(dict(role="user", content=self.dec(body)))
                if last:
                    P.tail = dict(kind="field", field="content")
            elif role == ASSISTANT:
                m, tail = self._assistant(body, last, types)
                P.msgs.append(m); prev_calls = m.get("tool_calls")
                if last:
                    P.tail = tail
            elif role == OBSERVATION:
                ms, tail, _ = self._observation(body, last)
                if prev_calls and blocks[bi - 1][0] == ASSISTANT:
                    for j, m in enumerate(ms):  # GLM drops ids; name tool results positionally for Inkling headers
                        if j < len(prev_calls):
                            m["name"] = prev_calls[j]["function"]["name"]
                P.msgs.extend(ms)
                if last:
                    P.tail = tail
            elif role == EOT:
                raise ParseError("eot mid-doc")
        if not P.msgs:
            raise ParseError("no messages")
        return P
