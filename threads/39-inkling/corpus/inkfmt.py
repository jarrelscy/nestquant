"""Inkling-2-Small native-format helpers: special ids, boundary detection, per-token boundary distance arrays.

Inkling chat template (chat_template.jinja sha256 e9cfd8e7...):
  message      = <|message_ROLE|>[name]<|content_TYPE|> body <|end_message|>
  assistant turn = [thinking msg] [text msg] [tool-call msgs...] <|content_model_end_sampling|>   (= EOS 200006)
Boundaries (index of the boundary TOKEN):
  think end  : the <|end_message|> closing a non-empty <|message_model|><|content_thinking|> message
  end of turn: the <|end_message|> closing the last model message of a turn, i.e. one immediately followed by
               <|content_model_end_sampling|> (pass end_at="sampling" to use the EOS token itself instead)
"""
import numpy as np

ENDOFTEXT = 199999
USER, MODEL, SYSTEM, TOOL = 200000, 200001, 200002, 200003
TEXT, END_SAMPLING, THINKING, END_MESSAGE = 200004, 200006, 200008, 200010
IMAGE, TOOL_ERROR, XML, BEGIN_OF_TEXT, INVOKE_JSON = 200005, 200022, 200024, 200028, 200049
EOS = END_SAMPLING
PAD = ENDOFTEXT  # Inkling defines no pad token; padding positions also carry segment -1
MSG_START = {USER, MODEL, SYSTEM, TOOL}
NB = 32  # boundary horizon, same as glmfmt


def boundaries(ids, chat=True, init=None, end_at="message"):
    """Return (think_end_positions, end_of_turn_positions).
    init: (role_id, content_type_id) of a message already open at ids[0] (fragments that start mid-message); a
    head thinking body is assumed non-empty."""
    th, en = [], []
    if not chat:
        return th, en
    n = len(ids)
    role, ctype, body = (init[0], init[1], 1) if init else (None, None, 0)
    for i, t in enumerate(ids):
        if t in MSG_START:
            role, ctype, body = t, None, 0
        elif t == END_MESSAGE:
            if role == MODEL and ctype == THINKING and body > 0:
                th.append(i)
            if role == MODEL and i + 1 < n and ids[i + 1] == END_SAMPLING:
                en.append(i + 1 if end_at == "sampling" else i)
            role, ctype, body = None, None, 0
        elif t == END_SAMPLING:
            role, ctype, body = None, None, 0
        elif role is not None and ctype is None and t >= 200000:
            ctype = t
        elif ctype is not None:
            body += 1
    return th, en


def bnd_arrays(n, th, en):
    """Identical semantics to glmfmt.bnd_arrays: d = b - p (1..32) to the next boundary b strictly after p,
    nearer kind wins, ties -> end."""
    p = np.arange(n)
    out = []
    for bs in (th, en):
        b = np.asarray(sorted(bs), np.int64)
        if len(b) == 0:
            out.append(np.zeros(n, np.int64)); continue
        k = np.searchsorted(b, p, side="right")
        d = np.where(k < len(b), b[np.minimum(k, len(b) - 1)] - p, 0)
        d[d > NB] = 0
        out.append(d)
    dth, den = out
    both = (dth > 0) & (den > 0)
    dth[both & (den <= dth)] = 0
    den[both & (dth > 0)] = 0
    return dth.astype(np.int8), den.astype(np.int8)
