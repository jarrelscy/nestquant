#!/usr/bin/env python3
"""T32: vocab -> token class table (no corpus data): 0 word, 1 number, 2 code-symbol, 3 other punct, 4 whitespace.
-> /tmp/nestquant/32-gbdt-sal/stats/tokclass.npy (uint8 [vocab])"""
import numpy as np
from tokenizers import Tokenizer

CODE = set("{}[]()<>=;_*/\\|&^%$#@~`+")
tk = Tokenizer.from_file("/tmp/nestquant/src/glm53-fp8/tokenizer.json")
V = tk.get_vocab_size(with_added_tokens=True)
cls = np.zeros(V, np.uint8)
for i in range(V):
    s = tk.decode([i], skip_special_tokens=False).strip()
    if not s:
        cls[i] = 4
    elif any(c.isalpha() for c in s):
        cls[i] = 0
    elif any(c.isdigit() for c in s):
        cls[i] = 1
    elif any(c in CODE for c in s):
        cls[i] = 2
    else:
        cls[i] = 3
np.save("/tmp/nestquant/32-gbdt-sal/stats/tokclass.npy", cls)
print(V, np.bincount(cls, minlength=5))
