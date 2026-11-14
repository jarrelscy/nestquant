"""split.py LOG OUTDIR: write each generated nq_matvec source block of LOG to OUTDIR/<i>_<name>__<tmpl>.metal"""
import os
import re
import sys

text = open(sys.argv[1]).read()
os.makedirs(sys.argv[2], exist_ok=True)
for i, m in enumerate(re.finditer(r"Generated source code for `([^`]*)`:\n```\n(.*?)\n```", text, re.S)):
    src = "#include <metal_stdlib>\nusing namespace metal;\ntypedef half float16_t;\n" + m.group(2)
    t = re.search(r'<(\d[\d, ]*)>\)', src)
    tag = t.group(1).replace(', ', '_') if t else ''
    open(os.path.join(sys.argv[2], f'{i:02d}_{m.group(1)}__{tag}.metal'), 'w').write(src)
