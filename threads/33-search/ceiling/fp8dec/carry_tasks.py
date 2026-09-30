"""T33l KV-carry measurement, step 1: dec.py --mode tf tasks = one contiguous SM120 stream span per task (PRIVATE).
Span = rows [row0 of the task's first sm120tf window, row0 + SPAN); the sm120tf trace windows fully inside the span were
captured WITHOUT context (each 2048 window alone); dec.py re-captures them WITH the span's preceding rows as context.
  carry_tasks.py OUT [SPAN=40960]   -> OUT/tasks.json, OUT/meta.json (task, a, b, windows [(global win idx, row0)])"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import sm120 as S1  # noqa: E402
OUT = sys.argv[1]; SPAN = int(sys.argv[2]) if len(sys.argv) > 2 else 40960
mp = np.load(S1.TFC)
names = [str(x) for x in mp["names"]]
tasks, meta = [], []
for ti, n in enumerate(names):
    gi = np.nonzero(mp["task"] == ti)[0]
    if not len(gi):
        continue
    z = np.load(f"{S1.SRC}/{n}.npz")
    a = int(mp["row0"][gi[0]]); b = min(a + SPAN, len(z["tok"]))
    w = [(int(g), int(mp["row0"][g])) for g in gi if mp["row0"][g] + 2048 <= b]
    tasks.append(dict(id=f"carry/{n}", prompt=z["tok"][a:b].astype(int).tolist(), tf=[]))
    meta.append(dict(task=n, ti=ti, a=a, b=b, windows=w))
    print(n, a, b, len(w), "windows", flush=True)
json.dump(tasks, open(f"{OUT}/tasks.json", "w"))
json.dump(meta, open(f"{OUT}/meta.json", "w"))
