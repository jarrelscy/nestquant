#!/usr/bin/env python3
"""T32 SM120 (b) prep: teacher-forced recapture corpus for the decode stretches of the 7 decode-flagged SM120 tasks.
Context rebuild: rows are kept in computed order with re-prefilled context dropped (first occurrence of each 8-gram),
so the task's kept-row stream is the best available approximation of the conversation each decode step saw.  Each
task's kept rows (first CAP rows) are cut into 2048-token windows; windows with >= MIN_DEC decode rows are kept (the
FP8 reference forward then sees <= 2047 tokens of rebuilt context per decode row).
PRIVATE -> $OUT/private/corpora/sm120tf.npy (+ .map.npz: per window task index, first stream row; per task name)."""
import glob
import os

import numpy as np

OUT = "/tmp/nestquant/32-gbdt-sal"
SRC = f"{OUT}/sm120_traces/serving/predictor/traces"
CAP = int(os.environ.get("CAP", "800000"))
MIN_DEC = int(os.environ.get("MIN_DEC", "64"))
SEQ = 2048
toks, wt, wr, names = [], [], [], []
for f in sorted(glob.glob(f"{SRC}/*.npz")):
    n = os.path.basename(f)[:-4]
    if n.startswith("probe-"):
        continue
    z = np.load(f)
    dec = z["dec"]
    if dec.sum() < 256:
        continue
    ti = len(names); names.append(n)
    tok = z["tok"][:CAP]; dec = dec[:CAP]
    for w in range(len(tok) // SEQ):
        if dec[w * SEQ:(w + 1) * SEQ].sum() >= MIN_DEC:
            toks.append(tok[w * SEQ:(w + 1) * SEQ]); wt.append(ti); wr.append(w * SEQ)
    print(n, "windows so far", len(toks), flush=True)
ids = np.concatenate(toks).astype(np.int32)
np.save(f"{OUT}/private/corpora/sm120tf.npy", ids)
np.savez(f"{OUT}/private/corpora/sm120tf.map.npz", task=np.asarray(wt), row0=np.asarray(wr), names=np.asarray(names))
print("windows", len(toks), "tokens", len(ids))
