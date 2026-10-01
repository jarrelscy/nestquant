"""T34: build the bf16conf corpus (token ids + window map) that nq_e2e reads for the BF16-teacher panel.
Source: brandonmusic/GLM-5.3-BF16-full-logits, reference-full-panel/calibration/panel-v1 (confirmation lane, 64 x 2048).
  python build_corpus.py TEACHER_DIR OUT_DIR   ->  OUT_DIR/bf16conf.npy (int32 [64*2048]), OUT_DIR/bf16conf.map.npz"""
import json, os, sys
import numpy as np

T, O = sys.argv[1], sys.argv[2]
P = f"{T}/reference-full-panel/calibration/panel-v1"
win = {w["window_id"]: w for w in json.load(open(f"{P}/panel.json"))["windows"]}
names = [f"confirmation-{i:04d}" for i in range(64)]
tok = np.concatenate([np.load(f"{P}/arrays/{n}.tokens.npy").astype(np.int32) for n in names])
assert tok.shape == (64 * 2048,), tok.shape
os.makedirs(O, exist_ok=True)
np.save(f"{O}/bf16conf.npy", tok)
np.savez(f"{O}/bf16conf.map.npz", chain=np.arange(64), task=np.arange(64), names=np.array(names),
         domain=np.array([win[n]["domain"] for n in names]))   # chain=arange: every predictor starts cold per window
print("wrote", O, tok.shape)
