#!/usr/bin/env python3
"""router weights (public model weights, not data): gate W [256,H] fp32, e_score_correction_bias, post-attn norm weight."""
import sys
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import nq_io
m = nq_io.FP8Model("/tmp/nestquant/src/glm53-fp8")
out = {}
for L in range(3, 78):
    p = f"model.layers.{L}"
    out[f"W{L}"] = m.tensor(f"{p}.mlp.gate.weight", "cpu").float().numpy()
    out[f"b{L}"] = m.tensor(f"{p}.mlp.gate.e_score_correction_bias", "cpu").float().numpy()
    out[f"n{L}"] = m.tensor(f"{p}.post_attention_layernorm.weight", "cpu").float().numpy()
np.savez("/tmp/nestquant/33-search/hprobe/router.npz", **out)
print(out["W10"].shape, out["W10"].dtype, out["n10"][:4])
