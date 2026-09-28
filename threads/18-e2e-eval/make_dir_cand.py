"""Smoke helper: write a dequantised-safetensors candidate dir (RTN) for some (layer, experts),
to exercise the dir: interface + reference fallback.  Usage: make_dir_cand.py OUT LAYER E0 E1 BITS"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_io, quantisers
from safetensors.torch import save_file
out, L, e0, e1, bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out, exist_ok=True)
m = nq_io.FP8Model(os.environ.get("NQ_FP8", "/tmp/nestquant/src/glm53-fp8"))
q = quantisers.RTN(bits=bits, group=128); q.dev = "cpu"
sd = {}
for e in range(e0, e1):
    W = q.expert(L, e, lambda: m.expert(L, e, "cpu"))
    for p, w in W.items():
        sd[f"model.layers.{L}.mlp.experts.{e}.{p}.weight"] = w.contiguous()
save_file(sd, f"{out}/layer_{L:03d}.safetensors")
print("wrote", len(sd), "tensors")
