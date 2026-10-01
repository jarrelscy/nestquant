"""T34: derive the k0 serve manifest (no fixed experts, all 4-bit slots floating) from the published TP4 manifest.
k0 = serving/tp4/manifest.json with default_allocation emptied and floating_default = fixed-26 + floating_default-51,
in that order (jF starts each request from this set; use n_float=77, or larger n_float with the same file).
  python mk_k0_manifest.py HF_REPO_DIR/serving/tp4/manifest.json OUT.json"""
import json, sys

m = json.load(open(sys.argv[1]))
fd = {L: m["default_allocation"][L] + m["floating_default"][L] for L in m["floating_default"]}
m["default_allocation"] = {L: [] for L in m["default_allocation"]}
m["floating_default"] = fd
m["notes"] = "T32 k=0 arm: no fixed set, floating_default = fixed26 + floating_default51 (use n_float=77). OFFLINE EVAL ONLY."
json.dump(m, open(sys.argv[2], "w"))
