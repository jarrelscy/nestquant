#!/usr/bin/env python
"""T25: build repo-top/config.json (text), config.mm.json (glm5v graft layout)
and config_text_diff.json (ours vs Vision repo text_config, every key)."""
import copy, json, os

SRC_CFG = "/tmp/nestquant/src/glm53-fp8/config.json"
VIS_CFG = "/tmp/nestquant/vision-graft/config.json"
OUT = "/tmp/nestquant/25-campaign/repo-top"
# same list as graft_vision53.py TOP_KEYS
TOP_KEYS = ["architectures", "model_type", "ignore_index", "media_placeholder_token_id",
            "pad_token_id", "eos_token_id", "use_unified_vision_chunk",
            "video_placeholder", "encoder_only", "language_only",
            "tie_word_embeddings", "dtype"]

src = json.load(open(SRC_CFG))
vis = json.load(open(VIS_CFG))

cfg = copy.deepcopy(src)
qc = cfg["quantization_config"]  # FP8 fields + modules_to_not_convert kept verbatim
qc["nestquant"] = {
    "format": "nestquant-v1",
    "routed_experts": "layers/L{L}/tp{s}.safetensors",
    "layers": "3..77",
    "manifest": "layers/L{L}/manifest.json",
    "tp": 8,
    "levels": [2, 4],
    "note": ("routed experts (model.layers.{3..77}.mlp.experts.*) are NestQuant; everything "
             "else incl. MTP layer 78 (and its routed experts) is the source FP8 passthrough "
             "in nonexpert-*.safetensors"),
}
assert {k: v for k, v in cfg.items() if k != "quantization_config"} == \
       {k: v for k, v in src.items() if k != "quantization_config"}
assert {k: v for k, v in qc.items() if k != "nestquant"} == src["quantization_config"]
assert cfg["num_nextn_predict_layers"] == 1
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

mm = {k: vis[k] for k in TOP_KEYS if k in vis}
mm["vision_config"] = vis["vision_config"]
mm["text_config"] = cfg
mm["quantization_config"] = cfg["quantization_config"]
json.dump(mm, open(os.path.join(OUT, "config.mm.json"), "w"), indent=1)

ours, theirs = src, vis["text_config"]
MISSING = "<absent>"
diff = []
for k in sorted(set(ours) | set(theirs)):
    a, b = ours.get(k, MISSING), theirs.get(k, MISSING)
    if a == b:
        continue
    if k == "quantization_config":
        a = {"quant_method": a.get("quant_method"), "n_modules_to_not_convert": len(a.get("modules_to_not_convert", []))}
        b = {"quant_method": b.get("quant_method"), "keys": sorted(b)}
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        idx = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        a, b = {"len": len(ours[k]), "differs_at": idx, "ours": [ours[k][i] for i in idx]}, \
               {"theirs": [theirs[k][i] for i in idx]}
    elif isinstance(a, list) and len(a) > 8:
        a = {"len": len(a), "set": sorted(set(map(str, a)))}
    if isinstance(b, list) and len(b) > 8:
        b = {"len": len(b), "set": sorted(set(map(str, b)))}
    diff.append({"key": k, "ours": a, "theirs": b})
# top-level glm5v keys vs our text config (informational)
top = [{"key": k, "mm_top": vis[k], "ours_text": src.get(k, MISSING)} for k in TOP_KEYS
       if k in vis and vis[k] != src.get(k, MISSING)]
json.dump({"text_config_diff": diff, "top_level_vs_text": top,
           "same_keys_count": sum(1 for k in set(ours) & set(theirs) if ours[k] == theirs[k])},
          open(os.path.join(OUT, "config_text_diff.json"), "w"), indent=1)
for d in diff:
    print(json.dumps(d)[:400])
print("top-level:", json.dumps(top))
