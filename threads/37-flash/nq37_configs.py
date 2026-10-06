#!/usr/bin/env python
"""T37: release top-level config + passthrough files for GLM-5.3-Flash NestQuant 1.5/4 (adapted from T25 nq25_configs.py).

config.json = the source config (native vision-language glm5_next: text_config + vision_config) with a
quantization_config.nestquant block added; every other key is verbatim (asserted).  generation_config, tokenizer,
chat template, processor config and LICENSE are copied byte for byte (sha256 checked).  No vision graft is needed:
Flash's own vision tower ships in vision_tower.safetensors (see nq37_nonexpert.py)."""
import argparse, copy, hashlib, json, os, shutil

SRC = "/tmp/nestquant/37-flash/fp8"
OUT = "/tmp/nestquant/37-flash/release"
COPY = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "processor_config.json", "LICENSE"]

NESTQUANT = {
    "format": "nestquant-v1",
    "routed_experts": "layers/L{L}/tp{s}.safetensors",
    "manifest": "layers/L{L}/manifest.json",
    "layers": "3..44",
    "n_layers": 42,
    "n_routed_experts": 288,
    "tp": 8,
    "levels": [1.5, 4],
    "base": {"bits": 1.5, "code": "pattern-rate bitshift trellis, w_p = 1 + bit(0xAAAA, p % 16)"},
    "residual": {"to_bits": 4, "K": {"gate_proj": 2.5, "up_proj": 2.5, "down_proj": 2.8125}},
    "rotation": "random signs + Hadamard-128, both sides",
    "fixed_set": {"file": "fixed_set.json", "n_per_layer": 19,
                  "rule": "boundary-weighted REAP (think/end d1 50, d2_4 20, d5_16 5, d17_32 2), 0.75 text / 0.25 vision"},
    "floating": {"n_per_layer_at_96GB": 48, "predictor": "serving/predictor/"},
    "mtp": ("MTP layer 45 is the source FP8 passthrough; its 288 routed experts are in mtp_experts-*.safetensors "
            "(not NestQuant) and can be skipped by runtimes that run without MTP"),
    "vision": "native vision tower (model.visual.*) kept, bf16 passthrough in vision_tower.safetensors",
    "note": ("routed experts of layers 3-44 (model.language_model.layers.{3..44}.mlp.experts.*) are NestQuant; "
             "everything else is the source checkpoint byte for byte (index: model.safetensors.index.json)"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = json.load(open(os.path.join(a.src, "config.json")))
    tc = src["text_config"]
    assert tc["model_type"] == "glm5_next_text" and tc["n_routed_experts"] == 288
    assert tc["num_hidden_layers"] == 45 and tc["first_k_dense_replace"] == 3 and tc["num_nextn_predict_layers"] == 1
    assert src["vision_config"]["model_type"] == "glm5_next_vision"
    cfg = copy.deepcopy(src)
    qc = cfg["quantization_config"]  # FP8 fields + modules_to_not_convert kept verbatim
    assert "nestquant" not in qc
    qc["nestquant"] = NESTQUANT
    assert {k: v for k, v in cfg.items() if k != "quantization_config"} == \
           {k: v for k, v in src.items() if k != "quantization_config"}
    assert {k: v for k, v in qc.items() if k != "nestquant"} == src["quantization_config"]
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in COPY:
        shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))
        assert sha(os.path.join(a.src, f)) == sha(os.path.join(a.out, f)), f
        print(f"copied {f} {os.path.getsize(os.path.join(a.out, f))} B sha ok")
    print("config.json written; nestquant block:", json.dumps(NESTQUANT, indent=1))


if __name__ == "__main__":
    main()
