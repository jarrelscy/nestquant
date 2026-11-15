#!/usr/bin/env python3
"""CPU demo: run Inkling-2-Small's vision tower on saved corpus_mm/pix/<id>.pt files and splice the output into the
token-embedding stream at the image slot. Shows the exact call a capture script needs.

  CUDA_VISIBLE_DEVICES= .venv/bin/python embed_demo.py [n_windows=1]

Vision weights (model.visual.* in model-00024) are renamed with the same rules as transformers'
conversion_mapping["inkling2"] and loaded into a standalone Inkling2VisionModel (~0.4 B params, bf16).
"""
import json, re, sys
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.inkling2 import modeling_inkling2 as M

MODEL = "/tmp/nq_inkling/model"
ROOT = "/tmp/nq_inkling/corpus_mm"
RULES = [
    (r"^model\.visual\.backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^model\.visual\.backbone\.embeddings\.", "embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "layers."),
    (r"^model\.visual\.proj_in\.", "merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "merger.linear_2."),
    (r"(layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(layers\.\d+)\.ls_1\.gamma", r"\1.lambda_1"),
    (r"(layers\.\d+)\.ls_2\.gamma", r"\1.lambda_2"),
]


def load_vision():
    cfg = AutoConfig.from_pretrained(MODEL)
    vc = cfg.vision_config
    vc._attn_implementation = "sdpa"
    vm = M.Inkling2VisionModel(vc).to(torch.bfloat16).eval()
    wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in wm if k.startswith("model.visual.")]
    sd = {}
    for shard in sorted({wm[k] for k in keys}):
        with safe_open(f"{MODEL}/{shard}", "pt") as f:
            for k in keys:
                if wm[k] == shard:
                    n = k
                    for a, b in RULES:
                        n = re.sub(a, b, n)
                    sd[n] = f.get_tensor(k)
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return vm, cfg


@torch.no_grad()
def main():
    nwin = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    torch.set_num_threads(8)
    vm, cfg = load_vision()
    win = [json.loads(l) for l in open(f"{ROOT}/windows.jsonl")]
    import numpy as np
    T = np.load(f"{ROOT}/tokens.npy", mmap_mode="r")
    for w in range(nwin):
        slots = win[w]["images"]
        pix = [torch.load(f"{ROOT}/{s['pix_file']}", weights_only=True) for s in slots]
        pv = torch.cat([p["pixel_values"] for p in pix])              # [n_img*256, 40, 40, 3] fp32
        grid = torch.cat([p["image_grid_thw"] for p in pix])          # [n_img, 3] = [[1,16,16], ...]
        # == Inkling2Model.get_image_features(pv, grid).pooler_output (a tuple split per image)
        feats = vm(pixel_values=pv, image_grid_thw=grid).pooler_output.split(grid.prod(-1).tolist())
        ids = torch.from_numpy(np.asarray(T[w], np.int64))
        mask = ids == cfg.image_token_id
        for s, f in zip(slots, feats):
            assert f.shape == (s["n_tokens"], cfg.text_config.hidden_size)
            assert mask[s["start"]:s["end"]].all()
        allf = torch.cat(feats)
        assert int(mask.sum()) == allf.shape[0]
        print(f"window {w}: {len(slots)} images -> features {tuple(allf.shape)} {allf.dtype}, "
              f"rms {allf.float().pow(2).mean().sqrt():.4f}, finite {bool(torch.isfinite(allf).all())}; "
              f"placeholders {int(mask.sum())} (masked_scatter order == slot order)")


if __name__ == "__main__":
    main()
