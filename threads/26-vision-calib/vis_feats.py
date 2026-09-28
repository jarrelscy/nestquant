"""T26: projected 5.2V vision features for every image of the mm corpus (calib + heldout).

Vision path = glm52 tools/capture53/vision_probe53.vision_forward (plain-torch MoonViT + ORIGINAL 5.2V projector, the
graft default; the same function the earlier campaign's capture_mm53.py used), copied here so the glm52 hybrid-loader
imports are not needed.  Input = the stored 448x448 jpg (as capture_mm53: np.asarray(Image.open(jpg).convert("RGB"))).

Output: CORPUS/feats.bf16  [N_images, 256, 6144] bf16 in images.json order, CORPUS/feats.json (shape, sha256, source
file sha256s, pixel-tensor cross-check).  GPU ~3 GB.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

VD = "/tmp/nestquant/vision-graft"
VC = dict(hidden=1152, heads=16, head_dim=72, layers=27, inter=4304, patch=14, pos_hw=64, merge=(2, 2), text_hidden=6144)
DEV = "cuda"


def load_side(path, prefix):
    from safetensors import safe_open
    f = safe_open(path, "pt")
    return {k[len(prefix):]: f.get_tensor(k).to(DEV) for k in f.keys() if k.startswith(prefix)}


def rope_freqs(h, w, head_dim):
    idx = torch.arange(h * w, device=DEV).float()
    x, y = idx % w, idx // w
    fr = 1.0 / (10000 ** (torch.arange(0, head_dim, 4, device=DEV)[: head_dim // 4].float() / head_dim))
    xc = torch.polar(torch.ones(h * w, head_dim // 4, device=DEV), torch.outer(x, fr))
    yc = torch.polar(torch.ones(h * w, head_dim // 4, device=DEV), torch.outer(y, fr))
    return torch.cat([xc.unsqueeze(-1), yc.unsqueeze(-1)], -1).reshape(h * w, -1)


def apply_rope(q, k, cis):
    cis = cis.unsqueeze(-2)
    q_ = torch.view_as_complex(q.float().view(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().view(*k.shape[:-1], -1, 2))
    return (torch.view_as_real(q_ * cis).flatten(-2).type_as(q), torch.view_as_real(k_ * cis).flatten(-2).type_as(k))


def patches(img_u8):
    ps = VC["patch"]
    H, W = img_u8.shape[:2]
    x = (img_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
    p = x.reshape(1, H // ps, ps, W // ps, ps, 3).transpose(0, 1, 3, 5, 2, 4)
    return torch.from_numpy(p.reshape(-1, 3, ps, ps).copy())


@torch.no_grad()
def vision_forward(p, gh, gw, vt, pj):
    """== vision_probe53.vision_forward (p = patches(img) on DEV as bf16). Returns [n_tok, 6144]."""
    D, NH, HD = VC["hidden"], VC["heads"], VC["head_dim"]
    h = F.conv2d(p, vt["patch_embed.proj.weight"].to(torch.bfloat16),
                 vt["patch_embed.proj.bias"].to(torch.bfloat16)).view(p.shape[0], -1)
    pe = vt["patch_embed.pos_emb.weight"].float()
    if (gh, gw) != (VC["pos_hw"], VC["pos_hw"]):
        pe = F.interpolate(pe.permute(2, 0, 1).unsqueeze(0), size=(gh, gw), mode="bicubic").squeeze(0).permute(1, 2, 0)
    h = h + pe.reshape(-1, D).to(torch.bfloat16)
    cis = rope_freqs(gh, gw, HD)
    for li in range(VC["layers"]):
        b = f"encoder.blocks.{li}."
        r = h
        n = F.layer_norm(h, (D,), vt[b + "norm0.weight"], vt[b + "norm0.bias"])
        qkv = F.linear(n, vt[b + "wqkv.weight"], vt[b + "wqkv.bias"])
        q, k, v = qkv.view(-1, 3, NH, HD).unbind(1)
        q, k = apply_rope(q, k, cis)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        o = o.transpose(0, 1).reshape(-1, D)
        h = r + F.linear(o, vt[b + "wo.weight"], vt[b + "wo.bias"])
        r = h
        n = F.layer_norm(h, (D,), vt[b + "norm1.weight"], vt[b + "norm1.bias"])
        n = F.gelu(F.linear(n, vt[b + "mlp.fc0.weight"], vt[b + "mlp.fc0.bias"]), approximate="tanh")
        h = r + F.linear(n, vt[b + "mlp.fc1.weight"], vt[b + "mlp.fc1.bias"])
    h = F.layer_norm(h, (D,), vt["encoder.final_layernorm.weight"], vt["encoder.final_layernorm.bias"])
    kh, kw = VC["merge"]
    h = h.view(gh // kh, kh, gw // kw, kw, D).permute(0, 2, 1, 3, 4).reshape(-1, kh * kw, D)
    h = F.layer_norm(h, (D,), pj["pre_norm.weight"], pj["pre_norm.bias"]).reshape(-1, kh * kw * D)
    h = F.gelu(F.linear(h, pj["linear_1.weight"], pj["linear_1.bias"]))
    return F.linear(h, pj["linear_2.weight"], pj["linear_2.bias"])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/tmp/nestquant/19-capture-mm/corpus/c2048_mm")
    ap.add_argument("--mm", default="/tmp/nestquant/calib-mm")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(6 / 80)
    from PIL import Image
    from safetensors import safe_open
    imgs = json.load(open(f"{a.corpus}/images.json"))
    if a.limit:
        imgs = imgs[:a.limit]
    vt = load_side(f"{VD}/vision_tower.safetensors", "vision_tower.")
    pj = load_side(f"{VD}/mm_projector.safetensors", "mm_projector.")
    dts = sorted({str(v.dtype) for v in list(vt.values()) + list(pj.values())})
    out = f"{a.corpus}/feats.bf16"
    xcheck = []
    with open(out + ".tmp", "wb") as f:
        for i, r in enumerate(imgs):
            u8 = np.asarray(Image.open(f"{a.mm}/{r['image_file']}").convert("RGB"))
            assert u8.shape == (448, 448, 3), (r, u8.shape)
            p = patches(u8)
            if i % 100 == 0:                                   # stored processor tensor: same patch order / values?
                with safe_open(f"{a.mm}/{r['tensor_file']}", "pt") as sf:
                    pt = sf.get_tensor(r["tensor_key"]).float()
                xcheck.append(dict(id=r["id"], max_abs=float((pt - p).abs().max()), mean_abs=float((pt - p).abs().mean())))
            y = vision_forward(p.to(DEV, torch.bfloat16), 32, 32, vt, pj)
            assert y.shape == (256, 6144) and torch.isfinite(y).all(), r
            y.to(torch.bfloat16).cpu().view(torch.int16).numpy().tofile(f)
            if i % 200 == 0:
                print(json.dumps(dict(i=i, id=r["id"], norm=float(y.float().norm(dim=1).mean()))), flush=True)
    os.replace(out + ".tmp", out)
    json.dump(dict(shape=[len(imgs), 256, 6144], dtype="bfloat16", order="images.json", sha256=sha(out),
                   vision_tower_sha256=sha(f"{VD}/vision_tower.safetensors"),
                   mm_projector_sha256=sha(f"{VD}/mm_projector.safetensors"), weight_dtypes=dts,
                   input="stored 448x448 jpg (as glm52 capture_mm53.py)", pixel_tensor_xcheck=xcheck,
                   peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)),
              open(f"{a.corpus}/feats.json", "w"), indent=1)
    print("done", len(imgs), "peak GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
