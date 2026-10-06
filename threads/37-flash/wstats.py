"""T37: GLM-5.3-Flash routed-expert weight statistics on CPU (FP8 block-scaled -> f32).
Checks the iid-Gaussian assumption the trellis rate model r(b) relies on: kurtosis before and after a
random-sign Hadamard-128 rotation on both axes (the encoder's incoherence step), plus row/col scale spread.
  python wstats.py -> /tmp/nestquant/37-flash/wstats.json"""
import json, os, sys
import torch
from safetensors import safe_open
torch.set_num_threads(int(os.environ.get("NT", "16")))
ROOT = "/tmp/nestquant/37-flash/fp8"
IDX = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
P = "model.language_model.layers.{L}.mlp.experts.{E}.{p}_proj.{w}"


def load(L, E, p):
    k = P.format(L=L, E=E, p=p, w="weight"); ks = P.format(L=L, E=E, p=p, w="weight_scale_inv")
    with safe_open(f"{ROOT}/{IDX[k]}", "pt") as f:
        w = f.get_tensor(k).float()
    with safe_open(f"{ROOT}/{IDX[ks]}", "pt") as f:
        s = f.get_tensor(ks).float()
    return w * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]


def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


H128 = had(128)


def rot(w, g):
    r, c = w.shape
    sr = torch.randint(0, 2, (r,), generator=g).float() * 2 - 1
    sc = torch.randint(0, 2, (c,), generator=g).float() * 2 - 1
    w = (w * sr[:, None] * sc[None]).view(r // 128, 128, c // 128, 128)
    return torch.einsum("ij,ajbk,kl->aibl", H128, w, H128).reshape(r, c)


def kurt(x):
    x = x.flatten().double(); x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


out = []
g = torch.Generator().manual_seed(0)
for L in [int(x) for x in os.environ.get("LAYERS", "3,12,24,36,44").split(",")]:
    for E in [0, 97, 191, 287]:
        for p in ("gate", "up", "down"):
            w = load(L, E, p)
            rs = w.pow(2).mean(1).sqrt(); cs = w.pow(2).mean(0).sqrt()
            # remove row/col scales the encoder learns (suh/svh) before rotating
            wn = w / rs[:, None]; wn = wn / wn.pow(2).mean(0).sqrt()[None]
            wr = rot(wn, g)
            o = dict(L=L, E=E, p=p, rms=float(w.pow(2).mean().sqrt()), kurt_raw=kurt(w), kurt_norm=kurt(wn),
                     kurt_rot=kurt(wr), row_cv=float(rs.std() / rs.mean()), col_cv=float(cs.std() / cs.mean()),
                     max_abs_rot=float(wr.abs().max()))
            out.append(o); print(json.dumps(o), flush=True)
json.dump(out, open("/tmp/nestquant/37-flash/wstats.json", "w"), indent=1)
