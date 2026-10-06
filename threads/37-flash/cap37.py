"""T37: layer-sequential CPU capture of GLM-5.3-Flash (FP8 checkpoint dequantised to f32) over calibration segments.

Runs the HF glm5_next decoder layers one at a time over every segment (attention local to a segment, positions reset per
segment, as the glmfmt corpus packing defines), carrying the 4-stream mHC hidden state between layers on /tmp.
Per MoE layer L writes OUT/L{L}/:
  A0.f32   [288, npk(4096)]  sum over routed rows of x x^T (gate/up input), packed upper triangle
  D0.f32   [288, npk(2048)]  sum over routed rows of h h^T (down input)
  sal.npy  [288, 3 kinds, 5 buckets, 2]  (sum p*||y||, row count); kinds none/think/end, buckets other,d1,d2_4,d5_16,d17_32
  ids.u16 / w.f16 / logit.f16  per-token routing [N, 8] / [N, 8] / [N, 288]  (PRIVATE, never uploaded)
After the last layer: OUT/final/ per-token CE and top-64 teacher logprobs for val segments.
Vision windows: image features come from Flash's own vision tower on the calib-mm jpgs (Glm5NextImageProcessor).
  python cap37.py --groups mm:fit,traces:fit --max-tokens N --out /tmp/nestquant/37-flash/cap/run0 [--layers 0-44]"""
import argparse, json, os, re, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("NT", "20")))
torch.set_grad_enabled(False)
from transformers import AutoConfig
from transformers.models.glm5_next import modeling_glm5_next as M

CK = "/tmp/nestquant/37-flash/fp8"
CORP = "/tmp/nestquant/corpus/glm53_calib_glmfmt_v1"
MM = "/tmp/nestquant/19-capture-mm/corpus/c2048_mm"
PFX = "model.language_model."
IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_F = {}
BUCKETS = [(1, 1), (2, 4), (5, 16), (17, 32)]


def get(k):
    fn = IDX[k]
    if fn not in _F:
        _F[fn] = safe_open(f"{CK}/{fn}", "pt")
    w = _F[fn].get_tensor(k)
    ks = k[: -len("weight")] + "weight_scale_inv" if k.endswith("weight") else None
    if ks in IDX:
        s = _F.setdefault(IDX[ks], safe_open(f"{CK}/{IDX[ks]}", "pt")).get_tensor(ks).float()
        s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        return w.float() * s
    return w.float()


RENAME = [(r"self_attn\.(f_a_proj|f_b_proj|dt_bias|A_log)", r"self_attn.forget_gate.\1"),
          (r"hc_attn_(fn|base|scale)", r"attn_hc.\1"), (r"hc_ffn_(fn|base|scale)", r"ffn_hc.\1")]


def layer_sd(L, ne):
    p = f"{PFX}layers.{L}."
    keys = [k for k in IDX if k.startswith(p) and not k.endswith("weight_scale_inv")]
    sd, gate, up, down, conv = {}, [None] * ne, [None] * ne, [None] * ne, {}
    for k in keys:
        r = k[len(p):]
        m = re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", r)
        if m:
            {"gate": gate, "up": up, "down": down}[m.group(2)][int(m.group(1))] = get(k); continue
        m = re.match(r"self_attn\.([qkv])_conv1d\.weight", r)
        if m:
            conv[m.group(1)] = get(k); continue
        for a, b in RENAME:
            r = re.sub(a, b, r)
        sd[r] = get(k)
    if gate[0] is not None:
        sd["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([g, u], 0) for g, u in zip(gate, up)])
        sd["mlp.experts.down_proj"] = torch.stack(down)
    if conv:
        sd["self_attn.conv1d.weight"] = torch.cat([conv["q"], conv["k"], conv["v"]], 0)
    return sd


def build_layer(cfg, L):
    with torch.device("meta"):
        lay = M.Glm5NextTextDecoderLayer(cfg, L)
    sd = layer_sd(L, cfg.n_routed_experts)
    lay.to_empty(device="cpu")
    missing, unexpected = lay.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("e_score_correction_bias") or m not in sd]
    assert not unexpected and not missing, (L, missing, unexpected)
    return lay.float().eval()


def npk(n):
    return n * (n + 1) // 2


class Cap:
    """Per-layer MoE statistics; wraps Glm5NextTextMoE.forward."""

    def __init__(self, moe, ne, n_tok):
        self.moe, self.ne = moe, ne
        self.A0 = torch.zeros(ne, 4096, 4096); self.D0 = torch.zeros(ne, 2048, 2048)
        self.sal = np.zeros((ne, 3, 5, 2))
        self.ids = np.zeros((n_tok, 8), np.uint16); self.w = np.zeros((n_tok, 8), np.float16)
        self.lg = np.zeros((n_tok, ne), np.float16)
        self.pos = 0; self.kind = self.bk = None
        moe.forward = self.forward

    def forward(self, hs):
        E = self.moe.experts
        logits, tw, ti = self.moe.gate(hs)
        x = hs.reshape(-1, hs.shape[-1]); n = x.shape[0]
        s = slice(self.pos, self.pos + n); self.pos += n
        self.ids[s] = ti.numpy(); self.w[s] = tw.numpy(); self.lg[s] = logits.numpy()
        out = torch.zeros_like(x)
        for e in torch.unique(ti).tolist():
            rows, slot = torch.where(ti == e)
            xe = x[rows]; h = E._apply_gate(xe @ E.gate_up_proj[e].T); y = h @ E.down_proj[e].T
            p = tw[rows, slot]
            out.index_add_(0, rows, y * p[:, None])
            self.A0[e].addmm_(xe.T, xe); self.D0[e].addmm_(h.T, h)
            v = (p * y.norm(dim=1)).numpy(); k = self.kind[rows.numpy()]; b = self.bk[rows.numpy()]
            np.add.at(self.sal[e], (k, b, 0), v); np.add.at(self.sal[e], (k, b, 1), 1)
        return out.view(hs.shape) + self.moe.shared_experts(hs)

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        iu4 = torch.triu_indices(4096, 4096); iu2 = torch.triu_indices(2048, 2048)
        with open(f"{d}/A0.f32", "wb") as f:
            for e in range(self.ne):
                f.write(self.A0[e][iu4[0], iu4[1]].numpy().tobytes())
        with open(f"{d}/D0.f32", "wb") as f:
            for e in range(self.ne):
                f.write(self.D0[e][iu2[0], iu2[1]].numpy().tobytes())
        np.save(f"{d}/sal.npy", self.sal)
        self.ids[: self.pos].tofile(f"{d}/ids.u16"); self.w[: self.pos].tofile(f"{d}/w.f16")
        self.lg[: self.pos].tofile(f"{d}/logit.f16")


def bucket_kind(think_d, end_d):
    kind = np.zeros(len(think_d), np.int64); bk = np.zeros(len(think_d), np.int64)
    for kk, d in ((1, think_d), (2, end_d)):
        for bi, (lo, hi) in enumerate(BUCKETS):
            m = (d >= lo) & (d <= hi) & (kind == 0)
            kind[m] = kk; bk[m] = bi + 1
    return kind, bk


def load_segments(groups, max_tokens):
    """-> list of dicts(tokens, think_d, end_d, img=[(pos, jpg)], val)."""
    segs = []
    for g in groups:
        name, split = g.split(":")
        root = MM if name == "mm" else f"{CORP}/{dict(traces='c2048_traces', c2048='c2048', c512='c512')[name]}"
        tok = np.load(f"{root}/tokens.npy"); sg = np.load(f"{root}/segments.npy")
        bt = np.load(f"{root}/bnd_think_d.npy") if os.path.exists(f"{root}/bnd_think_d.npy") else np.zeros_like(tok)
        be = np.load(f"{root}/bnd_end_d.npy") if os.path.exists(f"{root}/bnd_end_d.npy") else np.zeros_like(tok)
        rk = np.load(f"{root}/rowkind.npy") if name == "mm" else None
        imgs = {}
        if name == "mm":
            for r in json.load(open(f"{root}/images.json")):
                imgs.setdefault(r["window"], []).append((r["pos"], "/tmp/nestquant/calib-mm/" + r["image_file"]))
        a, b = json.load(open(f"{root}/split.json"))[split]
        for w in range(a, b):
            cut = np.flatnonzero(np.diff(sg[w])) + 1
            for s0, s1 in zip(np.r_[0, cut], np.r_[cut, tok.shape[1]]):
                if rk is not None and (rk[w, s0:s1] == 2).all():
                    continue  # pad
                im = [(p - s0, j) for p, j in imgs.get(w, []) if s0 <= p < s1]
                segs.append(dict(group=name, w=w, tokens=torch.from_numpy(tok[w, s0:s1].astype(np.int64)),
                                 think_d=bt[w, s0:s1], end_d=be[w, s0:s1], img=im, val=split == "val"))
        if max_tokens and sum(len(s["tokens"]) for s in segs) >= max_tokens:
            break
    if max_tokens:
        out, n = [], 0
        for s in segs:
            if n >= max_tokens:
                break
            out.append(s); n += len(s["tokens"])
        segs = out
    return segs


def image_embeds(cfg_full, segs):
    from PIL import Image
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
    jobs = [(si, p, j) for si, s in enumerate(segs) for p, j in s["img"]]
    if not jobs:
        return {}
    proc = Glm5NextImageProcessor(**{k: v for k, v in json.load(open(f"{CK}/processor_config.json"))["image_processor"].items()
                                     if k != "image_processor_type"})
    with torch.device("meta"):
        vis = M.Glm5NextVisionModel(cfg_full.vision_config)
    sd = {k[len("model.visual."):]: get(k) for k in IDX if k.startswith("model.visual.")}
    vis.to_empty(device="cpu"); vis.load_state_dict(sd, strict=True); vis.float().eval()
    out = {}
    for si, p, j in jobs:
        im = Image.open(j).convert("RGB")
        bf = proc(images=[im], return_tensors="pt")
        o = vis(bf["pixel_values"].float(), grid_thw=bf["image_grid_thw"]).pooler_output
        out[(si, p)] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="mm:fit,traces:fit")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-44")
    ap.add_argument("--no-stats", action="store_true", help="forward only (timing / sanity)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg_full = AutoConfig.from_pretrained(CK); cfg = cfg_full.text_config
    cfg.num_local_experts = cfg.n_routed_experts
    segs = load_segments(a.groups.split(","), a.max_tokens)
    N = sum(len(s["tokens"]) for s in segs)
    kind, bk = bucket_kind(np.concatenate([s["think_d"] for s in segs]), np.concatenate([s["end_d"] for s in segs]))
    print(f"{len(segs)} segments, {N} tokens, boundary rows {(kind > 0).sum()}", flush=True)
    json.dump(dict(groups=a.groups, max_tokens=a.max_tokens, n_tokens=N,
                   segs=[dict(group=s["group"], w=s["w"], n=len(s["tokens"]), val=s["val"]) for s in segs]),
              open(f"{a.out}/segments.json", "w"))
    L0, L1 = map(int, a.layers.split("-"))
    hp = f"{a.out}/hid_L{L0}.bf16"
    if L0 == 0:
        emb = get(PFX + "embed_tokens.weight")
        t0 = time.time(); ims = image_embeds(cfg_full, segs); print(f"vision {len(ims)} images {time.time()-t0:.0f}s", flush=True)
        H = torch.empty(N, cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16); o = 0
        for si, s in enumerate(segs):
            e = emb[s["tokens"]]
            for p, _ in s["img"]:
                f = ims[(si, p)]; m = (s["tokens"][p:] == cfg_full.image_token_id).nonzero()[: f.shape[0], 0] + p
                assert len(m) == f.shape[0], (si, p, len(m), f.shape)
                e[m] = f
            H[o:o + len(e)] = e[:, None].expand(-1, cfg.hc_mult, -1).bfloat16(); o += len(e)
        del emb
    else:
        H = torch.from_numpy(np.fromfile(hp, np.int16)).view(torch.bfloat16).view(N, cfg.hc_mult, cfg.hidden_size)
    for L in range(L0, L1 + 1):
        t0 = time.time(); lay = build_layer(cfg, L); tl = time.time() - t0
        cap = None
        if cfg.mlp_layer_types[L] == "sparse" and not a.no_stats:
            cap = Cap(lay.mlp, cfg.n_routed_experts, N); cap.kind, cap.bk = kind, bk
        o = 0; t0 = time.time()
        for s in segs:
            n = len(s["tokens"])
            if cap is not None:
                cap.kind, cap.bk = kind[o:o + n], bk[o:o + n]
            h = H[o:o + n].float()[None]
            out, _ = lay(h, attention_mask=torch.ones(1, n, dtype=torch.bool), position_ids=torch.arange(n)[None])
            H[o:o + n] = out[0].bfloat16(); o += n
        tf = time.time() - t0
        if cap is not None:
            cap.save(f"{a.out}/L{L}")
        H.view(torch.int16).numpy().tofile(f"{a.out}/hid_L{L + 1}.tmp"); os.replace(f"{a.out}/hid_L{L + 1}.tmp", f"{a.out}/hid_L{L + 1}.bf16")
        if os.path.exists(f"{a.out}/hid_L{L}.bf16") and L > L0:
            os.remove(f"{a.out}/hid_L{L}.bf16")
        print(f"L{L} {cfg.layer_types[L][:6]} {cfg.mlp_layer_types[L]} load {tl:.0f}s fwd {tf:.0f}s ({tf / N * 1e3:.1f} ms/tok)",
              flush=True)
        del lay, cap
    if L1 == cfg.num_hidden_layers - 1:
        norm = M.Glm5NextTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps); norm.weight.copy_(get(PFX + "norm.weight"))
        head = get("lm_head.weight"); hh = M.Glm5NextTextHyperHead()
        os.makedirs(f"{a.out}/final", exist_ok=True); o = 0; res = []
        for si, s in enumerate(segs):
            n = len(s["tokens"]); z = norm(hh(H[o:o + n].float()[None]))[0] @ head.T; o += n
            lp = F.log_softmax(z, -1); tgt = s["tokens"][1:]
            ce = -lp[:-1].gather(1, tgt[:, None])[:, 0]; acc = (lp[:-1].argmax(-1) == tgt).float()
            res.append(dict(seg=si, group=s["group"], val=s["val"], n=n, ce=float(ce.mean()), acc=float(acc.mean())))
            if s["val"]:
                v, i = lp.topk(64, -1); torch.save(dict(v=v.half(), i=i.int()), f"{a.out}/final/seg{si}.pt")
        json.dump(res, open(f"{a.out}/final/ce.json", "w"), indent=0)
        for g in sorted({r["group"] for r in res}):
            rr = [r for r in res if r["group"] == g]; nt = sum(r["n"] for r in rr)
            print(g, "ppl", float(np.exp(sum(r["ce"] * r["n"] for r in rr) / nt)), "acc", sum(r["acc"] * r["n"] for r in rr) / nt)


if __name__ == "__main__":
    main()
