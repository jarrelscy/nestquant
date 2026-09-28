#!/usr/bin/env python
"""T25: re-shard every non-routed-expert tensor of the GLM-5.3 FP8 source
(byte-for-byte FP8 passthrough) into repo-top/nonexpert-*.safetensors.

Excluded: model.layers.{3..77}.mlp.experts.*  (those become NestQuant).
Kept: everything else, incl. ALL of MTP layer 78 (its routed experts too).
Writes model.safetensors.index.json + nonexpert_manifest.json; verifies
dtype/shape/raw bytes of every tensor against the source.
"""
import hashlib, json, os, re, sys, time
from collections import Counter, defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/tmp/nestquant/src/glm53-fp8"
OUT = "/tmp/nestquant/25-campaign/repo-top"
SHARD_BYTES = 5 * 10**9
EXP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.")
LAY_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def excluded(name):
    m = EXP_RE.match(name)
    return bool(m) and 3 <= int(m.group(1)) <= 77


def category(name):
    if name == "model.embed_tokens.weight": return "embed_tokens"
    if name == "lm_head.weight": return "lm_head"
    if name == "model.norm.weight": return "final_norm"
    m = LAY_RE.match(name)
    if not m: return "UNCATEGORIZED"
    L, rest = int(m.group(1)), m.group(2)
    if L == 78:
        if rest.startswith("mlp.experts."): return "mtp78/routed_experts"
        if rest.startswith("mlp.shared_experts."): return "mtp78/shared_experts"
        if rest.startswith("mlp.gate."): return "mtp78/gate"
        if rest.startswith("self_attn."): return "mtp78/attention"
        if rest.startswith(("input_layernorm", "post_attention_layernorm")): return "mtp78/layer_norms"
        for k in ("eh_proj", "enorm", "hnorm", "shared_head"):
            if rest.startswith(k): return f"mtp78/{k}"
        return "UNCATEGORIZED"
    if L > 78: return "UNCATEGORIZED"
    if rest.startswith("self_attn."): return "attention_L0-77"
    if rest.startswith(("input_layernorm", "post_attention_layernorm")): return "layer_norms_L0-77"
    if L < 3 and re.match(r"mlp\.(gate|up|down)_proj\.", rest): return "dense_mlp_L0-2"
    if L >= 3 and rest.startswith("mlp.shared_experts."): return "shared_experts_L3-77"
    if L >= 3 and rest.startswith("mlp.gate."): return "router_gate_L3-77"
    return "UNCATEGORIZED"


def sort_key(name):
    m = LAY_RE.match(name)
    if name == "model.embed_tokens.weight": return (-1, [], name)
    if not m: return (10**6, [], name)  # model.norm, lm_head at the end
    parts = [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in m.group(2).split(".")]
    return (int(m.group(1)), parts, name)


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    keep = sorted((n for n in wm if not excluded(n)), key=sort_key)
    n_excl = len(wm) - len(keep)
    print(f"source tensors {len(wm)}  keep {len(keep)}  excluded {n_excl}", flush=True)

    # sizes from source headers
    info = {}
    for f in sorted(set(wm[n] for n in keep)):
        with safe_open(os.path.join(SRC, f), "pt") as h:
            for n in h.keys():
                if n in wm and wm[n] == f and not excluded(n):
                    sl = h.get_slice(n)
                    info[n] = (sl.get_dtype(), sl.get_shape())
    assert set(info) == set(keep), "index/header mismatch"
    esz = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4}
    nbytes = {n: esz[info[n][0]] * (1 if not info[n][1] else int(torch.tensor(info[n][1]).prod()))
              for n in keep}

    shards, cur, cur_b = [], [], 0
    for n in keep:
        if cur and cur_b + nbytes[n] > SHARD_BYTES:
            shards.append(cur); cur, cur_b = [], 0
        cur.append(n); cur_b += nbytes[n]
    if cur: shards.append(cur)
    M = len(shards)
    fname = lambda i: f"nonexpert-{i+1:05d}-of-{M:05d}.safetensors"

    new_wm, files, cats = {}, [], defaultdict(lambda: {"n_tensors": 0, "bytes": 0})
    n_ok = 0
    for i, names in enumerate(shards):
        t0 = time.time()
        by_src = defaultdict(list)
        for n in names: by_src[wm[n]].append(n)
        tensors = {}
        for f, ns in by_src.items():
            with safe_open(os.path.join(SRC, f), "pt") as h:
                for n in ns:
                    tensors[n] = h.get_tensor(n).clone()
        out_path = os.path.join(OUT, fname(i))
        save_file({n: tensors[n] for n in names}, out_path + ".part", metadata={"format": "pt"})
        os.rename(out_path + ".part", out_path)
        # verify from disk vs source (re-read source independently)
        with safe_open(out_path, "pt") as ho:
            assert set(ho.keys()) == set(names)
            for f, ns in by_src.items():
                with safe_open(os.path.join(SRC, f), "pt") as hs:
                    for n in ns:
                        a, b = hs.get_tensor(n), ho.get_tensor(n)
                        assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape), n
                        assert str(ho.get_slice(n).get_dtype()) == str(info[n][0]), n
                        assert torch.equal(a.contiguous().view(torch.uint8).flatten(),
                                           b.contiguous().view(torch.uint8).flatten()), n
                        n_ok += 1
        del tensors
        h = hashlib.sha256()
        with open(out_path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""): h.update(blk)
        files.append({"file": fname(i), "bytes": os.path.getsize(out_path),
                      "tensor_bytes": sum(nbytes[n] for n in names),
                      "n_tensors": len(names), "sha256": h.hexdigest(),
                      "first": names[0], "last": names[-1]})
        for n in names:
            new_wm[n] = fname(i)
            c = cats[category(n)]; c["n_tensors"] += 1; c["bytes"] += nbytes[n]
        print(f"[{i+1}/{M}] {fname(i)} {files[-1]['bytes']/1e9:.3f} GB "
              f"{len(names)} tensors verified ({time.time()-t0:.0f}s)", flush=True)

    total = sum(nbytes.values())
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))},
              open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
    mtp = [n for n in keep if n.startswith("model.layers.78.")]
    src_mtp = [n for n in wm if n.startswith("model.layers.78.")]
    manifest = {
        "source": SRC, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": "all source tensors except model.layers.{3..77}.mlp.experts.* ; FP8 passthrough byte-for-byte",
        "n_source_tensors": len(wm), "n_excluded": n_excl, "n_tensors": len(keep),
        "n_verified_identical": n_ok, "total_tensor_bytes": total,
        "total_file_bytes": sum(f["bytes"] for f in files),
        "dtype_counts": dict(Counter(str(info[n][0]) for n in keep)),
        "categories": dict(sorted(cats.items())),
        "uncategorized": [n for n in keep if category(n) == "UNCATEGORIZED"],
        "mtp78": {"n_tensors": len(mtp), "n_source": len(src_mtp),
                  "bytes": sum(nbytes[n] for n in mtp), "all_present": set(mtp) == set(src_mtp)},
        "files": files,
    }
    json.dump(manifest, open(os.path.join(OUT, "nonexpert_manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    assert n_ok == len(keep)


if __name__ == "__main__":
    main()
