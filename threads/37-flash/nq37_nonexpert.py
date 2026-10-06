#!/usr/bin/env python
"""T37: re-shard every non-routed-expert tensor of the GLM-5.3-Flash FP8 source (byte-for-byte passthrough,
FP8 where FP8, bf16/fp32 where not) into the release top level.  Adapted from T25 nq25_nonexpert.py.

Excluded:  model.language_model.layers.{3..44}.mlp.experts.*   (those become NestQuant 1.5/4)
Kept, three file groups (all listed in one model.safetensors.index.json):
  nonexpert-*.safetensors      text backbone: embeddings, lm_head, final norm, attention (KDA + DSA/MLA), mHC hc_*,
                               norms, dense MLP L0-2, shared experts, router gates + e_score_correction_bias, and the
                               MTP layer 45 except its routed experts
  vision_tower.safetensors     model.visual.* (native vision tower + merger, bf16)
  mtp_experts-*.safetensors    MTP layer 45 routed experts (FP8).  Same treatment as the GLM-5.3 releases (MTP layer
                               78 incl. its routed experts = FP8 passthrough), but in separate files so a memory-bound
                               runtime (96 GB Mac) can skip them.
Shards are written by raw byte copy (safetensors header + the source data bytes, no dtype round trip).  Verification
re-opens every output with safetensors.safe_open and compares dtype, shape and the sha256 of each tensor's bytes with
the source (read independently through the source headers).  Writes nonexpert_manifest.json and
nonexpert_tensor_sha256.json.
"""
import argparse, hashlib, json, os, re, struct, sys, time
from collections import Counter, defaultdict

import torch
from safetensors import safe_open

SRC = "/tmp/nestquant/37-flash/fp8"
OUT = "/tmp/nestquant/37-flash/release"
SHARD_BYTES = 5 * 10**9
P = "model.language_model."
EXP_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.")
LAY_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")
NQ_LAYERS = range(3, 45)
MTP = 45
ESZ = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


def excluded(name):
    m = EXP_RE.match(name)
    return bool(m) and int(m.group(1)) in NQ_LAYERS


def group(name):
    if name.startswith("model.visual."):
        return "vision"
    m = EXP_RE.match(name)
    if m and int(m.group(1)) == MTP:
        return "mtp_experts"
    return "nonexpert"


def category(name, layer_types):
    if name.startswith("model.visual."):
        return "vision/" + name.split(".")[2]
    if name == P + "embed_tokens.weight": return "embed_tokens"
    if name == "lm_head.weight": return "lm_head"
    if name == P + "norm.weight": return "final_norm"
    m = LAY_RE.match(name)
    if not m: return "UNCATEGORIZED"
    L, rest = int(m.group(1)), m.group(2)
    if L == MTP:
        if rest.startswith("mlp.experts."): return "mtp45/routed_experts"
        if rest.startswith("mlp.shared_experts."): return "mtp45/shared_experts"
        if rest.startswith("mlp.gate."): return "mtp45/gate"
        if rest.startswith("self_attn."): return "mtp45/attention_dsa"
        if rest.startswith(("input_layernorm", "post_attention_layernorm")): return "mtp45/layer_norms"
        for k in ("eh_proj", "enorm", "hnorm", "shared_head"):
            if rest.startswith(k): return f"mtp45/{k}"
        return "UNCATEGORIZED"
    if L > MTP: return "UNCATEGORIZED"
    if rest.startswith("self_attn."):
        return {"linear_attention": "attention_kda", "deepseek_sparse_attention": "attention_dsa"}.get(
            layer_types[L], "UNCATEGORIZED")
    if rest.startswith("hc_"): return "mhc_hyper_connections"
    if rest.startswith(("input_layernorm", "post_attention_layernorm")): return "layer_norms"
    if L < 3 and re.match(r"mlp\.(gate|up|down)_proj\.", rest): return "dense_mlp_L0-2"
    if L >= 3 and rest.startswith("mlp.shared_experts."): return "shared_experts_L3-44"
    if L >= 3 and rest.startswith("mlp.gate."): return "router_gate_L3-44"
    return "UNCATEGORIZED"


def sort_key(name):
    if name == P + "embed_tokens.weight": return (-1, [], name)
    m = LAY_RE.match(name)
    if not m:
        return (10**6, [], name)  # final norm, lm_head, visual at the end
    parts = [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in m.group(2).split(".")]
    return (int(m.group(1)), parts, name)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def tensor_bytes(fh, base, ent):
    a, b = ent["data_offsets"]
    fh.seek(base + a)
    buf = fh.read(b - a)
    assert len(buf) == b - a
    return buf


def write_shard(path, names, src_of, hdr, base):
    """Raw safetensors write: tensors ordered by element size (desc) so every tensor stays naturally aligned."""
    order = sorted(names, key=lambda n: (-ESZ[hdr[n]["dtype"]], names.index(n)))
    head, off = {"__metadata__": {"format": "pt"}}, 0
    for n in order:
        nb = hdr[n]["data_offsets"][1] - hdr[n]["data_offsets"][0]
        head[n] = {"dtype": hdr[n]["dtype"], "shape": hdr[n]["shape"], "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(head, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    fhs = {}
    with open(path + ".part", "wb") as out:
        out.write(struct.pack("<Q", len(hb))); out.write(hb)
        for n in order:
            f = src_of[n]
            if f not in fhs: fhs[f] = open(os.path.join(SRC, f), "rb")
            out.write(tensor_bytes(fhs[f], base[f], hdr[n]))
    for fh in fhs.values(): fh.close()
    os.rename(path + ".part", path)
    return 8 + len(hb) + off


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    globals()["SRC"] = a.src
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    layer_types = cfg["text_config"]["layer_types"]
    assert len(layer_types) == 45 and cfg["text_config"]["num_nextn_predict_layers"] == 1
    wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]

    hdr, base = {}, {}
    for f in sorted(set(wm.values())):
        h, b = read_header(os.path.join(SRC, f))
        base[f] = b
        for n, e in h.items():
            assert wm.get(n) == f, f"{n} in {f} but index says {wm.get(n)}"
            hdr[n] = e
    assert set(hdr) == set(wm), "index/header mismatch"
    nbytes = {n: e["data_offsets"][1] - e["data_offsets"][0] for n, e in hdr.items()}
    for n, e in hdr.items():
        exp = ESZ[e["dtype"]] * (1 if not e["shape"] else int(torch.tensor(e["shape"]).prod()))
        assert exp == nbytes[n], n

    keep = sorted((n for n in wm if not excluded(n)), key=sort_key)
    n_excl = len(wm) - len(keep)
    exp_src = Counter(int(EXP_RE.match(n).group(1)) for n in wm if EXP_RE.match(n))
    assert set(exp_src) == set(NQ_LAYERS) | {MTP} and all(v == 288 * 6 for v in exp_src.values()), exp_src
    print(f"source tensors {len(wm)}  keep {len(keep)}  excluded {n_excl} "
          f"({sum(nbytes[n] for n in wm if excluded(n))/1e9:.2f} GB)", flush=True)

    shards = []  # (fname_prefix, names)
    for g in ("nonexpert", "vision", "mtp_experts"):
        names = [n for n in keep if group(n) == g]
        if g == "vision":
            shards.append(("vision_tower", [names])); continue
        parts, cur, cur_b = [], [], 0
        for n in names:
            if cur and cur_b + nbytes[n] > SHARD_BYTES:
                parts.append(cur); cur, cur_b = [], 0
            cur.append(n); cur_b += nbytes[n]
        if cur: parts.append(cur)
        shards.append((g, parts))
    jobs = []
    for g, parts in shards:
        M = len(parts)
        for i, names in enumerate(parts):
            fn = f"{g}.safetensors" if g == "vision_tower" else f"{g}-{i+1:05d}-of-{M:05d}.safetensors"
            jobs.append((fn, names))

    new_wm, files, cats, tsha = {}, [], defaultdict(lambda: {"n_tensors": 0, "bytes": 0}), {}
    n_ok = 0
    for j, (fn, names) in enumerate(jobs):
        t0 = time.time()
        path = os.path.join(a.out, fn)
        fbytes = write_shard(path, names, wm, hdr, base)
        assert os.path.getsize(path) == fbytes
        # verify: output via safetensors (format check), source via its own header offsets
        fhs = {}
        with safe_open(path, "pt") as ho:
            assert set(ho.keys()) == set(names), fn
            for n in names:
                f = wm[n]
                if f not in fhs: fhs[f] = open(os.path.join(SRC, f), "rb")
                s_sha = hashlib.sha256(tensor_bytes(fhs[f], base[f], hdr[n])).hexdigest()
                t = ho.get_tensor(n)
                assert str(ho.get_slice(n).get_dtype()) == hdr[n]["dtype"], n
                assert list(t.shape) == list(hdr[n]["shape"]), n
                o_sha = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert s_sha == o_sha, n
                tsha[n] = o_sha
                n_ok += 1
        for fh in fhs.values(): fh.close()
        files.append({"file": fn, "bytes": fbytes, "tensor_bytes": sum(nbytes[n] for n in names),
                      "n_tensors": len(names), "sha256": file_sha(path), "first": names[0], "last": names[-1]})
        for n in names:
            assert n not in new_wm
            new_wm[n] = fn
            c = cats[category(n, layer_types)]; c["n_tensors"] += 1; c["bytes"] += nbytes[n]
        print(f"[{j+1}/{len(jobs)}] {fn} {fbytes/1e9:.3f} GB {len(names)} tensors sha-verified "
              f"({time.time()-t0:.0f}s)", flush=True)

    assert set(new_wm) == set(keep) and n_ok == len(keep)
    assert not any(excluded(n) for n in new_wm)
    total = sum(nbytes[n] for n in keep)
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    json.dump(dict(sorted(tsha.items())), open(os.path.join(a.out, "nonexpert_tensor_sha256.json"), "w"), indent=0)
    mtp = [n for n in keep if n.startswith(f"{P}layers.{MTP}.")]
    src_mtp = [n for n in wm if n.startswith(f"{P}layers.{MTP}.")]
    gb = lambda g: sum(nbytes[n] for n in keep if group(n) == g)
    manifest = {
        "source": "zai-org/GLM-5.3-Flash (FP8)", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": ("all source tensors except model.language_model.layers.{3..44}.mlp.experts.* ; "
                 "byte-for-byte passthrough"),
        "n_source_tensors": len(wm), "n_excluded": n_excl, "n_tensors": len(keep),
        "n_verified_identical": n_ok, "total_tensor_bytes": total,
        "total_file_bytes": sum(f["bytes"] for f in files),
        "group_bytes": {g: gb(g) for g in ("nonexpert", "vision", "mtp_experts")},
        "dtype_counts": dict(Counter(hdr[n]["dtype"] for n in keep)),
        "categories": dict(sorted(cats.items())),
        "uncategorized": [n for n in keep if category(n, layer_types) == "UNCATEGORIZED"],
        "mtp45": {"n_tensors": len(mtp), "n_source": len(src_mtp),
                  "bytes": sum(nbytes[n] for n in mtp),
                  "routed_expert_bytes": gb("mtp_experts"),
                  "all_present": set(mtp) == set(src_mtp),
                  "routed_experts_file_group": "mtp_experts-*.safetensors"},
        "files": files,
    }
    json.dump(manifest, open(os.path.join(a.out, "nonexpert_manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    assert not manifest["uncategorized"]


if __name__ == "__main__":
    main()
