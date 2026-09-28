"""Thread 25: safetensors container for the nq_layer TP8 shard files (lead decision 3; the .pt stay local only).

  layers/L{L}/tp{s}.safetensors   tensors only; key  E{e}.{proj}.{plane}   (proj gate|up|down; plane = nq_layer.split_expert's keys:
                                  base p4 word suh2 svh2 suh4 svh4 [var] [lrV lrU2 lrU4 ...], whatever T12 emits)
  metadata (str->str): format=nestquant-v1-tp-safetensors, layer, shard, tp, experts (comma list, dict order),
                       tree = json skeleton of the dict (key order + key types; tensors as {"T": key}, any non-tensor
                       value verbatim, e.g. T12's lrV_from) -> the loader rebuilds the tp{s}.pt dict exactly.
  Every non-tensor field lives in manifest.json (nq_layer's; files[] then lists the .safetensors, pt_files[] the .pt).

API
  load_shard(path)            -> {E: {proj: {plane: tensor}}}   == torch.load(tp{s}.pt) (same keys, order, dtypes, bytes)
  assemble(root, L, E)        -> nq_decode artifact of expert E from the 8 .safetensors (T12's nq_layer.assemble)
  convert(root, L)            -> writes tp{s}.safetensors, verifies round trip vs tp{s}.pt, rewrites manifest files[]
  check_decode(root, L, n)    -> every expert: artifact assembled from .safetensors == E{E}.pt artifact (raw bytes);
                                 decode L2+L4 equal on n seeded experts (None = all)

CLI: python nq25_st.py --root ROOT --layer L [--no-check]
Round trip note: tensor-level identity (keys/order/dtype/shape/bytes). The .pt's own file bytes are not reproducible
from any loader (torch.save stores the whole parent storage of view tensors, e.g. svh2 = slice of the full svh).
"""
import os, sys, json, time, hashlib, argparse
import torch
from safetensors.torch import save_file, safe_open
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
sys.path.insert(0, T12)
NSH = 8
FORMAT = "nestquant-v1-tp-safetensors"
PROJ = ("gate", "up", "down")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _tree(x, pre, ten):
    """nested dict -> json skeleton; tensors go to ten[key] and appear as {"T": key}; other values verbatim."""
    if isinstance(x, dict):
        return {"D": [[k, _tree(v, f"{pre}.{k}", ten)] for k, v in x.items()]}      # keeps key order + key types
    if torch.is_tensor(x):
        t = x.contiguous()
        ten[pre] = t.clone() if t.untyped_storage().nbytes() != t.numel() * t.element_size() else t
        return {"T": pre}
    return {"V": x}


def _untree(n, get):
    if "D" in n:
        return {k: _untree(v, get) for k, v in n["D"]}
    if "T" in n:
        return get(n["T"])
    return n["V"]


def save_shard(d, path, L, s):
    """d = nq_layer's tp{s}.pt dict {E: {proj: {plane: tensor | value}}}; key E{E}.{proj}.{plane}."""
    ten = {}
    tree = {"D": [[E, _tree(ex, f"E{E}", ten)] for E, ex in d.items()]}
    meta = dict(format=FORMAT, layer=str(L), shard=str(s), tp=str(NSH), experts=",".join(str(E) for E in d),
                tree=json.dumps(tree, separators=(",", ":")))
    save_file(ten, path + ".tmp", metadata=meta); os.chmod(path + ".tmp", 0o644); os.replace(path + ".tmp", path)


def load_shard(path, device="cpu"):
    """-> exactly the tp{s}.pt dict (keys, key order, key types, tensors, non-tensor values)."""
    with safe_open(path, framework="pt", device=device) as f:
        m = f.metadata()
        assert m.get("format") == FORMAT, m.get("format")
        return _untree(json.loads(m["tree"]), f.get_tensor)


def assemble(root, L, E, man=None, parts=None):
    """T12's nq_layer.assemble (single source of truth for the shard -> artifact rule, incl. future planes), reading
    the .safetensors instead of tp{s}.pt. parts: preloaded [load_shard(tp s)] to avoid re-reads."""
    import nq_layer as NL
    cache = {s: p for s, p in enumerate(parts)} if parts is not None else {}
    orig = torch.load

    def ld(f, *a, **k):
        b = os.path.basename(str(f))
        if b.startswith("tp") and b.endswith(".pt") and os.path.dirname(str(f)) == f"{root}/L{L}":
            s = int(b[2:-3])
            if s not in cache:
                cache[s] = load_shard(f"{root}/L{L}/tp{s}.safetensors")
            return cache[s]
        return orig(f, *a, **k)
    torch.load = ld
    try:
        return NL.assemble(root, L, E)
    finally:
        torch.load = orig


def same_tree(a, b, path="", ordered=True):
    """identical nested dict/list: same keys (in the same order if ordered), tensors equal dtype/shape/raw bytes,
    other values equal. -> list of differing paths."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or (list(a) != list(b) if ordered else set(a) != set(b)):
            return [f"{path} keys"]
        return [x for k in a for x in same_tree(a[k], b[k], f"{path}.{k}", ordered)]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [f"{path} len"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in same_tree(u, v, f"{path}[{i}]", ordered)]
    if not torch.is_tensor(a):
        return [] if (not torch.is_tensor(b) and type(a) is type(b) and a == b) else [f"{path} value"]
    if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))):
        return [path]
    return []


def convert(root, L):
    d = f"{root}/L{L}"
    p = f"{d}/manifest.json"
    man = json.load(open(p))
    pt_files = man.get("pt_files") or {f: v for f, v in man["files"].items() if f.endswith(".pt")}
    assert len(pt_files) == NSH, pt_files
    files, bad = {}, []
    for s in range(NSH):
        pt = torch.load(f"{d}/tp{s}.pt", weights_only=False)
        f = f"{d}/tp{s}.safetensors"
        save_shard(pt, f, L, s)
        bad += [f"tp{s}{x}" for x in same_tree(pt, load_shard(f))]
        files[f"tp{s}.safetensors"] = dict(sha256=sha256(f), bytes=os.path.getsize(f))
        del pt
    ok = not bad
    print(f"[L{L}] safetensors round-trip == tp.pt (keys, order, dtype, shape, bytes): {ok} ({len(bad)} mismatches)"
          f"{' ' + str(bad[:5]) if bad else ''}", flush=True)
    if ok:
        man.update(files=files, pt_files=pt_files, container=dict(
            format=FORMAT, key="E{e}.{proj}.{plane}", loader="threads/25-campaign/nq25_st.py (load_shard / assemble)",
            note="tensors in the file body; the dict skeleton (+ any non-tensor shard value) in the safetensors metadata \"tree\"; everything else in this manifest; planes as nq_layer.split_expert"))
        json.dump(man, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    return ok


def check_decode(root, L, n_decode=None, seed=0):
    """every expert: the artifact assembled from the .safetensors == E{E}.pt's artifact (all tensors raw bytes + proj
    meta; implies equal decode). Plus a real decode compare (L2 and L4) on n_decode seeded experts (None = all)."""
    import random
    import nq_decode as D
    man = json.load(open(f"{root}/L{L}/manifest.json"))
    parts = [load_shard(f"{root}/L{L}/tp{s}.safetensors") for s in range(NSH)]
    exps = list(man["experts"])
    dec = set(exps if n_decode is None else random.Random(seed * 1000 + L).sample(exps, min(n_decode, len(exps))))
    import nq_layer as NL
    have_pt = all(os.path.exists(f"{root}/L{L}/tp{s}.pt") for s in range(NSH))
    if have_pt:          # T12's assemble torch.loads all 8 tp.pt (~5 GB) per expert: serve them from a per-layer cache
        _load, cache = torch.load, {}              # (same objects, same T12 code path; 256x fewer unpickles)
        tp = {os.path.realpath(f"{root}/L{L}/tp{s}.pt") for s in range(NSH)}

        def cached_load(f, *a, **k):
            rp = os.path.realpath(f) if isinstance(f, str) else None
            if rp in tp:
                if rp not in cache:
                    cache[rp] = _load(f, *a, **k)
                return cache[rp]
            return _load(f, *a, **k)
        torch.load = cached_load
    bad_art, bad_dec, t12_only = [], [], {}
    for E in exps:
        art = torch.load(f"{root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        re = assemble(root, L, E, parts=parts)
        d_st = same_tree({pn: art[pn] for pn in re}, re, ordered=False)
        if have_pt:          # reference = T12's own assemble from tp.pt: the st path must equal it exactly; E.pt diffs
            rp = NL.assemble(root, L, E)            # that T12's shard format itself has (e.g. dropped meta keys) are
            d_pt = same_tree({pn: art[pn] for pn in rp}, rp, ordered=False)       # reported, not failed
            if same_tree(rp, re, ordered=False) or set(d_st) - set(d_pt):
                bad_art.append(E)
            for x in d_pt:
                t12_only.setdefault(x, []).append(E)
        elif d_st:
            bad_art.append(E)
        if E in dec:
            for Lv in (2, 4):
                if not all(torch.equal(x, y) for x, y in zip(D.decode_expert(art, Lv), D.decode_expert(re, Lv))):
                    bad_dec.append((E, Lv))
    if have_pt:
        torch.load = _load
    ok = not bad_art and not bad_dec
    print(f"[L{L}] safetensors artifact == E.pt artifact: {not bad_art} ({len(bad_art)}/{len(exps)} mismatches)", flush=True)
    if t12_only:
        print(f"[L{L}] note: E.pt vs T12 tp.pt assemble differ (inherent to T12's shard format, identical in the "
              f"safetensors): { {k: len(v) for k, v in t12_only.items()} }", flush=True)
    print(f"[L{L}] safetensors decode == artifact decode: {not bad_dec} ({len(bad_dec)} mismatches, "
          f"{len(dec)} experts decoded at L2+L4)", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--n-decode", type=int, default=None, help="decode-compare this many seeded experts (default all)")
    a = ap.parse_args()
    t0 = time.time()
    ok = convert(a.root, a.layer)
    if ok and not a.no_check:
        ok = check_decode(a.root, a.layer, a.n_decode)
    print(f"[L{a.layer}] st {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0 if ok else 1)
