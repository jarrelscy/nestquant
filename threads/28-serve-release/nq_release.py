"""Thread 28: export NestQuant layers (nestquant-v1, L{L}/tp{s}.pt|.safetensors + manifest.json) into the layout the
SM120 streaming server reads (streaming/repack.py's NQ_REPACK dir), one layer at a time, content-hashed.

  python nq_release.py ROOT OUT --tp 4 [--layers 3-77] [--jobs 12] [--threads 4] [--force]
  python nq_release.py ROOT OUT --tp 4 --index-only        # rebuild rank{r}.json / manifest.json / COMPLETE from layers/

OUT/serving/tp{N}/                                  (N = TP degree; = an NQ_REPACK dir once rank{r}.bin is assembled)
  rank{r}/L{L}.bin      P4 records of layer L, rank r: NE records x rec_bytes, expert order 0..NE-1 (p4rec.pack).
                        = exactly the bytes at offset (L-L0)*NE*rec_bytes of the assembled rank{r}.bin (nq_assemble.py)
  res/rank{r}/L{L}.pt   resident planes, nq-res-v1 (resident.save)
  layers/L{L}.json      per-layer block: every file's bytes + sha256, layout, default allocation, routing counts;
                        layer_hash = sha256 of this block's canonical json without the layer_hash/time fields
  rank{r}.json          streaming/stream_engine.RankFile index (format nq-p4rec-v1, rec_bytes, seg, L0, NE,
                        layers{L: experts, rg, rd}) + per layer: file, offset, bytes, sha256, res, res_sha256, layer_hash
  manifest.json         formats, TP, H/I, layout, per-layer layer_hash, default allocation (fixed level-4 set), floating
                        default, n_routed counts, fixed-set provenance
  COMPLETE              written only when every layer L0..L1 is present: {L: layer_hash} (+ manifest sha256)
Record layout (format nq-p4rec-v1, from streaming/p4rec.py): segments gu.p4 | gu.d4 | dn.p4 | dn.d4 | lr4 (U4_g|U4_u|U4_d
fp16 at RMAX = 4), each 256 B aligned, record rounded to 4 KiB, identical for every record of a TP degree. Any change to
that layout must bump REC_FORMAT (nq-p4rec-v2) here and in the model card; check_layout() enforces it.

Reuses nqload.RankLayer / p4rec.layout,pack / resident.save unchanged (runs on CPU: the sm120 extension import in moe.py is
stubbed, only its pure-python helpers are used). Idempotent: a (layer, rank) whose source fingerprint (layer manifest.json
sha256 + byte-determining code hashes) matches and whose outputs exist with the recorded sizes is skipped.
Refit hook: after a layer's manifest is finalized, `nq_release.py ROOT OUT --tp 4 --layers L` then nq_upload.py.
"""
import os, sys, json, time, hashlib, argparse, types
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
REC_FORMAT = "nq-p4rec-v1"
RES_FORMAT = "nq-res-v1"
SERVE_FORMAT = "nq-serve-v1"
L0, L1, NE = 3, 77, 256
EXPORT_VERSION = 1                                       # bump when export_rank's output bytes change
CODE = ["streaming/p4rec.py", "streaming/resident.py", "sm120/nqload.py", "sm120/moe.py",   # the code that determines
        "threads/12-reference-encoder/nq_decode.py", "threads/25-campaign/nq25_st.py"]      # the exported bytes


def _imports():
    if "build" not in sys.modules:                       # moe.py compiles the sm120 extension at import; we only need
        sys.modules["build"] = types.SimpleNamespace(get=lambda *a, **k: None)   # its pure-python pack helpers
    for p in (R + "/streaming", R + "/sm120"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import nqload, p4rec, resident
    return nqload, p4rec, resident


def sha256_file(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def layer_hash(block):
    return hashlib.sha256(canon({k: v for k, v in block.items() if k not in ("layer_hash", "time")}).encode()).hexdigest()


def code_id():
    h = hashlib.sha256()
    for f in CODE:
        h.update(f.encode()); h.update(open(f"{R}/{f}", "rb").read())
    return h.hexdigest()[:16]


def tpdir(out, tp):
    return f"{out}/serving/tp{tp}"


def layer_src(root, L):
    """-> (layer dir, manifest dict, manifest sha256)"""
    NQ, _, _ = _imports()
    d = NQ.layer_dir(root, L)
    b = open(f"{d}/manifest.json", "rb").read()
    return d, json.loads(b), hashlib.sha256(b).hexdigest()


_FS = {}
def fixed_set_info(man, L):
    """default allocation of layer L from its manifest, cross-checked against the fixed_set.json it names (sha256 must
    match), plus that file's floating_default / n_routed for L. Never truncates: the manifest's level4_experts is the set."""
    da = man["default_allocation"]
    lv4 = [int(e) for e in da["level4_experts"]]
    assert len(lv4) == da["K"] == len(set(lv4)), (L, len(lv4), da["K"])
    src = da["source"]
    if src not in _FS:
        b = open(src, "rb").read(); _FS[src] = (hashlib.sha256(b).hexdigest(), json.loads(b))
    sh, fj = _FS[src]
    assert sh == da["sha256"], f"L{L}: {src} sha {sh[:16]} != manifest default_allocation sha {da['sha256'][:16]}"
    assert [int(e) for e in fj["fixed_set"][str(L)]] == lv4, f"L{L}: fixed_set.json set != manifest level4_experts"
    fd = [int(e) for e in fj["floating_default"][str(L)]]
    assert not set(fd) & set(lv4)
    out = dict(level4_experts=lv4, K=da["K"], schema=da.get("schema"), fixed_set_sha256=sh, rule=da.get("rule"),
               floating_default=fd, n_routed=[int(x) for x in fj["n_routed"][str(L)]],
               stats_version=da.get("stats_version"))
    if "vision_n_routed" in fj:
        out["vision_n_routed"] = [int(x) for x in fj["vision_n_routed"][str(L)]]
    return out


def expected_layout(man, tp):
    """layout of a TP degree from the manifest alone (p4rec.layout needs an expert; sizes only depend on K / N / rk)"""
    NQ, PR, _ = _imports()
    from moe import proj_sizes, RK_OF
    pm = man["proj_meta"]; g = 8 // tp
    H, I = pm["down"]["n"], pm["down"]["k"] * g // 8
    rk = lambda p: RK_OF[float(pm[p]["res_rule"]["K"])]
    assert rk("gate") == rk("up")
    zg, zd = proj_sizes(2 * I, H, None, rk("gate")), proj_sizes(H, I, None, rk("down"))
    pad = lambda n: (n + 3) // 4 * 4                     # p4 tensors are int32 (pack_words pads to a word)
    ex = types.SimpleNamespace(gu=types.SimpleNamespace(p4=types.SimpleNamespace(numel=lambda: pad(zg["p4"]) // 4),
                                                        d4=types.SimpleNamespace(numel=lambda: zg["d4"] // 4)),
                               dn=types.SimpleNamespace(p4=types.SimpleNamespace(numel=lambda: pad(zd["p4"]) // 4),
                                                        d4=types.SimpleNamespace(numel=lambda: zd["d4"] // 4)))
    lay = PR.layout(ex, H, I)
    return dict(H=H, I=I, rec_bytes=lay["rec_bytes"], seg={k: list(v) for k, v in lay["seg"].items()},
                seg_order=list(PR.SEGS), seg_align=PR.SEG_ALIGN, rec_align=PR.ALIGN, rmax=PR.RMAX)


def check_layout(lay):
    """format nq-p4rec-v1 invariants"""
    _, PR, _ = _imports()
    assert lay["seg_order"] == ["gu.p4", "gu.d4", "dn.p4", "dn.d4", "lr4"] and lay["seg_align"] == 256 and \
        lay["rec_align"] == 4096 and lay["rmax"] == 4, f"record layout changed: bump REC_FORMAT ({lay})"
    o = 0
    for k in lay["seg_order"]:
        so, n = lay["seg"][k]
        assert so == (o + 255) // 256 * 256 and so % 256 == 0, (k, so, o)
        o = so + n
    assert lay["rec_bytes"] == (o + 4095) // 4096 * 4096, (lay["rec_bytes"], o)
    if "I" in lay:
        assert lay["seg"]["lr4"][1] == 4 * (2 * lay["I"] + lay["H"]) * 2


# ------------------------------------------------------------------------------------------------ per (layer, rank)
def export_rank(root, out, tp, L, r, fp, threads=4, force=False):
    """writes rank{r}/L{L}.bin + res/rank{r}/L{L}.pt, returns the rank entry (also cached in _work/L{L}.r{r}.json)"""
    import torch
    torch.set_num_threads(threads)
    NQ, PR, RS = _imports()
    T = tpdir(out, tp)
    wp = f"{T}/_work/L{L}.r{r}.json"; bp = f"{T}/rank{r}/L{L}.bin"; rp = f"{T}/res/rank{r}/L{L}.pt"
    if not force and os.path.exists(wp):
        e = json.load(open(wp))
        if e["fingerprint"] == fp and os.path.exists(bp) and os.path.getsize(bp) == e["rec"]["bytes"] \
                and os.path.exists(rp) and os.path.getsize(rp) == e["res"]["bytes"]:
            return e, True
    for p in (bp, rp, wp):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    t = time.time()
    RL = NQ.RankLayer(root, L, r, tp, dev="cpu")
    assert RL.experts == list(range(NE)), f"L{L}: experts != 0..{NE-1}"
    d, man, _ = layer_src(root, L)
    lay = PR.layout(RL.ex[0], RL.H, RL.I); exp = expected_layout(man, tp)
    assert {k: list(v) for k, v in lay["seg"].items()} == exp["seg"] and lay["rec_bytes"] == exp["rec_bytes"], (lay, exp)
    h = hashlib.sha256()
    with open(bp + ".tmp", "wb") as f:
        for E in range(NE):
            ex = RL.ex[E]
            assert PR.layout(ex, RL.H, RL.I)["seg"] == lay["seg"], (L, E)
            b = PR.pack(ex, lay); h.update(b); f.write(b)
        f.flush(); os.fsync(f.fileno())
    nb = os.path.getsize(bp + ".tmp"); assert nb == NE * lay["rec_bytes"]
    sp = f"{T}/_work/stage/rank{r}/L{L}.pt"; os.makedirs(os.path.dirname(sp), exist_ok=True)
    RS.save(RL, sp)                                      # same basename as the target: torch.save's zip prefix is the file
    rs = sha256_file(sp)                                 # name, so the bytes equal repack.py's res/rank{r}/L{L}.pt exactly
    os.replace(bp + ".tmp", bp); os.replace(sp, rp)
    e = dict(L=L, rank=r, tp=tp, fingerprint=fp,
             rec=dict(file=f"rank{r}/L{L}.bin", bytes=nb, sha256=h.hexdigest(), offset=(L - L0) * NE * lay["rec_bytes"]),
             res=dict(file=f"res/rank{r}/L{L}.pt", bytes=os.path.getsize(rp), sha256=rs),
             rg={str(E): RL.ex[E].rg for E in RL.experts}, rd={str(E): RL.ex[E].rd for E in RL.experts},
             H=RL.H, I=RL.I, secs=round(time.time() - t, 1))
    json.dump(e, open(wp + ".tmp", "w")); os.replace(wp + ".tmp", wp)
    return e, False


def _job(a):
    root, out, tp, L, r, fp, threads, force = a
    e, skipped = export_rank(root, out, tp, L, r, fp, threads, force)
    return L, r, skipped, e["secs"]


def fingerprint(man_sha, cid):
    """identity of a (layer, rank) export: source layer manifest (it lists every shard's sha256) + the byte-determining
    code + formats. The fixed set is NOT in it: it does not change the record/resident bytes, only layers/L{L}.json
    (finish_layer re-reads it every run), so a changed fixed set re-indexes without re-exporting."""
    return hashlib.sha256(f"{man_sha}|{cid}|{REC_FORMAT}|{RES_FORMAT}|{EXPORT_VERSION}".encode()).hexdigest()[:24]


def finish_layer(root, out, tp, L):
    """all ranks of L exported -> layers/L{L}.json (the per-layer block + layer_hash)"""
    T = tpdir(out, tp)
    d, man, msha = layer_src(root, L); fs = fixed_set_info(man, L); lay = expected_layout(man, tp); check_layout(lay)
    ranks = {}
    for r in range(tp):
        e = json.load(open(f"{T}/_work/L{L}.r{r}.json"))
        ranks[str(r)] = dict(rec=e["rec"], res=e["res"], rg=e["rg"], rd=e["rd"])
    blk = dict(format=SERVE_FORMAT, rec_format=REC_FORMAT, res_format=RES_FORMAT, L=L, tp=tp, L0=L0, NE=NE,
               layout=lay, ranks=ranks, default_allocation=fs,
               source=dict(format=man["format"], manifest_sha256=msha,
                           files={k: v["sha256"] for k, v in man["files"].items()},
                           config_id=man.get("campaign", {}).get("config_id")),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    blk["layer_hash"] = layer_hash(blk)
    p = f"{T}/layers/L{L}.json"; os.makedirs(os.path.dirname(p), exist_ok=True)
    old = json.load(open(p)) if os.path.exists(p) else None
    if old is not None and old.get("layer_hash") == blk["layer_hash"]:
        return old                                       # unchanged: keep the file bytes (its time field) as they are
    json.dump(blk, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    return blk


def build_index(out, tp, only=None, dest=None):
    """layers/*.json -> rank{r}.json, manifest.json, COMPLETE (only when L0..L1 all present). Deterministic given the
    layer blocks (no timestamps), so an unchanged release re-indexes to identical bytes. only = subset of layers,
    dest = write there instead of the tp dir (the uploader's per-layer partial index; == the full one once complete)."""
    T = tpdir(out, tp)
    blks = {}
    for f in sorted(os.listdir(f"{T}/layers")):
        if f.startswith("L") and f.endswith(".json"):
            b = json.load(open(f"{T}/layers/{f}")); assert layer_hash(b) == b["layer_hash"], f
            if only is None or b["L"] in only:
                blks[b["L"]] = b
    W = dest or T
    os.makedirs(W, exist_ok=True)
    Ls = sorted(blks)
    lay = blks[Ls[0]]["layout"]
    for L in Ls:
        assert blks[L]["layout"] == lay and blks[L]["tp"] == tp and blks[L]["rec_format"] == REC_FORMAT, L
    rb = lay["rec_bytes"]
    for r in range(tp):
        idx = dict(format=REC_FORMAT, serve_format=SERVE_FORMAT, tp=tp, rank=r, L0=L0, NE=NE, rec_bytes=rb,
                   seg={k: v for k, v in lay["seg"].items()}, seg_order=lay["seg_order"], rmax=lay["rmax"],
                   H=lay["H"], I=lay["I"], bin=f"rank{r}.bin", bin_bytes=(L1 - L0 + 1) * NE * rb, layers={})
        for L in Ls:
            e = blks[L]["ranks"][str(r)]
            assert e["rec"]["offset"] == (L - L0) * NE * rb and e["rec"]["bytes"] == NE * rb
            idx["layers"][str(L)] = dict(experts=list(range(NE)), rg=e["rg"], rd=e["rd"], file=e["rec"]["file"],
                                         offset=e["rec"]["offset"], bytes=e["rec"]["bytes"], sha256=e["rec"]["sha256"],
                                         res=e["res"]["file"], res_bytes=e["res"]["bytes"], res_sha256=e["res"]["sha256"],
                                         layer_hash=blks[L]["layer_hash"])
        _wjson(f"{W}/rank{r}.json", idx)
    fss = sorted({blks[L]["default_allocation"]["fixed_set_sha256"] for L in Ls})
    man = dict(format=SERVE_FORMAT, rec_format=REC_FORMAT, res_format=RES_FORMAT, source_format="nestquant-v1",
               tp=tp, L0=L0, L1=L1, NE=NE, layout=lay, layers_present=Ls,
               layer_hash={str(L): blks[L]["layer_hash"] for L in Ls},
               files=dict(records="rank{r}/L{L}.bin", resident="res/rank{r}/L{L}.pt", layer="layers/L{L}.json",
                          index="rank{r}.json", assembled="rank{r}.bin (nq_assemble.py; not published)"),
               fixed_set_sha256=fss,
               default_allocation={str(L): blks[L]["default_allocation"]["level4_experts"] for L in Ls},
               floating_default={str(L): blks[L]["default_allocation"]["floating_default"] for L in Ls},
               n_routed={str(L): blks[L]["default_allocation"]["n_routed"] for L in Ls},
               notes=["record of (L, E) in the assembled rank{r}.bin at ((L-L0)*NE + E) * rec_bytes; rank{r}/L{L}.bin "
                      "holds exactly the NE records of layer L (its offset in rank{r}.bin = rank{r}.json layers[L].offset)",
                      "default_allocation = fixed level-4 set per layer (always resident at level 4); floating_default = "
                      "top non-fixed experts by n_routed on the text calibration capture (seeds the floating 4-bit set)",
                      "the release is complete only when COMPLETE exists and every layer_hash in it matches layers/L{L}.json"])
    _wjson(f"{W}/manifest.json", man)
    full = Ls == list(range(L0, L1 + 1))
    cp = f"{W}/COMPLETE"
    if full:
        _wjson(cp, complete_doc(W, blks))
    elif os.path.exists(cp):
        os.remove(cp)
    return Ls, full


def complete_doc(T, blks):
    return dict(format=SERVE_FORMAT, rec_format=REC_FORMAT, layers={str(L): blks[L]["layer_hash"] for L in sorted(blks)},
                manifest_sha256=sha256_file(f"{T}/manifest.json"),
                index_sha256={f"rank{r}.json": sha256_file(f"{T}/rank{r}.json") for r in range(blks[min(blks)]["tp"])})


def _wjson(p, d):
    s = json.dumps(d, indent=1, sort_keys=True)
    if os.path.exists(p) and open(p).read() == s:
        return
    open(p + ".tmp", "w").write(s); os.replace(p + ".tmp", p)


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, b = (part.split("-") + [part])[:2]; out += list(range(int(a), int(b) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--layers", default=f"{L0}-{L1}"); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--force", action="store_true")
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    if not a.index_only:
        cid = code_id(); jobs = []
        for L in parse_layers(a.layers):
            d, man, msha = layer_src(a.root, L)
            assert man["format"] == "nestquant-v1" and all(os.path.exists(f"{d}/{f}") for f in man["files"]), f"L{L} incomplete"
            fixed_set_info(man, L); check_layout(expected_layout(man, a.tp))
            fp = fingerprint(msha, cid)
            jobs += [(a.root, a.out, a.tp, L, r, fp, a.threads, a.force) for r in range(a.tp)]
        todo = {}
        for j in jobs:
            todo.setdefault(j[3], set()).add(j[4])
        t0 = time.time()
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs, maxtasksperchild=1) as pool:
            for L, r, sk, secs in pool.imap_unordered(_job, jobs):
                todo[L].discard(r)
                print(f"L{L} rank{r}: {'unchanged' if sk else f'exported in {secs}s'} ({time.time()-t0:.0f}s)", flush=True)
                if not todo[L]:
                    b = finish_layer(a.root, a.out, a.tp, L)
                    print(f"L{L}: layer_hash {b['layer_hash'][:16]}", flush=True)
    Ls, full = build_index(a.out, a.tp)
    print(f"index: tp{a.tp} {len(Ls)} layers, COMPLETE={'yes' if full else 'no'}", flush=True)


if __name__ == "__main__":
    main()
