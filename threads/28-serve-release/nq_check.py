"""Validator for a NestQuant serving release (thread 28, format nq-serve-v1 / nq-p4rec-v1 / nq-res-v1).

  python nq_check.py DIR [--ref ROOT] [--experts 3] [--layers 3-77] [--ranks all|rot] [--hash all|none] [--dev cpu|cuda]
                         [--jobs 32] [--allow-incomplete]
DIR = serving/tp{N}/ (local build or an HF download); ROOT = the nestquant-v1 source (L{L}/ or layers/L{L}/; default:
DIR/../../ = the repo root of an HF download, whose layers/ are the reference safetensors).
 (1) structure: COMPLETE lists L0..L1 with layer_hash == hash of layers/L{L}.json, manifest/index sha256 == COMPLETE;
     layout obeys nq-p4rec-v1 (segment order, 256 B segments, 4 KiB records, RMAX 4) and == the layout the reference
     manifest implies; every index entry: offset == (L-L0)*NE*rec_bytes, bytes == NE*rec_bytes == file size, res size;
     index/manifest/layer blocks agree; default allocation == the reference layer manifest's level4_experts (no
     truncation), floating_default disjoint from it, n_routed has NE counts. An assembled rank{r}.bin, if present, must
     have the index's bin_bytes and hold each checked record at its offset.
 (2) hashes: sha256 of every record block and resident file (--hash all, threaded) vs the index.
     With --layers (a partial download), data files absent for layers outside --layers are reported and skipped; the
     metadata of every layer and every data file that is present are still checked.
 (3) decode, a few experts per layer (seeded; one fixed-set expert, one lr-rank-0 expert when present, the rest random),
     on rank L % N (--ranks rot) or every rank (--ranks all):
     (a) record bytes == p4rec.pack of the reference expert (nqload.RankLayer on the reference shards), byte-exact
     (b) resident planes (resident.load) == the reference kernel planes, bit-exact (base, var, sc2, sc4, lr, ranks)
     (c) independent decode: dense weights from the PUBLISHED planes (resident base/var + record P4 words unpacked from
         the p4rec sub-array layout + record d4 block words) through moe.dense_W at levels 2 and 4 == the reference
         decoder (threads/12 nq_decode ring_levels, rotated Q2/Q4) of the reference shards, bitwise, gate|up and down;
         record U4 segment == reference lrU4, resident lr == reference lrV|lrU2
Prints one line per failure and 'NQ CHECK PASS' / 'NQ CHECK FAIL'; exit code 0 / 1. GPU use (--dev cuda) is capped at
12 GB via set_per_process_memory_fraction."""
import os, sys, json, time, random, hashlib, argparse, types
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nq_release as NR

BAD = []
def bad(msg):
    BAD.append(msg); print("  FAIL " + msg, flush=True)


def unpack_words(p4, bits, nrec):
    """inverse of moe.pack_words: int32 sub-array layout -> [nrec, NW] int64 (uint32 values)"""
    import torch
    from moe import split_bits
    n4, n2, n1, nh, T = split_bits(bits); NW = (bits + 31) // 32
    b = p4.contiguous().view(torch.uint8).cpu().long(); o = 0
    w = torch.zeros(nrec, NW, dtype=torch.int64)
    def take(nb, cnt):
        nonlocal o
        x = b[o:o + nb * cnt].view(cnt, nb); o += nb * cnt
        return (x << (8 * torch.arange(nb))).sum(1)
    k = 0
    if n4:
        w[:, :4 * n4] = take(4, nrec * 4 * n4).view(nrec, 4 * n4); k = 4 * n4
    if n2:
        w[:, k:k + 2] = take(4, nrec * 2).view(nrec, 2); k += 2
    if n1:
        w[:, k] = take(4, nrec); k += 1
    if nh:
        w[:, k] = take(2, nrec)
    if T:
        A = bits - T; nb = (nrec * T + 32) // 8
        bb = b[o:o + nb]; bit = ((bb[:, None] >> torch.arange(8)) & 1).reshape(-1)[:nrec * T].view(nrec, T)
        w[:, A // 32] |= (bit << torch.arange(T)).sum(1) << (A % 32); o += nb
    assert o == b.numel(), (o, b.numel())
    return w


def published_proj(res_p, rec_bytes, seg, name, N, K, dev):
    """kernel Proj namespace from the published planes (resident: base/var/rk; record: p4/d4 segments)"""
    import torch
    from moe import proj_sizes, rbits, pack_words
    z = proj_sizes(N, K, None, res_p.rk); nrec = z["nrec_r"]
    o, n = seg[name + ".p4"]; p4 = torch.frombuffer(bytearray(rec_bytes[o:o + n]), dtype=torch.int32)
    o, n = seg[name + ".d4"]; d4 = torch.frombuffer(bytearray(rec_bytes[o:o + n]), dtype=torch.int32).long()
    p4w = unpack_words(p4, rbits(res_p.rk), nrec)
    assert torch.equal(pack_words(p4w, rbits(res_p.rk)), p4), "unpack/pack round trip"
    return types.SimpleNamespace(N=N, K=K, rk=res_p.rk, z=z, flags=None, fl=None, base=res_p.base.to(dev),
                                 var=res_p.var.to(dev), p4w=p4w.to(dev), Mb=(d4 & 255).to(dev), Nn=((d4 >> 8) & 255).to(dev))


def check_structure(D, ref, a):
    import torch
    T = D; man = json.load(open(f"{T}/manifest.json")); tp = man["tp"]; L0, NE = man["L0"], man["NE"]
    for k, v in (("format", NR.SERVE_FORMAT), ("rec_format", NR.REC_FORMAT), ("res_format", NR.RES_FORMAT)):
        if man.get(k) != v:
            bad(f"manifest {k} {man.get(k)} != {v}")
    comp = json.load(open(f"{T}/COMPLETE")) if os.path.exists(f"{T}/COMPLETE") else None
    if comp is None and not a.allow_incomplete:
        bad("COMPLETE missing")
    want = list(range(L0, man["L1"] + 1))
    if comp is not None:
        if sorted(int(L) for L in comp["layers"]) != want:
            bad(f"COMPLETE layers != {L0}..{man['L1']}")
        if comp["manifest_sha256"] != NR.sha256_file(f"{T}/manifest.json"):
            bad("manifest.json sha256 != COMPLETE")
        for f, h in comp["index_sha256"].items():
            if NR.sha256_file(f"{T}/{f}") != h:
                bad(f"{f} sha256 != COMPLETE")
    lay = man["layout"]
    try:
        NR.check_layout(lay)
    except AssertionError as e:
        bad(f"layout: {e}")
    Ls = man["layers_present"]
    blks = {}
    for L in Ls:
        p = f"{T}/layers/L{L}.json"
        if not os.path.exists(p):
            bad(f"L{L}: layers/L{L}.json missing"); continue
        b = json.load(open(p)); blks[L] = b
        if NR.layer_hash(b) != b["layer_hash"]:
            bad(f"L{L}: layer block hash {NR.layer_hash(b)[:16]} != stored {b['layer_hash'][:16]}")
        if man["layer_hash"][str(L)] != b["layer_hash"] or (comp and comp["layers"].get(str(L)) != b["layer_hash"]):
            bad(f"L{L}: layer_hash differs between layers/, manifest, COMPLETE")
        if b["layout"] != lay:
            bad(f"L{L}: layout != manifest layout")
        da = b["default_allocation"]
        if man["default_allocation"][str(L)] != da["level4_experts"] or man["floating_default"][str(L)] != da["floating_default"] \
                or man["n_routed"][str(L)] != da["n_routed"]:
            bad(f"L{L}: manifest allocation/routing != layer block")
        if set(da["floating_default"]) & set(da["level4_experts"]) or len(da["n_routed"]) != NE or len(da["level4_experts"]) != da["K"]:
            bad(f"L{L}: allocation inconsistent")
        if ref is not None:
            try:
                d, rman, msha = NR.layer_src(ref, L)
                if [int(e) for e in rman["default_allocation"]["level4_experts"]] != da["level4_experts"]:
                    bad(f"L{L}: fixed set != reference manifest level4_experts")
                if msha != b["source"]["manifest_sha256"]:
                    bad(f"L{L}: reference manifest sha != layer block source (different encode)")
                el = NR.expected_layout(rman, tp)
                if el["rec_bytes"] != lay["rec_bytes"] or el["seg"] != lay["seg"]:
                    bad(f"L{L}: layout != layout implied by the reference manifest")
            except FileNotFoundError:
                if L in sel_layers(a, Ls):
                    bad(f"L{L}: reference layer missing")
    rb = lay["rec_bytes"]; files = []; sel = set(sel_layers(a, Ls)); skipped = [0]
    for r in range(tp):
        idx = json.load(open(f"{T}/rank{r}.json"))
        if idx["rec_bytes"] != rb or idx["seg"] != lay["seg"] or idx["L0"] != L0 or idx["NE"] != NE or idx["tp"] != tp \
                or idx["format"] != NR.REC_FORMAT:
            bad(f"rank{r}.json header != manifest")
        if sorted(int(L) for L in idx["layers"]) != Ls:
            bad(f"rank{r}.json layers != manifest layers_present")
        binp = f"{T}/{idx.get('bin', f'rank{r}.bin')}"
        if os.path.exists(binp) and os.path.getsize(binp) != idx["bin_bytes"]:
            bad(f"rank{r}.bin size {os.path.getsize(binp)} != {idx['bin_bytes']}")
        for L in Ls:
            e = idx["layers"].get(str(L))
            if e is None:
                continue
            be = blks.get(L, {}).get("ranks", {}).get(str(r), {})
            if e["offset"] != (L - L0) * NE * rb:
                bad(f"rank{r} L{L}: offset {e['offset']} != {(L - L0) * NE * rb}")
            if e["bytes"] != NE * rb:
                bad(f"rank{r} L{L}: bytes {e['bytes']} != NE*rec_bytes")
            if be and (be["rec"]["sha256"] != e["sha256"] or be["res"]["sha256"] != e["res_sha256"] or
                       be["rec"]["offset"] != e["offset"] or be["rec"]["file"] != e["file"] or be["res"]["file"] != e["res"]):
                bad(f"rank{r} L{L}: index entry != layer block")
            if e["experts"] != list(range(NE)):
                bad(f"rank{r} L{L}: experts != 0..NE-1")
            src = blks.get(L, {}).get("source", {})
            if src and e.get("source_manifest_sha256", src["manifest_sha256"]) != src["manifest_sha256"]:
                bad(f"rank{r} L{L}: index source_manifest_sha256 != layer block")
            for k in NR.ROT_KEYS:                        # rotation fields: index == layer block (absent on both = default)
                if src and e.get(k) != src.get(k):
                    bad(f"rank{r} L{L}: index {k} {e.get(k)} != layer block {src.get(k)}")
            for f, nb, h in ((e["file"], e["bytes"], e["sha256"]), (e["res"], e["res_bytes"], e["res_sha256"])):
                p = f"{T}/{f}"
                if not os.path.exists(p):
                    if f == e["file"] and os.path.exists(binp):          # blocks may be --moved into rank{r}.bin
                        pass
                    elif a.layers and L not in sel:                      # partial download: only --layers need data
                        skipped[0] += 1
                    else:
                        bad(f"rank{r} L{L}: {f} missing")
                    continue
                if os.path.getsize(p) != nb:
                    bad(f"rank{r} L{L}: {f} size {os.path.getsize(p)} != {nb}"); continue
                files.append((p, h, f"rank{r} L{L}: {f}"))
    sp = f"{T}/{NR.STAMP}"
    if os.path.exists(sp):                               # artifact_stamp.json: run_c2.sh key over the source manifests
        st = json.load(open(sp)); i0 = json.load(open(f"{T}/rank0.json"))["layers"]   # (index == blocks checked above;
        msha = {int(L): e.get("source_manifest_sha256") for L, e in i0.items()}         #  a refit download has few blocks)
        if st.get("key") != NR.artifact_key(msha) or {int(k): v for k, v in st.get("layers", {}).items()} != msha:
            bad(f"{NR.STAMP} key {st.get('key')} != artifact key {NR.artifact_key(msha)} of the layer blocks")
        if comp is not None and "stamp_sha256" in comp and NR.sha256_file(sp) != comp["stamp_sha256"]:
            bad(f"{NR.STAMP} sha256 != COMPLETE")
    elif comp is not None and "stamp_sha256" in comp:
        bad(f"{NR.STAMP} missing (COMPLETE lists it)")
    if skipped[0]:
        print(f"    {skipped[0]} data files of layers outside --layers absent (partial download): not checked", flush=True)
    if a.hash == "all":
        t = time.time()
        with ThreadPoolExecutor(a.jobs) as ex:
            for (p, h, what), got in zip(files, ex.map(lambda x: NR.sha256_file(x[0]), files)):
                if got != h:
                    bad(f"{what} sha256 {got[:16]} != index {h[:16]}")
        print(f"(2) sha256 of {len(files)} files, {sum(os.path.getsize(f[0]) for f in files)/1e9:.1f} GB in {time.time()-t:.0f}s", flush=True)
    return man, blks


def sel_layers(a, Ls):
    return [L for L in Ls if L in set(NR.parse_layers(a.layers))] if a.layers else Ls


def read_rec(T, idx, L, E, rb):
    e = idx["layers"][str(L)]; p = f"{T}/{e['file']}"
    if os.path.exists(p):
        with open(p, "rb") as f:
            f.seek(E * rb); blk = f.read(rb)
    else:
        blk = None
    binp = f"{T}/{idx.get('bin', 'rank%d.bin' % idx['rank'])}"
    asm = None
    if os.path.exists(binp):
        with open(binp, "rb") as f:
            f.seek(e["offset"] + E * rb); asm = f.read(rb)
    return blk, asm


def pick_experts(rng, b, rg, rd, NE, n):
    """one fixed-set expert, one with an lr rank of 0 (if any), the rest random; sorted"""
    pick = [rng.choice(b["default_allocation"]["level4_experts"])]
    z = [E for E in range(NE) if rg[E] == 0 or rd[E] == 0]
    if z:
        pick.append(rng.choice(z))
    while len(pick) < n:
        E = rng.randrange(NE)
        if E not in pick:
            pick.append(E)
    return sorted(set(pick[:max(n, 1)]))


def check_decode(T, ref, man, blks, a):
    import torch
    NQ, PR, RS = NR._imports()
    from moe import dense_W
    dev = a.dev
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, 12 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(a.threads)
    tp, NE, lay = man["tp"], man["NE"], man["layout"]; rb = lay["rec_bytes"]; seg = {k: tuple(v) for k, v in lay["seg"].items()}
    H, I = lay["H"], lay["I"]; nexp = 0; t0 = time.time()
    for L in sel_layers(a, man["layers_present"]):
        rng = random.Random(1000 + L); b = blks[L]
        ranks = range(tp) if a.ranks == "all" else [L % tp]
        for r in ranks:
            idx = json.load(open(f"{T}/rank{r}.json")); e = idx["layers"][str(L)]
            rg, rd = {int(k): v for k, v in e["rg"].items()}, {int(k): v for k, v in e["rd"].items()}
            pick = pick_experts(rng, b, rg, rd, NE, a.experts)
            RL = NQ.RankLayer(ref, L, r, tp, experts=pick, dev="cpu")
            res, rH, rI = RS.load(f"{T}/{e['res']}", "cpu")
            if (rH, rI) != (H, I) or sorted(res) != list(range(NE)):
                bad(f"rank{r} L{L}: resident H/I/experts")
            for E in pick:
                w = f"rank{r} L{L} E{E}"; ref_ex = RL.ex[E]
                blk, asm = read_rec(T, idx, L, E, rb)
                want = PR.pack(ref_ex, dict(seg=seg, rec_bytes=rb))
                src = blk if blk is not None else asm
                if src is None:
                    bad(f"{w}: no record source"); continue
                if blk is not None and blk != want:
                    bad(f"{w}: (a) record block bytes != pack(reference)")
                if asm is not None and asm != want:
                    bad(f"{w}: (a) assembled rank{r}.bin record != pack(reference)")
                x = res[E]
                ok = x.rg == ref_ex.rg == rg[E] and x.rd == ref_ex.rd == rd[E] and x.gu.rk == ref_ex.gu.rk and x.dn.rk == ref_ex.dn.rk
                for nm, u, v in (("gu.base", x.gu.base, ref_ex.gu.base), ("gu.var", x.gu.var, ref_ex.gu.var),
                                 ("dn.base", x.dn.base, ref_ex.dn.base), ("dn.var", x.dn.var, ref_ex.dn.var),
                                 ("sc2", x.sc[2], ref_ex.sc[2]), ("sc4", x.sc[4], ref_ex.sc[4])):
                    if not torch.equal(u.cpu(), v.cpu()):
                        ok = False; bad(f"{w}: (b) resident {nm} != reference")
                if (x.lr is None) != (ref_ex.lr is None) or (x.lr is not None and not torch.equal(x.lr.cpu(), ref_ex.lr.cpu())):
                    ok = False; bad(f"{w}: (b) resident lr != reference")
                if not ok:
                    bad(f"{w}: (b) resident planes/ranks differ")
                # (c) independent decode of the published planes vs the reference decoder
                art = RL.art(E); _, _, (Qg, Qu, Qd) = NQ.kernel_expert(art, dev, want_Q=True)
                pg = published_proj(x.gu, src, seg, "gu", 2 * I, H, dev); pd = published_proj(x.dn, src, seg, "dn", H, I, dev)
                for lv in (2, 4):
                    Wgu = dense_W(pg, lv, 4, torch.float16); Wd = dense_W(pd, lv, 4, torch.float16)
                    if not torch.equal(Wgu, torch.cat([Qg[lv], Qu[lv]]).to(dev)):
                        bad(f"{w}: (c) level-{lv} gate|up decode != reference ({(Wgu != torch.cat([Qg[lv], Qu[lv]]).to(dev)).sum().item()} values)")
                    if not torch.equal(Wd, Qd[lv].to(dev)):
                        bad(f"{w}: (c) level-{lv} down decode != reference ({(Wd != Qd[lv].to(dev)).sum().item()} values)")
                    del Wgu, Wd
                o, n = seg["lr4"]; u4 = torch.frombuffer(bytearray(src[o:o + n]), dtype=torch.float16)
                def lrp(pn):
                    P = art[pn]; nn, kk = P["meta"]["n"], P["meta"]["k"]
                    if "lr" not in P["base"]:
                        zz = lambda c: torch.zeros(0, c, dtype=torch.float16); return zz(kk), zz(nn), zz(nn)
                    return P["base"]["lr"]["V"].cpu(), P["base"]["lr"]["U2"].cpu(), P["p4"]["lr"]["U4"].cpu()
                (Vg, U2g, U4g), (Vu, U2u, U4u), (Vd, U2d, U4d) = (lrp(p) for p in ("gate", "up", "down"))
                f = lambda *t: torch.cat([q.reshape(-1) for q in t]).half()
                U4 = f(U4g, U4u, U4d); m = U4.numel()
                if not torch.equal(u4[:m], U4) or u4[m:].abs().sum() != 0:
                    bad(f"{w}: (c) record U4 segment != reference lrU4 (+ zero pad)")
                if x.lr is not None and not torch.equal(x.lr.cpu(), f(Vg, U2g, U2u, Vd, U2d)):
                    bad(f"{w}: (c) resident lr != reference lrV|lrU2")
                nexp += 1
                del pg, pd
            del RL, res
            if dev == "cuda":
                torch.cuda.empty_cache()
        print(f"  L{L}: decode checked {len(pick)} experts x {len(list(ranks))} ranks ({time.time()-t0:.0f}s)", flush=True)
    print(f"(3) decode: {nexp} (layer, rank, expert) checks in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--ref", default=None); ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--layers", default=None); ap.add_argument("--ranks", default="rot", choices=["rot", "all"])
    ap.add_argument("--hash", default="all", choices=["all", "none"]); ap.add_argument("--dev", default="cpu")
    ap.add_argument("--jobs", type=int, default=32); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--allow-incomplete", action="store_true"); ap.add_argument("--no-decode", action="store_true")
    a = ap.parse_args(); T = os.path.abspath(a.dir)
    ref = a.ref or os.path.dirname(os.path.dirname(T))
    NQ, _, _ = NR._imports()
    if not os.path.exists(f"{NQ.layer_dir(ref, NR.L0)}/manifest.json"):
        print(f"no reference layers under {ref} (--ref ROOT)"); ref = None
    t = time.time()
    man, blks = check_structure(T, ref, a)
    print(f"(1) structure: tp{man['tp']} {len(man['layers_present'])} layers, rec_bytes {man['layout']['rec_bytes']} "
          f"({time.time()-t:.0f}s, {len(BAD)} failures so far)", flush=True)
    if not a.no_decode:
        if ref is None:
            bad("decode check needs the reference layers (--ref)")
        else:
            check_decode(T, ref, man, blks, a)
    print(json.dumps(dict(dir=T, tp=man["tp"], failures=len(BAD), secs=round(time.time() - t))))
    print("NQ CHECK", "PASS" if not BAD else "FAIL", flush=True)
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
