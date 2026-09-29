"""Check an existing NQ_REPACK record dir (built by streaming/repack.py, or by nq_assemble.py) against a NestQuant serving
release, read-only. Stdlib only; nothing is written.

  python nq_verify_repack.py --release RELDIR --repack /home/jarrelscy/nq-p4rec/hf [--layers 3-77] [--jobs 8]

RELDIR = a downloaded serving/tp{N}/ with at least rank{r}.json, manifest.json, COMPLETE (the blocks are not needed:
the release hashes are in rank{r}.json). For every rank r and every layer L that the record dir's rank{r}.json lists
(or --layers): sha256 of rank{r}.bin[offset : offset + NE*rec_bytes] (offset = (L-L0)*NE*rec_bytes) must equal the
release's rank{r}.json layers[L].sha256, and sha256 of res/rank{r}/L{L}.pt must equal layers[L].res_sha256; the header
(format, tp, rank, L0, NE, rec_bytes, seg) and the per-layer experts / rg / rd must be equal too.
Prints one line per (layer, rank) and a summary; exit 0 iff every checked layer is byte-identical. Layers that differ are
the ones `nq_assemble.py --src RELDIR --into REPACK` would (re)install; equal ones it adopts without writing.
"""
import os, sys, json, hashlib, argparse
from concurrent.futures import ThreadPoolExecutor

CH = 1 << 26
HDR_KEYS = ("format", "tp", "rank", "L0", "NE", "rec_bytes", "seg")


def sha_range(p, off, n):
    h = hashlib.sha256(); done = 0
    with open(p, "rb") as f:
        f.seek(off)
        while done < n:
            b = f.read(min(CH, n - done))
            if not b:
                return None
            h.update(b); done += len(b)
    return h.hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CH), b""):
            h.update(b)
    return h.hexdigest()


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, b = (part.split("-") + [part])[:2]; out += list(range(int(a), int(b) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True); ap.add_argument("--repack", required=True)
    ap.add_argument("--layers", default=None); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); S, R = a.release, a.repack
    man = json.load(open(f"{S}/manifest.json")); tp = man["tp"]
    comp = json.load(open(f"{S}/COMPLETE")) if os.path.exists(f"{S}/COMPLETE") else None
    if comp is None:
        print(f"WARNING: {S}/COMPLETE missing (release incomplete)")
    else:
        for f, h in comp["index_sha256"].items():
            if sha_file(f"{S}/{f}") != h:
                sys.exit(f"{S}/{f} sha256 != COMPLETE: re-download the release index files")
    rel = [json.load(open(f"{S}/rank{r}.json")) for r in range(tp)]
    rep = [json.load(open(f"{R}/rank{r}.json")) for r in range(tp)]
    bad = 0
    for r in range(tp):
        for k in HDR_KEYS:
            if rep[r].get(k) != rel[r].get(k):
                print(f"rank{r}.json header {k}: repack {rep[r].get(k)} != release {rel[r].get(k)}"); bad += 1
    if bad:
        sys.exit("record layout differs: not comparable")
    jobs = []
    for r in range(tp):
        Ls = parse_layers(a.layers) if a.layers else sorted(int(L) for L in rep[r]["layers"])
        for L in Ls:
            jobs.append((r, L))

    def check(job):
        r, L = job; e = rel[r]["layers"].get(str(L)); c = rep[r]["layers"].get(str(L)); why = []
        if e is None:
            return r, L, ["not in the release"]
        if c is None:
            return r, L, ["not in the repack's rank json"]
        if comp is not None and comp["layers"].get(str(L)) != e["layer_hash"]:
            why.append("release index layer_hash != COMPLETE")
        if c.get("experts") != e["experts"]:
            why.append("experts")
        if {str(k): v for k, v in c.get("rg", {}).items()} != e["rg"] or {str(k): v for k, v in c.get("rd", {}).items()} != e["rd"]:
            why.append("rg/rd (lr ranks)")
        off = (L - rel[r]["L0"]) * rel[r]["NE"] * rel[r]["rec_bytes"]
        assert off == e["offset"], (r, L, off, e["offset"])
        bp = f"{R}/rank{r}.bin"
        h = sha_range(bp, off, e["bytes"]) if os.path.exists(bp) else None
        if h != e["sha256"]:
            why.append(f"records sha256 {str(h)[:16]} != release {e['sha256'][:16]}")
        rp = f"{R}/res/rank{r}/L{L}.pt"
        h = sha_file(rp) if os.path.exists(rp) else None
        if h != e["res_sha256"]:
            why.append(f"resident sha256 {str(h)[:16]} != release {e['res_sha256'][:16]}")
        return r, L, why

    res = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        for r, L, why in ex.map(check, jobs):
            res[(r, L)] = why
            print(f"rank{r} L{L}: {'IDENTICAL' if not why else 'DIFFERS: ' + '; '.join(why)}", flush=True)
    diff = sorted({L for (r, L), w in res.items() if w})
    same = sorted({L for (r, L), w in res.items() if not w} - set(diff))
    print(f"SUMMARY: {len(same)} layers byte-identical on all checked ranks, {len(diff)} differ: {diff}")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
