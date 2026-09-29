"""Assemble the published per-layer record blocks of a NestQuant serving release into the single record file per TP rank
that the SM120 streaming server opens (NQ_REPACK dir). Pure byte copy, no decode; stdlib only.

Mode 1, in place (DIR becomes the NQ_REPACK dir):
  python nq_assemble.py DIR [--ranks 0,1,2,3] [--move] [--no-verify] [--allow-incomplete]
DIR = a downloaded serving/tp{N}/ (rank{r}.json, rank{r}/L{L}.bin, res/rank{r}/L{L}.pt, COMPLETE).
For every rank r: DIR/rank{r}.bin gets each block rank{r}/L{L}.bin pwritten at rank{r}.json layers[L].offset
(= (L-L0)*NE*rec_bytes); each block's sha256 is checked while copying (--no-verify skips it). A layer whose bytes are
already in place (recorded in DIR/rank{r}.assembled.json with the same sha256) is skipped, so after a refit only the
changed layers are rewritten. --move deletes each block after it is copied and verified (halves the disk need).
Afterwards DIR is an NQ_REPACK dir: rank{r}.bin + rank{r}.json + res/rank{r}/L{L}.pt (+ artifact_stamp.json).

Mode 2, into an existing record dir (e.g. the serve's NQ_REPACK_DIR, built by streaming/repack.py or by this script):
  python nq_assemble.py --src SRC --into RECDIR [--layers 3-6] [--dry-run] [--verify-existing] [--force] [--no-readback]
SRC = a downloaded serving/tp{N}/: rank{r}.json + manifest.json + COMPLETE + artifact_stamp.json, plus the blocks
(rank{r}/L{L}.bin, res/rank{r}/L{L}.pt, layers/L{L}.json) of the layers to install (a refit download has only those).
Candidates = --layers (default: every layer of the release). Per (layer, rank), against RECDIR/rank{r}.json:
  - entry with the release's layer_hash + sha256 + res_sha256 and the resident file at its size: already current, skip
    (--verify-existing re-hashes it anyway);
  - entry without a layer_hash (repack.py-built) or with another one: the region of rank{r}.bin and res/rank{r}/L{L}.pt
    are hashed; equal to the release -> adopted (index entry only, no data written), else installed;
  - no entry: installed.  --force installs every candidate.
Install of (L, r): L is first dropped from RECDIR/rank{r}.json (the serve then treats it as not NQ, never as torn), the
block is pwritten at its offset and fsynced, the written range is read back (page cache dropped) and must hash to the
release sha256, res/rank{r}/L{L}.pt is written as a temp file, hashed, read back and renamed over, and only then is
the layer's release entry (rg/rd, hashes, rotation fields) put back into rank{r}.json. Every index write is tmp+rename
under RECDIR/rank{r}.lock (the lock streaming/repack.py merges under). Other layers' bytes and entries are never
touched. At the start artifact_stamp.json is set to key "updating" (sm120/eval/run_c2.sh then refuses the dir); at the
end it is rewritten from the per-layer source manifest sha256 of the layers RECDIR holds (= run_c2.sh's key, and == the
release's artifact_stamp.json when RECDIR holds exactly the release). Interrupted? Rerun the same command: finished
layers are skipped, a layer caught mid-install has no index entry and is installed again.
All planning (hashing of existing layers, presence of every needed block in SRC) is done before the first write;
--dry-run stops after it and prints the plan.
"""
import os, sys, json, hashlib, argparse, fcntl, time

CH = 1 << 26
STAMP = "artifact_stamp.json"
ROT_KEYS = ("in_had_down", "had_sign_seed", "ics_down")
HDR_KEYS = ("format", "tp", "rank", "L0", "NE", "rec_bytes", "seg")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CH), b""):
            h.update(b)
    return h.hexdigest()


def sha256_range(fd, off, n, drop=False):
    if drop and hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    h = hashlib.sha256(); done = 0
    while done < n:
        b = os.pread(fd, min(CH, n - done), off + done)
        if not b:
            break
        h.update(b); done += len(b)
    return h.hexdigest() if done == n else None


def artifact_key(msha):
    """= sm120/eval/run_c2.sh KEY: sha256 of the lines "L{L} <sha256 of layers/L{L}/manifest.json>\\n" in sort -V order,
    first 16 hex chars (nq_release.artifact_key)"""
    return hashlib.sha256("".join(f"L{L} {msha[L]}\n" for L in sorted(msha)).encode()).hexdigest()[:16]


def wtext(p, s):
    t = f"{p}.{os.getpid()}.tmp"
    with open(t, "w") as f:
        f.write(s); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, b = (part.split("-") + [part])[:2]; out += list(range(int(a), int(b) + 1))
    return out


# --------------------------------------------------------------------------------------------------- mode 1 (in place)
def assemble_in_place(a):
    D = a.dir
    if not os.path.exists(f"{D}/COMPLETE") and not a.allow_incomplete:
        sys.exit(f"{D}/COMPLETE missing: the release is not complete (use --allow-incomplete to assemble what is there)")
    done = json.load(open(f"{D}/COMPLETE"))["layers"] if os.path.exists(f"{D}/COMPLETE") else None
    tp = json.load(open(f"{D}/manifest.json"))["tp"]
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else list(range(tp))
    for r in ranks:
        idx = json.load(open(f"{D}/rank{r}.json")); rb = idx["rec_bytes"]
        st_p = f"{D}/rank{r}.assembled.json"; st = json.load(open(st_p)) if os.path.exists(st_p) else {}
        bp = f"{D}/{idx.get('bin', f'rank{r}.bin')}"
        fd = os.open(bp, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size < idx["bin_bytes"]:
                os.ftruncate(fd, idx["bin_bytes"])
            for L, e in sorted(idx["layers"].items(), key=lambda kv: int(kv[0])):
                assert e["offset"] == (int(L) - idx["L0"]) * idx["NE"] * rb and e["bytes"] == idx["NE"] * rb, L
                if done is not None:
                    assert done.get(L) == e["layer_hash"], f"L{L}: index layer_hash != COMPLETE"
                if st.get(L) == e["sha256"]:
                    continue
                src = f"{D}/{e['file']}"
                if not os.path.exists(src):
                    sys.exit(f"rank{r} L{L}: {src} missing and not yet assembled")
                h = hashlib.sha256(); n = 0
                with open(src, "rb") as f:
                    while True:
                        b = f.read(CH)
                        if not b:
                            break
                        if not a.no_verify:
                            h.update(b)
                        os.pwrite(fd, b, e["offset"] + n); n += len(b)
                assert n == e["bytes"], (L, n, e["bytes"])
                if not a.no_verify and h.hexdigest() != e["sha256"]:
                    sys.exit(f"rank{r} L{L}: sha256 mismatch in {src}")
                os.fsync(fd)
                st[L] = e["sha256"]
                json.dump(st, open(st_p + ".tmp", "w")); os.replace(st_p + ".tmp", st_p)
                if a.move:
                    os.remove(src)
                print(f"rank{r} L{L}: {n/2**20:.0f} MiB at {e['offset']}", flush=True)
        finally:
            os.close(fd)
        print(f"rank{r}: {bp} assembled ({len(st)} layers)", flush=True)


# ------------------------------------------------------------------------------------------- mode 2 (into a record dir)
def load_release(S, allow_incomplete):
    """release index of SRC, verified along the chain COMPLETE -> rank{r}.json / manifest.json / artifact_stamp.json"""
    man = json.load(open(f"{S}/manifest.json")); tp = man["tp"]
    comp = json.load(open(f"{S}/COMPLETE")) if os.path.exists(f"{S}/COMPLETE") else None
    if comp is None and not allow_incomplete:
        sys.exit(f"{S}/COMPLETE missing: the release is incomplete (a refit upload in progress?); wait, or --allow-incomplete")
    idx = [json.load(open(f"{S}/rank{r}.json")) for r in range(tp)]
    if comp is not None:
        for f, h in comp["index_sha256"].items():
            if sha256_file(f"{S}/{f}") != h:
                sys.exit(f"{S}/{f} sha256 != COMPLETE (mixed download of two release versions? re-download the index files)")
        if sha256_file(f"{S}/manifest.json") != comp["manifest_sha256"]:
            sys.exit(f"{S}/manifest.json sha256 != COMPLETE")
        if "stamp_sha256" in comp and sha256_file(f"{S}/{STAMP}") != comp["stamp_sha256"]:
            sys.exit(f"{S}/{STAMP} sha256 != COMPLETE")
        for r in range(tp):
            for L, e in idx[r]["layers"].items():
                if comp["layers"].get(L) != e["layer_hash"]:
                    sys.exit(f"rank{r} L{L}: index layer_hash != COMPLETE")
    for r in range(tp):
        assert idx[r]["rank"] == r and idx[r]["tp"] == tp, (r, idx[r]["rank"], idx[r]["tp"])
    return man, idx


def res_path(R, r, L):
    return f"{R}/res/rank{r}/L{L}.pt"


class Index:
    """RECDIR/rank{r}.json, read-modify-written under rank{r}.lock (repack.py's lock) with tmp+rename"""
    def __init__(s, R, r, rel):
        s.p = f"{R}/rank{r}.json"; s.lk = f"{R}/rank{r}.lock"; s.rel = rel

    def read(s):
        return json.load(open(s.p)) if os.path.exists(s.p) else None

    def update(s, fn):
        with open(s.lk, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            cur = s.read()
            if cur is None:                              # fresh record dir: the release header, no layers yet
                cur = {k: v for k, v in s.rel.items() if k != "layers"}; cur["layers"] = {}
            else:                                        # repack.py header + the release's extra header keys
                for k in HDR_KEYS:
                    if k in cur and cur[k] != s.rel[k]:
                        sys.exit(f"{s.p}: {k} {cur[k]} != release {s.rel[k]} (a different record layout: not a drop-in)")
                for k, v in s.rel.items():
                    if k != "layers":
                        cur.setdefault(k, v)
            fn(cur["layers"])
            wtext(s.p, json.dumps(cur, indent=1, sort_keys=True))


def current(R, r, L, cur, e, fd, rehash):
    """-> 'current' | 'adopt' | 'install' for (L, r) given RECDIR entry cur (or None) and release entry e"""
    rp = res_path(R, r, L)
    if cur is None or fd is None or not os.path.exists(rp) or os.path.getsize(rp) != e["res_bytes"]:
        return "install"
    if os.fstat(fd).st_size < e["offset"] + e["bytes"]:
        return "install"
    same = cur.get("layer_hash") == e["layer_hash"] and cur.get("sha256") == e["sha256"] and \
        cur.get("res_sha256") == e["res_sha256"]
    if same and not rehash:
        return "current"
    if cur.get("experts") != e["experts"] or {str(k): v for k, v in cur.get("rg", {}).items()} != e["rg"] or \
            {str(k): v for k, v in cur.get("rd", {}).items()} != e["rd"]:
        return "install"
    ok = sha256_range(fd, e["offset"], e["bytes"]) == e["sha256"] and sha256_file(rp) == e["res_sha256"]
    return ("current" if same else "adopt") if ok else "install"


def copy_verify(src, fd, off, nbytes, want, readback):
    """pwrite src at off (hashing the source while copying), fsync, read the range back from disk: both must be want"""
    h = hashlib.sha256(); n = 0
    with open(src, "rb") as f:
        for b in iter(lambda: f.read(CH), b""):
            h.update(b); os.pwrite(fd, b, off + n); n += len(b)
    if n != nbytes or h.hexdigest() != want:
        return f"{src}: {n} B, sha256 {h.hexdigest()[:16]} != release {want[:16]} (bad download)"
    os.fsync(fd)
    if readback and sha256_range(fd, off, nbytes, drop=True) != want:
        return f"read-back of the written range at {off} does not hash to the release sha256"
    return None


def copy_file_verify(src, dst, want, readback):
    t = f"{dst}.nq_assemble.tmp"; os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = hashlib.sha256()
    with open(src, "rb") as f, open(t, "wb") as g:
        for b in iter(lambda: f.read(CH), b""):
            h.update(b); g.write(b)
        g.flush(); os.fsync(g.fileno())
    if h.hexdigest() != want:
        os.remove(t); return f"{src}: sha256 {h.hexdigest()[:16]} != release {want[:16]} (bad download)"
    if readback:
        fd = os.open(t, os.O_RDONLY)
        try:
            ok = sha256_range(fd, 0, os.fstat(fd).st_size, drop=True) == want
        finally:
            os.close(fd)
        if not ok:
            os.remove(t); return f"read-back of {t} does not hash to the release sha256"
    os.replace(t, dst)
    return None


def write_stamp(R, tp, rel_idx, S):
    """artifact_stamp.json from the layers every rank of RECDIR holds (per-layer source manifest sha256)"""
    idx = [json.load(open(f"{R}/rank{r}.json")) for r in range(tp)]
    Ls = sorted(set.intersection(*[set(int(L) for L in x["layers"]) for x in idx]))
    msha, unknown = {}, []
    for L in Ls:
        es = [x["layers"][str(L)] for x in idx]
        m = {e.get("source_manifest_sha256") for e in es}
        if len(m) != 1 or None in m:
            unknown.append(L)
        else:
            msha[L] = m.pop()
    if unknown or not Ls:
        s = json.dumps(dict(artifact="unknown", key="unknown", note="layers without a release entry (never adopted): "
                            f"{unknown}; run nq_assemble.py --into over them", layers_present=Ls)) + "\n"
    else:
        rs = f"{S}/{STAMP}"; rel = json.load(open(rs)) if os.path.exists(rs) else None
        if rel is not None and {int(k): v for k, v in rel["layers"].items()} == msha:
            s = open(rs).read()                          # RECDIR == the release: its stamp, byte for byte
        else:
            s = json.dumps(dict(artifact="hf://jarrelscy/GLM-5.3-NestQuant-2-4bit (layers/, subset)", key=artifact_key(msha),
                                source="nq_assemble.py --into", layers={str(L): msha[L] for L in Ls})) + "\n"
    wtext(f"{R}/{STAMP}", s)
    return json.loads(s)["key"], Ls


def into(a):
    S, R = a.src, a.into
    man, rel = load_release(S, a.allow_incomplete); tp = man["tp"]
    rb = rel[0]["rec_bytes"]; Ls_rel = sorted(int(L) for L in rel[0]["layers"])
    cand = parse_layers(a.layers) if a.layers else Ls_rel
    miss = [L for L in cand if L not in Ls_rel]
    if miss:
        sys.exit(f"layers {miss} are not in the release index")
    os.makedirs(R, exist_ok=True)
    ix = [Index(R, r, rel[r]) for r in range(tp)]
    t0 = time.time(); plan = {}; need = []
    for r in range(tp):
        cur = ix[r].read()
        if cur is not None:
            for k in HDR_KEYS:
                if k in cur and cur[k] != rel[r][k]:
                    sys.exit(f"{R}/rank{r}.json: {k} {cur[k]} != release {rel[r][k]} (a different record layout: not a drop-in)")
        bp = f"{R}/rank{r}.bin"; fd = os.open(bp, os.O_RDONLY) if os.path.exists(bp) else None
        try:
            for L in cand:
                e = rel[r]["layers"][str(L)]
                assert e["offset"] == (L - rel[r]["L0"]) * rel[r]["NE"] * rb and e["bytes"] == rel[r]["NE"] * rb, (r, L)
                c = None if cur is None else cur["layers"].get(str(L))
                act = "install" if a.force else current(R, r, L, c, e, fd, a.verify_existing)
                plan[(L, r)] = act
                if act == "install":
                    for f in (e["file"], e["res"]):
                        if not os.path.exists(f"{S}/{f}"):
                            need.append(f)
        finally:
            if fd is not None:
                os.close(fd)
    cnt = {k: sorted({L for (L, r), v in plan.items() if v == k}) for k in ("current", "adopt", "install")}
    npair = {k: sum(v == k for v in plan.values()) for k in cnt}
    print(f"plan ({time.time()-t0:.0f}s, (layer, rank) pairs): current {npair['current']}, adopt {npair['adopt']} "
          f"(layers {cnt['adopt']}), install {npair['install']} (layers {cnt['install']})", flush=True)
    if need:
        sys.exit(f"{len(need)} block files needed for the install are not in {S}, e.g. {need[:4]}: download them "
                 f"(--include 'serving/tp{tp}/rank*/L{{L}}.bin' 'serving/tp{tp}/res/rank*/L{{L}}.pt' for those layers)")
    if a.dry_run:
        return
    if cnt["adopt"] or cnt["install"]:
        wtext(f"{R}/{STAMP}", json.dumps(dict(artifact="updating", key="updating", note="nq_assemble.py --into in progress "
                                              "or interrupted: rerun it"), indent=None) + "\n")
    for r in range(tp):
        adopt = [L for L in cand if plan[(L, r)] == "adopt"]
        if adopt:
            ix[r].update(lambda lay: lay.update({str(L): rel[r]["layers"][str(L)] for L in adopt}))
            print(f"rank{r}: adopted {adopt} (bytes already equal the release)", flush=True)
        todo = [L for L in cand if plan[(L, r)] == "install"]
        if not todo:
            continue
        bp = f"{R}/rank{r}.bin"
        fd = os.open(bp, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            for L in todo:
                e = rel[r]["layers"][str(L)]; t = time.time()
                ix[r].update(lambda lay: lay.pop(str(L), None))
                err = copy_verify(f"{S}/{e['file']}", fd, e["offset"], e["bytes"], e["sha256"], not a.no_readback) or \
                    copy_file_verify(f"{S}/{e['res']}", res_path(R, r, L), e["res_sha256"], not a.no_readback)
                if err:
                    sys.exit(f"rank{r} L{L}: {err}; L{L} left out of {R}/rank{r}.json (the serve runs it without NQ); "
                             f"fix the download and rerun")
                ix[r].update(lambda lay: lay.__setitem__(str(L), e))
                print(f"rank{r} L{L}: installed {e['bytes']/2**20:.0f} MiB at {e['offset']} + resident "
                      f"{e['res_bytes']/2**20:.0f} MiB, verified ({time.time()-t:.1f}s)", flush=True)
        finally:
            os.close(fd)
    key, Ls = write_stamp(R, tp, rel, S)
    print(f"{R}: {len(Ls)} layers, artifact_stamp key {key} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?"); ap.add_argument("--ranks", default=None); ap.add_argument("--move", action="store_true")
    ap.add_argument("--no-verify", action="store_true"); ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--src"); ap.add_argument("--into"); ap.add_argument("--layers", default=None)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--verify-existing", action="store_true")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--no-readback", action="store_true")
    a = ap.parse_args()
    if a.into or a.src:
        if not (a.into and a.src) or a.dir:
            ap.error("--into needs --src (and no positional DIR)")
        if os.path.realpath(a.src) == os.path.realpath(a.into):
            ap.error("--src and --into are the same dir: use mode 1 (nq_assemble.py DIR)")
        into(a)
    elif a.dir:
        assemble_in_place(a)
    else:
        ap.error("DIR or --src/--into required")


if __name__ == "__main__":
    main()
