"""Assemble the published per-layer record blocks of a NestQuant serving release into the single record file per TP rank
that the SM120 streaming server opens (NQ_REPACK dir). Pure byte copy, no decode; stdlib only.

  python nq_assemble.py DIR [--ranks 0,1,2,3] [--move] [--no-verify] [--allow-incomplete]
DIR = a downloaded serving/tp{N}/ (rank{r}.json, rank{r}/L{L}.bin, res/rank{r}/L{L}.pt, COMPLETE).
For every rank r: DIR/rank{r}.bin gets each block rank{r}/L{L}.bin pwritten at rank{r}.json layers[L].offset
(= (L-L0)*NE*rec_bytes); each block's sha256 is checked while copying (--no-verify skips it). A layer whose bytes are
already in place (recorded in DIR/rank{r}.assembled.json with the same sha256) is skipped, so after a refit only the
changed layers are rewritten. --move deletes each block after it is copied and verified (halves the disk need).
Afterwards DIR is an NQ_REPACK dir: rank{r}.bin + rank{r}.json + res/rank{r}/L{L}.pt.
"""
import os, sys, json, hashlib, argparse

CH = 1 << 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--ranks", default=None); ap.add_argument("--move", action="store_true")
    ap.add_argument("--no-verify", action="store_true"); ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args(); D = a.dir
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


if __name__ == "__main__":
    main()
