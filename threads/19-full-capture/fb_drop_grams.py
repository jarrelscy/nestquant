"""Thread 19: delete the per-expert gram objects (A0/A2/D0/D2/Dc.f32) of one backed-up set from flashblade, keeping
everything else, and rewrite that set's done markers / latest file as small-only (flashblade budget, user rule < 5 TB).

    fb_drop_grams.py --root /tmp/nestquant/19-capture-glmfmt --prefix s3://.../19-capture-glmfmt --set stats0 [--go]

Targets = PREFIX/stats/<version>/{grams} for the versions ROOT/<set>/L{L} links to (local /tmp copies are untouched).
Refuses unless --protect's set (default full) is complete on S3: every done_<protect> marker present for 75 layers,
not small-only (unless --protect-small-ok), and every file it lists present with the right size.  Dry run unless --go.
"""
import argparse
import json
import os
import tempfile

from fb_backup19 import GRAMS, aws, head, latest_name, marker_dir, put_json, s3_listing, split


def fetch_json(uri):
    with tempfile.NamedTemporaryFile(suffix=".json") as t:
        aws("s3", "cp", uri, t.name, "--only-show-errors")
        return json.load(open(t.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--set", default="stats0")
    ap.add_argument("--protect", default="full", help="set that must be fully backed up (with grams) first")
    ap.add_argument("--protect-small-ok", action="store_true",
                    help="accept a small-only protect set (lead's order: final small-only -> drop stats0 grams -> final grams)")
    ap.add_argument("--no-protect", action="store_true", help="skip the protect-set check (user-authorised cleanup)")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    prefix = a.prefix.rstrip("/")
    lst = s3_listing(prefix + "/")
    # 1. the protected set must be complete and verified
    n_ok = 0
    for L in ([] if a.no_protect else range(3, 78)):
        d = fetch_json(f"{prefix}/{marker_dir(a.protect)}/L{L}.json")
        if d["small_only"] and not a.protect_small_ok:
            raise SystemExit(f"{a.protect} L{L} is small-only; refusing")
        bad = [r["key"] for r in d["files"] if lst.get(r["key"]) != r["size"]]
        if bad:
            raise SystemExit(f"{a.protect} L{L}: {len(bad)} objects missing/size mismatch, e.g. {bad[0]}")
        if not d["small_only"] and not {os.path.basename(r["key"]) for r in d["files"]} >= GRAMS:
            raise SystemExit(f"{a.protect} L{L}: grams not all listed")
        n_ok += 1
    print(json.dumps(dict(protect=a.protect, layers_verified=n_ok, objects=len(lst))), flush=True)
    # 2. targets: grams of this set's versions only
    protect_versions = set()
    for L in ([] if a.no_protect else range(3, 78)):
        protect_versions.add(fetch_json(f"{prefix}/{marker_dir(a.protect)}/L{L}.json")["version"])
    targets = []
    for L in range(3, 78):
        v = os.path.basename(os.path.realpath(f"{a.root}/{a.set}/L{L}"))
        assert v not in protect_versions, (L, v)
        assert v != os.path.basename(os.path.realpath(f"{a.root}/stats/L{L}")) or a.set == "stats", (L, v)
        targets += [(L, f"{prefix}/stats/{v}/{g}") for g in sorted(GRAMS) if f"{prefix}/stats/{v}/{g}" in lst]
    tb = sum(lst[k] for _, k in targets) / 1e12
    print(json.dumps(dict(set=a.set, gram_objects=len(targets), tb=round(tb, 4), go=a.go)), flush=True)
    if not a.go:
        return
    for _, k in targets:
        b, key = split(k)
        aws("s3api", "delete-object", "--bucket", b, "--key", key)
        assert head(k) is None, k
    # 3. markers: drop the gram records, flag small-only
    tmpdir = f"{a.root}/logs"
    for L in range(3, 78):
        uri = f"{prefix}/{marker_dir(a.set)}/L{L}.json"
        if head(uri) is None:
            continue
        d = fetch_json(uri)
        d["files"] = [r for r in d["files"] if os.path.basename(r["rel"]) not in GRAMS]
        d["bytes"] = sum(r["size"] for r in d["files"])
        d["small_only"] = True
        d["grams_dropped"] = "fb_drop_grams.py (flashblade budget); local /tmp copy kept"
        put_json(d, uri, tmpdir)
    lat = f"{prefix}/{latest_name(a.set)}.json"
    d = fetch_json(lat)
    d["small_only"] = True
    d["grams_dropped"] = True
    put_json(d, lat, tmpdir)
    after = sum(s3_listing(prefix + "/").values()) / 1e12
    print(json.dumps(dict(deleted=len(targets), prefix_tb_after=round(after, 4))), flush=True)


if __name__ == "__main__":
    main()
