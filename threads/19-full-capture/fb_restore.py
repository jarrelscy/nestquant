"""Thread 19: restore a capture root from the flashblade backup written by fb_backup19.py.

    fb_restore.py --prefix s3://annalise-shared-prod/jarrel/nestquant/19-capture-glmfmt --root /tmp/nestquant/19-capture-glmfmt
                  [--layers 3-77] [--no-global]

Downloads PREFIX/latest.json and every PREFIX/done/L{L}.json, fetches each file into ROOT at its relative key, and
verifies size + sha256 against the marker.  It then rebuilds the links (stats/L{L} -> L{L}.vN and
stats0/L{L} -> ../stats/L{L}.vN), writes eval/val and eval/matched as real directories, and puts the global files
(plan.json, protocol/progress, markers) back.  A restored root is usable directly with
nq19_load.Capture(root=ROOT, stats="stats0").  Boundary-row paths are resolved relative to ROOT.
--set stats1 restores the chunk-0 + traces snapshot, --set full the final stats; restore stats0 first if several are wanted.
Files that are already present with the right sha256 are skipped.
"""
import argparse
import json
import os
import subprocess
import tempfile

from fb_backup19 import AWS, ENDPOINT, ENV, latest_name, marker_dir, sha256


def aws(*args):
    r = subprocess.run([AWS, "--endpoint-url", ENDPOINT, *args], env=ENV, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"aws {' '.join(args[:3])}: {r.stderr[-500:]}")
    return r.stdout


def fetch_json(uri):
    with tempfile.NamedTemporaryFile(suffix=".json") as t:
        aws("s3", "cp", uri, t.name, "--only-show-errors")
        return json.load(open(t.name))


def get(rec, dst, strict=True):
    """strict=False (global files: mutable progress/markers shared by several backup sets) -> warn and keep the
    current object on a sha mismatch; layer files are always strict."""
    if os.path.exists(dst) and os.path.getsize(dst) == rec["size"] and sha256(dst) == rec["sha256"]:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    aws("s3", "cp", rec["key"], dst + ".part", "--only-show-errors")
    if os.path.getsize(dst + ".part") != rec["size"] or sha256(dst + ".part") != rec["sha256"]:
        if strict:
            raise RuntimeError(f"checksum mismatch for {rec['key']}")
        print(json.dumps(dict(warning="global file changed since the marker was written; kept current", key=rec["key"])))
    os.replace(dst + ".part", dst)
    return "ok"


def link(target, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.lexists(path):
        os.remove(path)
    os.symlink(target, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--layers", default="3-77")
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--set", default="stats0", help="stats0 / stats1 / full")
    a = ap.parse_args()
    prefix = a.prefix.rstrip("/")
    lo, hi = [int(v) for v in a.layers.split("-")]
    latest = fetch_json(f"{prefix}/{latest_name(a.set)}.json")
    if latest.get("small_only"):
        print("WARNING: this backup is small-only (no per-expert gram files); stats will be incomplete")
    if not a.no_global:
        for rec in latest["global_files"]:
            get(rec, f"{a.root}/{rec['rel'][len('global/'):]}", strict=False)
    # latest_*.json is rewritten only at the end of each backup pass -> also take every per-layer done marker present
    ls = aws("s3", "ls", f"{prefix}/{marker_dir(a.set)}/")
    marked = {int(t[1:-5]) for t in ls.split() if t.startswith("L") and t.endswith(".json")}
    for L in sorted(set(latest["layers_done"]) | marked):
        if not lo <= L <= hi:
            continue
        d = fetch_json(f"{prefix}/{marker_dir(a.set)}/L{L}.json")
        n = [get(rec, f"{a.root}/{rec['rel']}") for rec in d["files"]]
        link(d["version"], f"{a.root}/stats/L{L}")
        if a.set != "full":
            link(f"../stats/{d['version']}", f"{a.root}/{a.set}/L{L}")
        print(json.dumps(dict(layer=L, files=len(n), fetched=n.count("ok"), bytes=d["bytes"])), flush=True)


if __name__ == "__main__":
    main()
