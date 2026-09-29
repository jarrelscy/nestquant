"""Thread 28 refit hook: call after any layer is (re)finalized -- T27 PV-tuning post-pass, a re-encode, a new fixed set --
so the serving release follows the nestquant-v1 layers.

  python nq_refit_hook.py ROOT --layers 17,40-42 [--out /tmp/nestquant/28-serve-release/out] [--tps 4] [--upload]

ROOT = the nestquant-v1 tree that holds the finalized layers (L{L}/tp{s}.pt|.safetensors + manifest.json, e.g.
/tmp/nestquant/nq-encode-v1 after the refit layers were finalized in place, or an HF-layout root with layers/L{L}/).
For every TP degree in --tps (default: every serving/tp{N} already built under OUT):
  1. nq_release.py ROOT OUT --tp N --layers LAYERS    re-exports only (layer, rank)s whose source manifest changed;
                                                     rewrites layers/L{L}.json, the index, manifest and local COMPLETE
  2. nq_check.py OUT/serving/tpN --ref ROOT --layers LAYERS --ranks all   (structure + hashes of everything, decode of
                                                     the refit layers on every rank); stops on FAIL
  3. with --upload: nq_upload.py OUT --tp N --go     only layers whose content hash changed are sent; the Hub's COMPLETE
                                                     is deleted first and rewritten last
Publish the refit's nestquant-v1 layers (layers/L{L}/, threads/25 nq25_upload.py) in the same session so the Hub's
safetensors and serving blocks agree (nq_check on a download compares the two).
"""
import os, sys, glob, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/tmp/nestquant/28-serve-release/out"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"step failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--layers", required=True); ap.add_argument("--out", default=OUT)
    ap.add_argument("--tps", default=None); ap.add_argument("--upload", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tps = [int(x) for x in a.tps.split(",")] if a.tps else sorted(int(p.rsplit("tp", 1)[1]) for p in glob.glob(f"{a.out}/serving/tp*"))
    if not tps:
        sys.exit(f"no serving/tp* under {a.out}; pass --tps")
    py = sys.executable
    for tp in tps:
        run([py, f"{HERE}/nq_release.py", a.root, a.out, "--tp", str(tp), "--layers", a.layers, "--jobs", str(a.jobs)])
        run([py, f"{HERE}/nq_check.py", f"{a.out}/serving/tp{tp}", "--ref", a.root, "--layers", a.layers, "--ranks", "all"])
        if a.upload:
            run([py, f"{HERE}/nq_upload.py", a.out, "--tp", str(tp), "--go"])


if __name__ == "__main__":
    main()
