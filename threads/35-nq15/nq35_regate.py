"""T35 box-move regate: a fresh nq35_layer.py re-encode of expert E must equal an existing campaign artifact bit for bit
(every tensor dtype/shape/raw bytes + every meta value, T25 same_tree), ignoring only wall-clock/host meta keys.

  python nq35_regate.py --new R/L10/experts/E7.pt --ref /tmp/nestquant/35-nq15/enc_b175/L10/experts/E7.pt [--json out]
Exit 3 on mismatch."""
import os, sys, json, argparse, re
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/home/coder/git/nestquant/threads/12-reference-encoder", "/home/coder/git/nestquant/threads/25-campaign",
          "/home/coder/git/nestquant/threads/23-encode-throughput", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402,F401
import torch                     # noqa: E402
import nq25_st as ST             # noqa: E402

IGN = re.compile(r"\.meta(\.[^. ]+)*\.(time|encoder|host|seconds|elapsed)( value| len)")      # wall-clock / host only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--json")
    a = ap.parse_args()
    new = torch.load(a.new, weights_only=False, map_location="cpu")
    ref = torch.load(a.ref, weights_only=False, map_location="cpu")
    bad = ST.same_tree(ref, new, ordered=False)
    ign = [x for x in bad if IGN.fullmatch(x)]
    bad = [x for x in bad if x not in ign]
    nt = nb = 0

    def walk(x):
        nonlocal nt, nb
        if isinstance(x, dict):
            [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)):
            [walk(v) for v in x]
        elif torch.is_tensor(x):
            nt += 1; nb += x.numel() * x.element_size()
    walk(ref)
    res = dict(new=a.new, ref=a.ref, n_tensors=nt, tensor_bytes=nb, n_mismatch=len(bad), mismatches=bad[:50],
               ignored=ign, match=not bad)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print(f"regate {os.path.basename(a.new)}: {'MATCH' if not bad else 'MISMATCH'} ({nt} tensors, {nb} bytes) "
          f"ignored={ign} {bad[:8]}", flush=True)
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
