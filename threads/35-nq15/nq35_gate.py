"""T35 first-hold gate: the campaign encoder path (nq35_layer.py, nq15 installed) at NQ35_BASE_K=2 + production res_K
must reproduce a shipped v1 expert bit for bit.

  python nq35_gate.py --layer 10 --expert E --gate-root R [--v1 /tmp/nestquant/nq-encode-v1]

Reads R/L{L}/experts/E{E}.pt (written by nq35_layer.py --no-finalize at base 2) and the v1 artifact assembled from
v1/L{L}/tp*.safetensors (nq25_st.assemble), and compares every tensor (dtype/shape/raw bytes) and meta value with
T23 common.compare (meta time/encoder ignored, as the T25 refcheck).  Writes R/gate_L{L}_E{E}.json; exit 3 on mismatch."""
import os, sys, json, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T23 = "/home/coder/git/nestquant/threads/23-encode-throughput"
T25 = "/home/coder/git/nestquant/threads/25-campaign"
for p in (T12, T25, T23, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402,F401
import torch                     # noqa: E402
import nq25_st as ST             # noqa: E402
import common as C               # noqa: E402
from nq25_finalize import normalize   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--gate-root", required=True); ap.add_argument("--v1", default="/tmp/nestquant/nq-encode-v1")
    a = ap.parse_args()
    L, E = a.layer, a.expert
    ref = ST.assemble(a.v1, L, E)
    new = torch.load(f"{a.gate_root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
    normalize(ref, new)
    nt, nb, bad = C.compare(ref, new)
    res = dict(layer=L, expert=E, v1=a.v1, n_tensors=nt, tensor_bytes=nb, n_mismatch=len(bad), mismatches=bad[:50],
               match=not bad, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(res, open(f"{a.gate_root}/gate_L{L}_E{E}.json", "w"), indent=1)
    print(f"gate L{L} E{E}: {'MATCH' if not bad else 'MISMATCH'} ({nt} tensors, {nb} bytes) {bad[:8]}", flush=True)
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
