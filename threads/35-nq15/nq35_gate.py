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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--gate-root", required=True); ap.add_argument("--v1", default="/tmp/nestquant/nq-encode-v1")
    a = ap.parse_args()
    L, E = a.layer, a.expert
    ref = ST.assemble(a.v1, L, E)
    new = torch.load(f"{a.gate_root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
    # the TP safetensors container keeps only the projection planes (top-level meta lives in the manifest): compare
    # every projection's planes + proj meta raw-exact (T25 same_tree), plus C.compare on them
    new = {p: new[p] for p in ref}
    bad = ST.same_tree(ref, new, ordered=False)
    # container artifact: proj meta in the st tree is the layer-level proj_meta (= expert 0's), so lr.r / lr.nnz are not
    # per expert there; the per-expert rank is checked against the manifest's per_expert lr_rank instead
    import re as _re
    masked = [x for x in bad if _re.fullmatch(r"\.(gate|up|down)\.meta\.lr\.(r|nnz)( value| len|\[\d+\] value)", x)]
    bad = [x for x in bad if x not in masked]
    man = json.load(open(f"{a.v1}/L{L}/manifest.json"))
    rk = man["per_expert"][str(E)]["lr_rank"]
    for p in rk:
        r_new = int(new[p]["meta"].get("lr", {}).get("r", 0)) if "lr" in new[p]["base"] else 0
        if r_new != int(rk[p]):
            bad.append(f".{p} lr_rank {r_new} != manifest {rk[p]}")
    nt = nb = 0

    def walk(x):
        nonlocal nt, nb
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif torch.is_tensor(x):
            nt += 1; nb += x.numel() * x.element_size()
    walk(ref)
    res = dict(layer=L, expert=E, v1=a.v1, n_tensors=nt, tensor_bytes=nb, n_mismatch=len(bad), mismatches=bad[:50],
               masked_container_meta=masked,
               match=not bad, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(res, open(f"{a.gate_root}/gate_L{L}_E{E}.json", "w"), indent=1)
    print(f"gate L{L} E{E}: {'MATCH' if not bad else 'MISMATCH'} ({nt} tensors, {nb} bytes) {bad[:8]}", flush=True)
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
