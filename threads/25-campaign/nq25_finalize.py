"""Thread 25 finalize job for one layer (the driver's "fin" worker):

  1. [refcheck, batched-encoder layers] re-encode expert E with the reference nq_layer path into ROOT/_ref and compare
     it with the campaign's ROOT/L{L}/experts/E{E}.pt (T23's common.compare: every tensor dtype/shape/raw bytes + every
     meta value; ignored: meta.info.*.time, meta.encoder, and nq_layer-only meta keys whose value is empty/zero/None).
     Mismatch -> print "REFCHECK MISMATCH" and exit 3 before finalizing (the driver re-encodes the layer with nq_layer).
  2. nq_layer finalize (--experts 0:0 [--check-decode]) -> tp{s}.pt + manifest.json
  3. nq25_st.convert: tp{s}.safetensors (round trip == tp{s}.pt), manifest files[] -> safetensors, pt_files[] kept
  4. nq25_st.check_decode: artifact assembled from .safetensors == E{E}.pt (every expert, raw bytes) + decode L2/L4
     equal on --n-decode seeded experts (-1 = all)

  python nq25_finalize.py --layer L --out ROOT [--ref-expert E --ref-cmd JSON] --fin-cmd JSON [--no-st-check]
"""
import os, sys, json, argparse, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
T23 = "/home/coder/git/nestquant/threads/23-encode-throughput"
NQ_ONLY_META = ("ocol", "lr", "lr_rank", "ocol_idx")


def trivial(v):
    if isinstance(v, dict):
        return all(trivial(x) for x in v.values())
    return v in (None, 0, [], (), "", False)


def normalize(ref, bat):
    for k in NQ_ONLY_META:
        for x, y in ((ref, bat), (bat, ref)):
            if k in x["meta"] and k not in y["meta"] and trivial(x["meta"][k]):
                del x["meta"][k]
    for x in (ref, bat):
        x["meta"].pop("encoder", None)


def refcheck(root, L, E, cmd):
    import torch
    sys.path.insert(0, T23)
    import common as C
    t0 = time.time()
    rp = f"{root}/_ref/L{L}/experts/E{E}.pt"
    if not os.path.exists(rp):
        rc = subprocess.call(cmd)
        if rc != 0 or not os.path.exists(rp):
            print(f"[L{L}] REFCHECK ERROR: reference encode rc {rc}", flush=True)
            sys.exit(4)
    ref = torch.load(rp, weights_only=False, map_location="cpu")
    bat = torch.load(f"{root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
    enc = bat["meta"].get("encoder", "nq_layer")
    normalize(ref, bat)
    nt, nb, bad = C.compare(ref, bat)
    res = dict(layer=L, expert=E, encoder=enc, n_tensors=nt, tensor_bytes=nb, mismatches=bad[:50], n_mismatch=len(bad),
               match=not bad, s=round(time.time() - t0), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(f"{root}/refcheck", exist_ok=True)
    json.dump(res, open(f"{root}/refcheck/L{L}.json", "w"), indent=1)
    if bad:
        print(f"[L{L}] REFCHECK MISMATCH E{E} ({enc} vs nq_layer): {len(bad)} paths {bad[:5]}", flush=True)
        sys.exit(3)
    print(f"[L{L}] refcheck E{E} {enc} == nq_layer: MATCH ({nt} tensors, {nb} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ref-expert", type=int); ap.add_argument("--ref-cmd")
    ap.add_argument("--fin-cmd", required=True); ap.add_argument("--no-st-check", action="store_true")
    ap.add_argument("--n-decode", type=int, default=8, help="-1 = decode-compare every expert")
    a = ap.parse_args()
    L = a.layer
    if a.ref_expert is not None:
        refcheck(a.out, L, a.ref_expert, json.loads(a.ref_cmd))
    rc = subprocess.call(json.loads(a.fin_cmd))
    if rc != 0:
        print(f"[L{L}] nq_layer finalize rc {rc}", flush=True); sys.exit(5)
    sys.path.insert(0, HERE)
    import nq25_st as S
    t0 = time.time()
    if not S.convert(a.out, L):
        sys.exit(6)
    if not a.no_st_check and not S.check_decode(a.out, L, None if a.n_decode < 0 else a.n_decode):
        sys.exit(7)
    print(f"[L{L}] st done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
