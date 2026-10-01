"""T35 campaign encoder: T12 nq_layer.py (unchanged CLI, encode + finalize + --check-decode) with the nq15 pattern-rate
base installed.  Base K from env NQ35_BASE_K (default 1.75); pass the matched residual with --res-k (nq15.CONFIGS).

  NQ35_BASE_K=1.75 python nq35_layer.py --layer L --experts a:b --no-finalize --stats ROOT --stats-mm VROOT --mm-w 0.25 \
      --out OUT --source SRC --fixed-set F --res-k 2.25,2.25,2.5625

NQ35_BASE_K=2 + no --res-k is byte-identical to production nq_layer (the first-hold gate re-encodes a shipped v1
expert this way and compares raw bytes).
T29 layers (nq35_t29.LAYERS = L3-L6, as shipped v1): the down projection is encoded with the T29 Had512 input rotation +
flat ics (nq35_t29.install_encoder) and the finalize run (--experts 0:0) adds fin29's manifest fields."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
for p in (T12, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402  installs the base-K extension over nq_encode / nq_decode
import nq_encode as NE           # noqa: E402

import nq35_t29 as T29           # noqa: E402

NE.BASE_K = float(os.environ.get("NQ35_BASE_K", "1.75"))
if __name__ == "__main__":
    argv = list(sys.argv)
    L = int(argv[argv.index("--layer") + 1])
    t29 = T29.is_t29(L) and not os.environ.get("NQ35_NO_T29")        # NQ35_NO_T29=1: plain Had128 everywhere
    if t29:
        T29.install_encoder(L)
    print(f"nq35_layer L{L} base_K={NE.BASE_K} t29={t29}", flush=True)
    sys.argv[0] = f"{T12}/nq_layer.py"
    runpy.run_path(f"{T12}/nq_layer.py", run_name="__main__")
    if t29 and "--no-finalize" not in argv:
        out = argv[argv.index("--out") + 1]
        T29.patch_manifest(out, L)
        print(f"nq35_layer L{L}: T29 manifest fields written", flush=True)
