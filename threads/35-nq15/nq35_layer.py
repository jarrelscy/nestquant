"""T35 campaign encoder: T12 nq_layer.py (unchanged CLI, encode + finalize + --check-decode) with the nq15 pattern-rate
base installed.  Base K from env NQ35_BASE_K (default 1.75); pass the matched residual with --res-k (nq15.CONFIGS).

  NQ35_BASE_K=1.75 python nq35_layer.py --layer L --experts a:b --no-finalize --stats ROOT --stats-mm VROOT --mm-w 0.25 \
      --out OUT --source SRC --fixed-set F --res-k 2.25,2.25,2.5625

NQ35_BASE_K=2 + no --res-k is byte-identical to production nq_layer (the first-hold gate re-encodes a shipped v1
expert this way and compares raw bytes)."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
for p in (T12, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402  installs the base-K extension over nq_encode / nq_decode
import nq_encode as NE           # noqa: E402

NE.BASE_K = float(os.environ.get("NQ35_BASE_K", "1.75"))
if __name__ == "__main__":
    sys.argv[0] = f"{T12}/nq_layer.py"
    runpy.run_path(f"{T12}/nq_layer.py", run_name="__main__")
