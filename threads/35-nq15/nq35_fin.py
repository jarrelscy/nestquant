"""T35 campaign finalize: T25 nq25_finalize.py (nq_layer finalize via --fin-cmd, safetensors convert + round trip,
artifact-from-safetensors == E{E}.pt raw bytes, L2/L4 decode compare) with the nq15 base-K decoder installed in this
process (the st check decodes in-process; the artifacts carry meta base_K, so no env is needed here)."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T25 = "/home/coder/git/nestquant/threads/25-campaign"
for p in (T12, T25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402,F401
if __name__ == "__main__":
    sys.argv[0] = f"{T25}/nq25_finalize.py"
    runpy.run_path(f"{T25}/nq25_finalize.py", run_name="__main__")
