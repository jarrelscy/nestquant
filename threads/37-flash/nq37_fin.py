"""T37 finalize: T25 nq25_finalize main() (nq_layer finalize via --fin-cmd, safetensors convert + round trip,
L2/L4 decode compare) under nq37_env."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq37_env                  # noqa: E402,F401
import nq25_finalize             # noqa: E402

if __name__ == "__main__":
    nq25_finalize.main()
