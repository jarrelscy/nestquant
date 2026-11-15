"""Inkling finalize: T25 nq25_finalize main() under nq_ink_env."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_ink_env                # noqa: E402,F401
import nq25_finalize             # noqa: E402

if __name__ == "__main__":
    nq25_finalize.main()
