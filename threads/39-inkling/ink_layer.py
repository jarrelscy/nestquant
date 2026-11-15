"""Inkling encoder: T12 nq_layer main() (unchanged CLI) under nq_ink_env (D 4608, F 2304, 384 experts, TP2 shards,
nq15 base K = NQ35_BASE_K).  Pass --experts explicitly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_ink_env as E           # noqa: E402

if __name__ == "__main__":
    print(f"ink_layer base_K={E.NE.BASE_K} D={E.nq19.D} F={E.nq19.F} NEXP={E.nq19.NEXP} NSH={E.NL.NSH}", flush=True)
    sys.argv[0] = "nq_layer.py"
    E.NL.main()
