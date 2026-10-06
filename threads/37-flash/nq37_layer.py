"""T37 encoder: T12 nq_layer main() (unchanged CLI) under nq37_env (Flash dims/keys, nq15 base K = NQ35_BASE_K,
fixed rule n=19).  Pass --experts explicitly (T12 default 0:256)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq37_env as E37           # noqa: E402

if __name__ == "__main__":
    print(f"nq37_layer base_K={E37.NE.BASE_K} D={E37.nq19.D} NEXP={E37.nq19.NEXP}", flush=True)
    sys.argv[0] = "nq_layer.py"
    E37.NL.main()
