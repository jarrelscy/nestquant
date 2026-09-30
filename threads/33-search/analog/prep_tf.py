"""sm120tf blocks into our own dir (T32 sm120.prep_tf with output paths redirected; inputs read-only)."""
import os as _o
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _o.environ[_v] = "1"
import os
import alib as A
S1 = A.S1
S1.BLK = A.BLK
S1.PD = f"{A.OUT}/sm120pd"
S1.T.OUT = f"{A.OUT}/sm120pd"
os.makedirs(S1.PD, exist_ok=True)
S1.prep_tf()
