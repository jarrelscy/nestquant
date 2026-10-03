# nq-prefill: imported at startup by every Python process of the NQ serve container (PYTHONPATH=/nq/sm120/serve).
# Unless NQ_PREFILL_BORROW=0 (default 1) installs the EngineCore-side prefill-borrow patches (nq_pb_engine.py)
# as post-import hooks. Any failure leaves the process untouched.
import os
if os.environ.get('NQ_PREFILL_BORROW','1')=='1':
    try:
        import nq_pb_engine as _nq_pb_engine;_nq_pb_engine.install()
    except Exception as _e:
        import sys;print(f'nq_pb sitecustomize: {_e!r}',file=sys.stderr)
