# nq-prefill: imported at startup by every Python process of the NQ serve container (PYTHONPATH=/nq/sm120/serve).
# Unless NQ_PREFILL_BORROW=0 (default 1) installs the EngineCore-side prefill-borrow patches (nq_pb_engine.py)
# as post-import hooks. Any failure leaves the process untouched.
import os
if os.environ.get('NQ_PREFILL_BORROW','1')=='1':
    try:
        import nq_pb_engine as _nq_pb_engine;_nq_pb_engine.install()
    except Exception as _e:
        import sys;print(f'nq_pb sitecustomize: {_e!r}',file=sys.stderr)
if os.environ.get('NQ_DBG_FUSE_M') or os.environ.get('NQ_DBG_LIN_ROWSPLIT') or os.environ.get('NQ_DBG_MLA_BMM_ROWSPLIT') or os.environ.get('NQ_DBG_NO_BF16_RED') or os.environ.get('NQ_DBG_LMHEAD_FP32') or os.environ.get('NQ_DBG_FP8_OPROJ_ONLY') or os.environ.get('NQ_DBG_FP8_TARGETS'):     # nq-kld debug numerics switches (nq_dbg_numerics.py)
    try:
        import nq_dbg_numerics as _nq_dbg;_nq_dbg.install()
    except Exception as _e:
        import sys;print(f'nq_dbg sitecustomize: {_e!r}',file=sys.stderr)
