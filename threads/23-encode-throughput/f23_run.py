"""Run a thread-23 driver with the frac23 candidate encoder aliased in (nq_encode_batch -> nq_encode_batch_f23), so the
pinned nq_encode_batch.py is never edited:   python f23_run.py check_bitid.py batch ...  |  f23_run.py nq_layer_batch.py ..."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: F401  (thread-12 pin on sys.path)
import nq_encode_batch_f23
sys.modules["nq_encode_batch"] = nq_encode_batch_f23
print("f23_run: nq_encode_batch -> nq_encode_batch_f23 (frac23 default %s)" % nq_encode_batch_f23.DEFAULT_OPTS.get("frac23"), flush=True)
import atexit


@atexit.register
def _usage():
    fr = sys.modules.get("frac23")
    print("f23_run: frac23 calls %d rings %d" % tuple(fr.CALLS) if fr else "f23_run: frac23 NOT USED", flush=True)


script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(HERE, script) if not os.path.isabs(script) else script, run_name="__main__")
