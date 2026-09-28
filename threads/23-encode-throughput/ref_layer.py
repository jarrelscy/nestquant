"""Run thread 12's nq_layer.main() (reference layer path) with the thread-23 import setup (NQ23_T12 pin, pinned CUDA
ext build dir). Same CLI as nq_layer.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
C.setup()
import nq_layer as NL
sys.argv[0] = os.path.join(C.T12, "nq_layer.py")
NL.main()
