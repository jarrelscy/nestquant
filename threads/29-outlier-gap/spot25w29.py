"""threads/25 nq25_spot.py with the decoder swapped for nq29_had.decode_expert29 (honours in_had_down in E.pt meta;
== the pinned decoder at 128).  Same args as nq25_spot.py; used to redo the README spot numbers for the L3-L6 refit."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nq29_had as NH
import nq_decode as D
_orig = D.decode_expert
D.decode_expert = lambda art, level, device="cuda": (_orig(art, level, device) if NH.width_of(art) == 128
                                                      else NH.decode_expert29(art, level, device))
sys.argv = ["nq25_spot.py"] + sys.argv[1:]
runpy.run_path("/home/coder/git/nestquant/threads/25-campaign/nq25_spot.py", run_name="__main__")
