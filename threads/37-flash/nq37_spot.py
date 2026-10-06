"""T37 spot check: T25 nq25_spot (2 experts per layer, encode's H, matched eval) under nq37_env, with the 256-expert
literals -> 288 and the L2 anchor at EXL3 K = NQ35_BASE_K (as nq35_spot)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq37_env as E37           # noqa: E402
import harness as h              # noqa: E402
import nq_patvit as PV           # noqa: E402

KB = E37.NE.BASE_K
_q = h.quantize_exl3_like


def quantize_exl3_like(W, H, K, **kw):
    if K == 2 and KB != 2:
        K = KB
        if PV.is_pat(K):
            kw.update(quantizer=PV.patq, codebook=h.codebook_lut("mul1"))
    return _q(W, H, K, **kw)


h.quantize_exl3_like = quantize_exl3_like
if __name__ == "__main__":
    src = f"{E37.NQ}/25-campaign/nq25_spot.py"
    code = open(src).read()
    n = code.count("minlength=256") + code.count("range(256)")
    assert n == 2, n
    code = code.replace("minlength=256", f"minlength={E37.NEXP}").replace("range(256)", f"range({E37.NEXP})")
    argv = list(sys.argv)
    sys.argv[0] = src
    exec(compile(code, src, "exec"), dict(__name__="__main__", __file__=src))
    L = argv[argv.index("--layer") + 1]; out = argv[argv.index("--out") + 1]
    p = f"{out}/spot/L{L}.json"
    r = json.load(open(p)); r["exl3_lo_K"] = KB; r["note"] = "EXL3-2 rows are EXL3 at K = exl3_lo_K (T37 nq37_spot)"
    json.dump(r, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
