"""T35 campaign spot check: T25 nq25_spot.py (2 experts per layer, encode's H, matched eval) with the nq15 decoder and
the L2 anchor at the campaign's base rate: rows named "EXL3-2" are EXL3 at K = NQ35_BASE_K (pattern rates through
the frac Viterbi + mul1 LUT codebook, as pilot15.exl3_anchor).  Flag rule unchanged (L2 vs EXL3-base > l2-thr %,
L4 vs EXL3-4 > l4-thr %).  The output json gets exl3_lo_K."""
import os, sys, json, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T25 = "/home/coder/git/nestquant/threads/25-campaign"
T05 = "/home/coder/git/nestquant/threads/05-exl3-harness"
for p in (T12, T25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402,F401
import harness as h              # noqa: E402
import nq_patvit as PV           # noqa: E402

KB = float(os.environ.get("NQ35_BASE_K", "1.75"))
_q = h.quantize_exl3_like


def quantize_exl3_like(W, H, K, **kw):
    if K == 2 and KB != 2:
        K = KB
        if PV.is_pat(K):
            kw.update(quantizer=PV.patq, codebook=h.codebook_lut("mul1"))
    return _q(W, H, K, **kw)


h.quantize_exl3_like = quantize_exl3_like
if __name__ == "__main__":
    argv = list(sys.argv)
    sys.argv[0] = f"{T25}/nq25_spot.py"
    runpy.run_path(f"{T25}/nq25_spot.py", run_name="__main__")
    L = argv[argv.index("--layer") + 1]; out = argv[argv.index("--out") + 1]
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else ""
    p = f"{out}/spot/L{L}{tag}.json"
    r = json.load(open(p)); r["exl3_lo_K"] = KB; r["note"] = "EXL3-2 rows are EXL3 at K = exl3_lo_K (T35 nq35_spot)"
    json.dump(r, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
