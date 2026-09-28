"""Thread 14 on thread 12's frozen level-2 stack (sign + two-sided G beta 0.5 + blend 0.3, inner 2, ref15 fold):
re-fit ONLY the level-4 residual with per-projection pattern-rate K (kernel/ref15 convention masks), bit-exact
through nq_decode pack/stream_states and ref15_spec.decode_unit.

python t12pat.py 16:36,16:92 [--plus]      (--plus: also the ~4.08 bpw g/u 1.9375 / down 2.3125 point + matched EXL3)
Results: results_t12pat/L{L}_E{E}.json. Artifacts: /tmp/nestquant/14-level4-floor/t12pat.
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T15 = "/home/coder/git/nestquant/threads/15-level4-decode"
os.environ.setdefault("NQ_RES", f"{HERE}/results_t12pat")
sys.path.insert(0, T12); sys.path.insert(0, T15); sys.path.insert(0, HERE)
import torch
import nq_run as R
import nq_decode as D
import nq_encode as NE
import harness as h
import ref15_spec as R15
import patvit

# ---------------- pattern table (kernel ring-position convention: w_p = KA + ((MASK >> (p % 16)) & 1))
NEW = {1.875: (1, 0xFEFE), 1.9375: (1, 0xFFFE), 2.3125: (2, 0x9248)}      # 2.25 = (2, 0x8888) already in both
for K, v in NEW.items():
    D.PATTERNS[K] = v; R15.PATTERNS[K] = v
Qm = NE._Qm()
for K, d in {1.875: 1.02225, 1.9375: 1.0201, 2.25: 1.0135, 2.3125: 1.0124}.items():
    Qm.LDLQ_DRIFT[K] = d
NE.RES_KS = (1.5, 2, 2.5, 3, 1.875, 1.9375, 2.25, 2.3125)


def vsteps(K):
    """EXL3/Viterbi-order step widths. Viterbi step i <-> kernel position p = (-i) mod 256 and
    state(p) = (state(p+1) << w_p | sym_p) & 0xFFFF, so the step INTO i shifts in w_{(-i) mod 256} bits."""
    KA, MASK = D.PATTERNS[K]
    return [KA + ((MASK >> ((-i) % 16)) & 1) for i in range(256)]


def patq(tiles, K):
    K = float(K)
    q, idx = [], []
    St = vsteps(K)
    for a in range(0, tiles.shape[0], 128):
        v, i = patvit._run(tiles[a:a + 128].float(), St)
        q.append(v); idx.append(i)
    return torch.cat(q), torch.cat(idx)


def is_pat(K):
    return bool((2 * float(K)) % 1)


_vit = NE.viterbi
def viterbi(rings, K):
    if is_pat(K):
        return patq(rings, K)[1].long() & 0xFFFF
    return _vit(rings, K)
NE.viterbi = viterbi

_gss = h._g_scale_search
def gss(samples, K, quantizer):
    return _gss(samples, K, patq if is_pat(K) else quantizer)
h._g_scale_search = gss


def selftest():
    """Viterbi-order states -> kernel order -> packed ref15 stream -> re-derived states: must be identical."""
    torch.manual_seed(0)
    Rg = NE.Ring("cuda")
    x = torch.randn(64, 128, 16, device="cuda")
    for K in (1.875, 1.9375, 2.25, 2.3125):
        st = viterbi(Rg.to_rings(x), K)
        _, mm = NE.roundtrip(Rg.kstates(st), K)
        wbits = D.widths(K, "cpu")[2]
        print(f"selftest K {K} {D.PATTERNS[K]} ring bits {wbits} (= {wbits // 16} u16) roundtrip mismatches {mm}",
              flush=True)
        assert mm == 0


def run(L, E, plus):
    R.SCR = "/tmp/nestquant/12-reference-encoder" if os.path.exists(
        f"/tmp/nestquant/12-reference-encoder/H_l{L}_e{E}.pt") else "/tmp/nestquant/14-level4-floor/t12H"
    os.makedirs(R.SCR, exist_ok=True)                  # (only written when T12 has no cached H for this expert)
    R.ART = "/tmp/nestquant/14-level4-floor/t12pat"
    book = R.Book(f"{R.RES}/L{L}_E{E}.json")
    t0 = time.time()
    log = lambda m: print(f"[{L}:{E}] {m} {time.time()-t0:.0f}s", flush=True)
    data = h.load_expert(L, E)
    HG = R.glm_H(data, L, E)
    Ws = data.teacher
    ev = book.R.setdefault("eval", {})
    methods, extra = {}, {}
    if "EXL3-4" not in ev:
        for K in (2, 4):
            q = []
            for pi, pn in enumerate(R.PROJ):
                Wq, _ = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, count=1, sigma_reg=R.SIG[pn])
                q.append(Wq.cpu()); h.free_scratch()
            methods[f"EXL3-{K}"] = q; extra[f"EXL3-{K}"] = dict(bpw=K + 16 * (6144 + 2048) / (6144 * 2048))
    Ps = {pn: NE.prep(Ws[pi], HG["H"][pi], 1, R.SIG[pn], G=HG["G"][pi]) for pi, pn in enumerate(R.PROJ)}
    log(f"prep gsr {[{k: round(v, 3) for k, v in Ps[p]['gsr'].items()} for p in R.PROJ]}")
    fit = R.Fit(Ps, R.BASE_VAR, tag=f"L{L}_E{E}_")
    dense, info = fit.run("nq")
    ref_bits = R.bits_expert(info, 4)
    for Lv in (2, 4):
        methods[f"nq/L{Lv}"] = dense[Lv]; extra[f"nq/L{Lv}"] = R.summary_info(info, Lv)
    log(f"ref fit L4 {ref_bits:.4f} L2 {R.bits_expert(info, 2):.4f}")
    variants = [("pat", {"gate": 1.875, "up": 1.875, "down": 2.25})]
    if plus:
        variants.append(("patplus", {"gate": 1.9375, "up": 1.9375, "down": 2.3125}))
    bits = {}
    for nm, Ks in variants:
        rules = {p: dict(kind="uniform", K=Ks[p]) for p in R.PROJ}
        dense, info = fit.run(nm, rules=rules)
        bits[nm] = R.bits_expert(info, 4)
        methods[f"{nm}/L4"] = dense[4]
        extra[f"{nm}/L4"] = dict(R.summary_info(info, 4), Ks=Ks, L2_bpw=R.bits_expert(info, 2),
                                 L2_equals_base=all(info[p]["L2_equal"] and info[p]["base_identical"] for p in R.PROJ))
        log(f"{nm} L4 {bits[nm]:.4f} base+L2 identical {[info[p]['base_identical'] and info[p]['L2_equal'] for p in R.PROJ]} "
            f"bitexact {[info[p]['bitexact'] for p in R.PROJ]} ref15 {[info[p]['ref15_mismatch'] for p in R.PROJ]} "
            f"stream {[info[p]['stream_mismatch'] for p in R.PROJ]}")
        h.free_scratch()
    R.evaluate_into(book, data, methods, extra); methods, extra = {}, {}
    for T in sorted({round(ref_bits, 4)} | ({round(bits["patplus"], 4)} if plus else set())):
        nm = f"EXL3-4+{T}"
        if nm not in ev:
            q, bpw = R.exl3_matched(Ws, HG, T)
            methods[nm] = q; extra[nm] = dict(bpw=bpw, rule="K5 on lowest-index 16-blocks")
    R.evaluate_into(book, data, methods, extra)
    book.R["time_s"] = time.time() - t0
    book.save()
    for p in Ps:
        NE.free(Ps[p])
    del fit; h.free_scratch(); torch.cuda.empty_cache()
    tab = {k: (round(v["all/routed"], 3), round(v["all/forced"], 3), round(v["ood/forced"], 3)) for k, v in book.R["eval"].items()}
    log(f"DONE {tab}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts")
    ap.add_argument("--plus", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    selftest()
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        run(L, E, a.plus)


if __name__ == "__main__":
    main()
