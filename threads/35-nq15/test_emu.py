"""T35: CPU check of AdaptEmu on real experts: W_nq2 vs FP8 ref rel-MSE, W_nq4 vs ref, s=1 exactness, s scaling."""
import sys, torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
import nq_io, emu_adapt
fp8 = nq_io.FP8Model("/tmp/nestquant/src/glm53-fp8")
PD = "/tmp/nestquant/src/predec/farm"
q = emu_adapt.AdaptEmu(s="1.5", lo=f"{PD}/nq2", hi=f"{PD}/nq4", manifest="/tmp/nestquant/32-gbdt-sal/k0_manifest.json",
                       chain="map", predictor="gbdt", gmode="sync", n_float="54", hm="0.7")
q.begin_layer(int(sys.argv[1]) if len(sys.argv) > 1 else 40, "cpu")
L = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for e in (0, 77, 200):
    cache = {}
    ref = lambda: cache.setdefault("w", fp8.expert(L, e, "cpu"))
    W4 = q.expert_level(L, e, ref, 4); Wb = q.expert_level(L, e, ref, 2)
    W2 = q.lo.expert(L, e, ref); R = ref()
    for p in R:
        r = R[p].float(); n = r.square().sum()
        rel = lambda w: float((w.float() - r).square().sum() / n)
        print(f"L{L} E{e} {p:9s} {tuple(R[p].shape)} {R[p].dtype}/{W2[p].dtype}  nq2 {rel(W2[p]):.4f}  nq4 {rel(W4[p]):.5f}"
              f"  emu(s=1.5) {rel(Wb[p]):.4f} (x{rel(Wb[p]) / rel(W2[p]):.3f}, expect 2.25)")
    q.s = 1.0
    W1 = q.expert_level(L, e, ref, 2)
    print("  s=1 bit-exact:", all(torch.equal(W1[p], W2[p]) for p in R), W2["gate_proj"].dtype)
    q.s = 1.5
    q.end_layer(L)
