import sys, torch, math
sys.path.insert(0, "."); import nq_run as R, nq_encode as NE, harness as h
torch.cuda.set_per_process_memory_fraction(12/80); torch.backends.cuda.matmul.allow_tf32 = False
def stat(W, H, cnt, sig, tag):
    P = NE.prep(W, H, cnt, sig)
    Din = P["Din"]                                   # [m,128,128] conditional covs (128-block LDL)
    # 16-block LDL of the full H: D16 blocks
    _, D16 = NE.ldl_blocks(P["Hr"], 16, sig)
    # MSE-coding loss vs ideal: per chunk AM(diag of D_a)/GM(eig D_a); block16: AM(diag D16)/GM(eig D16) grouped per 128
    am128 = torch.stack([Din[i].diagonal().mean() for i in range(len(Din))])
    lam = torch.linalg.eigvalsh(Din.double()).clamp_min(1e-30)
    gm = lam.log().mean(-1).exp()
    d16 = torch.stack([D16[i].diagonal() for i in range(len(D16))]).view(len(Din), 128)
    lam16 = torch.linalg.eigvalsh(D16.double()).clamp_min(1e-30).view(len(Din), 128)
    print(tag, "128-block: mean tr(D)/128 = %.4g; dB loss AM(diag)/GM(eig) = %.2f dB; 16-block: AM(diag)=%.4g dB loss %.2f; GM ratio 128/16 = %.3f" % (
        float(am128.mean()), float(10*torch.log10(am128.double()/gm).mean()), float(d16.mean()),
        float(10*torch.log10(d16.double().mean(-1)/lam16.log().mean(-1).exp()).mean()), float((gm/lam16.log().mean(-1).exp()).mean())), flush=True)
    # sum-level: expected distortion proportional to sum over chunks of AM(diag D) (MSE coding, no intra feedback)
    print(tag, "   sum tr D128 / sum tr D16 = %.3f" % float(am128.sum() / d16.mean(-1).sum()), flush=True)
    NE.free(P)
d = h.load_expert(55, 70, source=R.MIMO_SRC, statistics=R.MIMO_STATS.format(L=55, E=70), capture=None)
st = d.stats; cnt = st["metadata"]["training_rows"]
stat(d.teacher[2].float(), st["grams"][1].float(), cnt, 0.03, "mimo down s0.03")
stat(d.teacher[0].float(), st["grams"][0].float(), cnt, 0.03, "mimo gate s0.03")
g = h.load_expert(16, 36, capture=None); HG = R.glm_H(g, 16, 36)
stat(g.teacher[2].float(), HG["H"][2], 1, 1.0, "glm down s1.0")
stat(g.teacher[0].float(), HG["H"][0], 1, 0.5, "glm gate s0.5")
