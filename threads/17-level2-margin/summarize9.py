import json, math
d = json.load(open("results/confirm9.json"))
cands2 = ["nat@2","sign@2","sg4@2","sign+G+b0.3@2","sg4+G+b0.3@2","sign+G+seqW@2"]
cands4 = ["nat@4","sign+G+b0.3@4","sg4+G+b0.3@4"]
for ref, cands in (("exl3_2", cands2), ("exl3_4", cands4)):
    print(f"\nvs {ref}: rel % (routed/forced/ood), per expert then geomean, worst")
    print("expert  " + "  ".join(f"{c:>22s}" for c in cands))
    agg = {c: {k: [] for k in ("routed","forced","ood")} for c in cands}
    for ex, r in d.items():
        row = []
        for c in cands:
            v = [100*(r[c][k]/r[ref][k]-1) for k in ("routed","forced","ood")]
            for k, x in zip(("routed","forced","ood"), v): agg[c][k].append(x)
            row.append("/".join(f"{x:+.1f}" for x in v))
        print(f"{ex:8s}" + "  ".join(f"{s:>22s}" for s in row))
    print("geomean " + "  ".join(f"{'/'.join(f'{100*(math.exp(sum(math.log(1+x/100) for x in agg[c][k])/len(agg[c][k]))-1):+.2f}' for k in ('routed','forced','ood')):>22s}" for c in cands))
    print("worst   " + "  ".join(f"{'/'.join(f'{max(agg[c][k]):+.1f}' for k in ('routed','forced','ood')):>22s}" for c in cands))
    print("n=", len(d))
