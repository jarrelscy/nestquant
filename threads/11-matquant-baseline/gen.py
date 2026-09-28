import json, sys
S = "/tmp/nestquant/11-matquant-baseline/cfg"
import os; os.makedirs(S, exist_ok=True)
ratios = [("n2", 1, 1e-4), ("r0.1", 1, 0.1), ("r0.3", 1, 0.3), ("r1", 1, 1), ("r3", 1, 3), ("r10", 1, 10), ("r30", 1, 30), ("n4", 1e-4, 1)]
batches = []
for had in [False, True]:
    for g in [128, 64]:
        batches.append([dict(name=f"centre_dual_g{g}_h{int(had)}_{t}", l2=a, l4=b, group=g, had=had) for t, a, b in ratios])
for had in [False, True]:
    batches.append([dict(name=f"centre_shared_g128_h{int(had)}_{t}", l2=a, l4=b, group=128, had=had, feedback="shared") for t, a, b in ratios if t not in ("n2","n4")])
    batches.append([dict(name=f"msb_dual_g128_h{int(had)}_{t}", l2=a, l4=b, group=128, had=had, kind="msb") for t, a, b in ratios[1:-1:2]])
    batches.append([dict(name=f"nu_dual_g128_h{int(had)}_{t}", l2=a, l4=b, group=128, had=had, kind="nu") for t, a, b in ratios])
    batches.append([dict(name=f"sep_dual_g128_h{int(had)}_{t}", l2=a, l4=b, group=128, had=had, kind="sep") for t, a, b in ratios[1:-1]])
for i, b in enumerate(batches):
    open(f"{S}/b{i:02d}.json", "w").write(json.dumps(b))
print(len(batches))
