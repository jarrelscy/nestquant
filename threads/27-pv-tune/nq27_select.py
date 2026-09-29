"""T27 pilot expert selection (deterministic): flagged spot experts + 8 seeded random experts per layer with
>= 256 routed val rows; control layer L30: 8 random.  Writes experts.json."""
import json
import numpy as np
import torch

VAL = "/tmp/nestquant/19-capture-glmfmt/eval/val/layer_{L}.pt"
FLAG = {3: [148, 199], 4: [168, 199], 5: [81, 134], 6: [87, 99], 30: []}
out = {}
for L, fl in FLAG.items():
    ids = torch.load(VAL.format(L=L), weights_only=True, mmap=True)["ids"]
    rows = np.bincount(ids.flatten().numpy(), minlength=256)
    pool = [e for e in range(256) if e not in fl and rows[e] >= 256]
    rnd = sorted(int(e) for e in np.random.default_rng(2700 + L).choice(pool, 8, replace=False))
    out[str(L)] = dict(flagged=fl, random=rnd, val_rows={str(e): int(rows[e]) for e in fl + rnd})
    print(L, fl, rnd, [int(rows[e]) for e in fl + rnd])
json.dump(out, open("experts.json", "w"), indent=1)
