"""Stage-1 reproduction check vs orbit-duet pilot rows (training sample x, route index) and matched capture."""
import sys, json, numpy as np, torch
L = int(sys.argv[1]); OUTD = sys.argv[2]; experts = [int(v) for v in sys.argv[3].split(",")]
ORB = "/home/coder/git/orbit-duet/runs"
T = 65536
x = torch.from_numpy(np.fromfile(f"{OUTD}/acts/L{L}/x.bf16", dtype=np.uint16)[:T * 6144].reshape(T, 6144).view(np.int16)).view(torch.bfloat16)
ids = torch.from_numpy(np.fromfile(f"{OUTD}/acts/L{L}/ids.u8", dtype=np.uint8)[:T * 8].reshape(T, 8)).long()
p = torch.from_numpy(np.fromfile(f"{OUTD}/acts/L{L}/p.f32", dtype=np.float32)[:T * 8].reshape(T, 8))
res = {}
for E in experts:
    ts = torch.load(f"{ORB}/glm53_pilot_matched_l{L}/statistics/l{L}_e{E}_training_sample.pt", weights_only=True, mmap=True)
    st = torch.load(f"{ORB}/glm53_pilot_matched_l{L}/statistics/l{L}_e{E}.pt", weights_only=True, mmap=True)["metadata"]["source"]
    nr, nc = st["routed_rows"], st["context_rows"]
    rows, slots = torch.where(ids == E)
    ctx = torch.randperm(T, generator=torch.Generator().manual_seed(20260925))[:nc]
    mine = torch.cat([x[rows], x[ctx]])
    pm = p[rows, slots]
    d = (mine.float() - ts["x"].float())
    res[E] = dict(routed=[len(rows), nr], x_bitwise=bool(torch.equal(mine.view(torch.int16), ts["x"].view(torch.int16))),
                  x_maxabs=float(d.abs().max()), x_rel=float(d.norm() / ts["x"].float().norm()),
                  n_diff_elems=int((d != 0).sum()),
                  p_bitwise=bool(torch.equal(pm, ts["p"][:nr])) if len(rows) == nr else None,
                  positions_match=bool(torch.equal(torch.cat([rows, ctx]), ts["positions"])) if len(rows) == nr else None)
m = torch.load(f"{OUTD}/eval/matched/layer_{L}.pt", weights_only=True)
o = torch.load(f"{ORB}/glm53_matched_context_pilot_v1_capture/layer_{L}.pt", weights_only=True, mmap=True)
dm = m["x"].float() - o["x"].float()
res["matched"] = dict(x_bitwise=bool(torch.equal(m["x"].view(torch.int16), o["x"].view(torch.int16))),
                      x_rel=float(dm.norm() / o["x"].float().norm()), n_diff=int((dm != 0).sum()),
                      ids_equal=bool(torch.equal(m["ids"], o["ids"])), p_equal=bool(torch.equal(m["p"], o["p"])),
                      meta_equal=all(torch.equal(m[k], o[k]) for k in ("document_ids", "token_positions")) and m["domains"] == o["domains"])
print(json.dumps(res, indent=1))
