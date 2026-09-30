"""regime split of gap (arm B - arm A) on a stream: regime.py STREAM TAG A B [chains-subset] -> table."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
stream, tag, A, B = sys.argv[1:5]
chs = [int(c) for c in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
lab = dict(np.load(f"{S.OUT}/private/labels_{stream}.npz"))
if os.environ.get("SUBCH"):                      # res computed on a chain subset (evalall --chains)
    keep = np.isin(lab["chain"], [int(c) for c in os.environ["SUBCH"].split(",")])
    lab = {k: v[keep] for k, v in lab.items()}
R = {L: np.load(f"{S.OUT}/private/res/{stream}/{tag}/L{L}.npz") for L in S.LAYERS}
base = np.ones(len(lab["pos"]), bool) if chs is None else np.isin(lab["chain"], chs)
rp = lab["rpos"]
regs = {
  "all": base,
  "think(flag0)": base & (lab["ans"] < 0.5), "answer(flag1)": base & (lab["ans"] >= 0.5),
  "chain tok<256": base & (lab["pos"] * 16 < 256), "chain tok<1024": base & (lab["pos"] * 16 < 1024),
  "chain tok>=1024": base & (lab["pos"] * 16 >= 1024),
  "req tok<256": base & (rp < 256), "req tok 256-1024": base & (rp >= 256) & (rp < 1024), "req tok>=1024": base & (rp >= 1024),
  "prose": base & (lab["tcls"] == 0), "code": base & (lab["tcls"] == 1), "math": base & (lab["tcls"] == 2),
}
bands = {"L3-6": range(3, 7), "L7-20": range(7, 21), "L21-40": range(21, 41), "L41-60": range(41, 61), "L61-77": range(61, 78)}
def stat(m, layers=S.LAYERS):
    ha, hb, sh, gc = [], [], [], []
    for L in layers:
        a, b = R[L][A][m], R[L][B][m]
        tot = R[L][A][base][:, 1].sum()
        ha.append(a[:, 0].sum() / a[:, 1].sum()); hb.append(b[:, 0].sum() / b[:, 1].sum())
        sh.append(a[:, 1].sum() / tot); gc.append((b[:, 0].sum() - a[:, 0].sum()) / tot)
    return np.mean(ha) * 100, np.mean(hb) * 100, np.mean(sh) * 100, np.mean(gc) * 100
tot_gap = stat(base)[3]
print(f"[{stream} {tag}] A={A} B={B}  (share = % of salience; gap contrib = pts of the all-slot metric; % of total gap)")
print(f"{'regime':18s} {'blk%':>6s} {'sal%':>6s} {'A':>6s} {'B':>6s} {'B-A':>6s} {'contrib':>7s} {'%gap':>6s}")
for n, m in regs.items():
    if m.sum() == 0: continue
    ha, hb, sh, gc = stat(m)
    print(f"{n:18s} {m.sum()/base.sum()*100:6.1f} {sh:6.1f} {ha:6.2f} {hb:6.2f} {hb-ha:6.2f} {gc:7.2f} {gc/tot_gap*100:6.1f}")
for n, ls in bands.items():
    ha, hb, sh, gc = stat(base, ls)
    print(f"{n:18s} {'':6s} {'':6s} {ha:6.2f} {hb:6.2f} {hb-ha:6.2f}")
