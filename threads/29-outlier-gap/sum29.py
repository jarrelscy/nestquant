"""T29 table: per arm, val routed % vs EXL3-K (T27 refs, same H), vs shipped (prod = nq-encode-v1 art) and vs EXL3-K with
the same extra rotation (drot*/xrot* arms), mean/worst over expert groups.
  python sum29.py [--arms a,b] [--per-expert]"""
import os, sys, json, glob, argparse
import numpy as np

RES = "/tmp/nestquant/29-outlier-gap/res"
EARLY = ["3:138", "4:229", "4:98", "4:63", "4:160", "5:21", "5:204", "6:154", "5:118", "3:175"]
BAND = ["30:222", "39:37", "29:61", "30:158", "42:199", "45:229"]
CTRL = ["30:109", "30:126", "30:160", "30:7"]
GROUPS = dict(early=EARLY, band=BAND, ctrl=CTRL)


def load():
    out = {}
    for d in glob.glob(f"{RES}/L*_E*"):
        L, E = os.path.basename(d)[1:].split("_E")
        key = f"{L}:{E}"
        out[key] = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{d}/*.json")}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms"); ap.add_argument("--per-expert", action="store_true")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    R = load()
    arms = a.arms.split(",") if a.arms else sorted({x for v in R.values() for x in v})
    rows = []
    for arm in arms:
        for g, ex in GROUPS.items():
            for Lv in (2, 4):
                vx, vb, bp, vr, n = [], [], [], [], 0
                rt = [t for t in arm.split("+") if t.startswith("drot") or t.startswith("xrot")]
                xr = f"X{Lv}+" + "+".join(rt) if rt else None
                for e in ex:
                    r = R.get(e, {})
                    if arm not in r or f"X{Lv}" not in r or "prod" not in r:
                        continue
                    ev = r[arm]["eval"].get(f"L{Lv}")
                    if ev is None:
                        continue
                    x = r[f"X{Lv}"]["eval"][f"L{Lv}"]["routed"]; b = r["prod"]["eval"][f"L{Lv}"]["routed"]
                    vx.append(100 * (ev["routed"] / x - 1)); vb.append(100 * (ev["routed"] / b - 1))
                    if xr and xr in r and xr != arm:
                        vr.append(100 * (ev["routed"] / r[xr]["eval"][f"L{Lv}"]["routed"] - 1))
                    m = r[arm].get("meta", {})
                    if "bpw" in m:
                        bp.append(m["bpw"][str(Lv)] if str(Lv) in m["bpw"] else m["bpw"][Lv])
                    if a.per_expert:
                        print(f"  {arm:28s} {e:7s} L{Lv} {ev['routed']:7.3f}  vsX {vx[-1]:+7.2f}  vsBase {vb[-1]:+7.2f}")
                if vx:
                    rows.append((arm, g, Lv, len(vx), np.mean(vx), np.max(vx), np.mean(vb), np.max(vb),
                                 np.mean(vr) if vr else float("nan"), np.max(vr) if vr else float("nan"),
                                 np.mean(bp) if bp else float("nan")))
    hdr = f"{'arm':28s} {'grp':5s} lvl  n  vsEXL3 mean/worst  vsShipped mean/worst  vsEXL3+rot mean/worst   bpw"
    print(hdr)
    for arm, g, Lv, n, mx, wx, mb, wb, mr, wr, bp in rows:
        print(f"{arm:28s} {g:5s} L{Lv} {n:3d}  {mx:+7.2f} {wx:+7.2f}   {mb:+7.2f} {wb:+7.2f}     {mr:+7.2f} {wr:+7.2f}   {bp:.4f}")


if __name__ == "__main__":
    main()
