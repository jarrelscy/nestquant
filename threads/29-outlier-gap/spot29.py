"""T29 spot eval of the in_had_down 512 + flat-ics refit vs the shipped nq-encode-v1 layer (upload gate 3).
  python spot29.py --layer L [--experts a:b] [--exl3]
Per expert: val routed score (t29_run.Scorer, = T27 val/base all/routed) of
  ship  = nq_decode.decode_expert(nq-encode-v1/L{L}/experts/E{E}.pt)           (pinned decoder)
  new   = nq29_had.decode_expert29(nq-encode-h512/L{L}/experts/E{E}.pt)        (decoder at in_had_down)
at levels 2 and 4, plus bpw of both.  Experts with < FORCED_BELOW val routed rows are also scored on all/forced
(every val row, weight 1) and judged on that (routed is noise there, as in nq25_spot).
--exl3: also EXL3-2 / EXL3-4 (t29_run.exl3_arm, same H) on the nq25_spot selection (fixed-set member with most rows,
seeded ordinary expert) of each layer.
-> ROOT29/spot29/L{L}_{a}_{b}.jsonl (one line per expert); the gate is: new <= ship at L2 and L4 on every expert."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq29_had as NH
import nq_decode as D

SHIP = "/tmp/nestquant/nq-encode-v1"
ROOT29 = "/tmp/nestquant/29-outlier-gap/nq-encode-h512"
SRC = "/tmp/nestquant/src/glm53-fp8"
FORCED_BELOW = 64
PROJ = ("gate", "up", "down")


def bpw(art):
    nw = {p: art[p]["meta"]["k"] * art[p]["meta"]["n"] for p in PROJ}
    per = {p: D.bits_per_level(art[p]) for p in PROJ}
    return {lv: sum(per[p][lv] * nw[p] for p in PROJ) / sum(nw.values()) for lv in (2, 4)}


def select25(cap, L, fixed_path):
    fixed = [int(e) for e in json.load(open(fixed_path))["fixed_set"][str(L)]]
    ids = torch.load(cap.eval_path(L, "val"), weights_only=True, mmap=True)["ids"]
    rows = np.bincount(ids.flatten().numpy(), minlength=256)
    ok_f = [e for e in fixed if rows[e] >= 256]
    ef = max(ok_f or fixed, key=lambda e: rows[e])
    rest = [e for e in range(256) if e not in fixed]
    ok_o = [e for e in rest if rows[e] >= 256]
    eo = int(np.random.default_rng(L).choice(ok_o)) if ok_o else max(rest, key=lambda e: rows[e])
    return ef, eo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--experts", default="0:256")
    ap.add_argument("--exl3", action="store_true"); ap.add_argument("--root", default=ROOT29)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import t29_run as T, nq19_load, harness as h, nq_layer as NL, nq_encode as NE
    L = a.layer
    sman = json.load(open(f"{SHIP}/L{L}/manifest.json"))
    cap = nq19_load.Capture(root=f"{SHIP}/_stats")
    hcap = NL.open_stats(f"{SHIP}/_stats", f"{SHIP}/_stats_mm", 0.25)
    e0, e1 = map(int, a.experts.split(":"))
    Es = list(range(e0, e1))
    sel = select25(cap, L, sman["campaign"]["fixed_set"]["path"]) if a.exl3 else ()
    if a.exl3:
        Es = list(sel)
    od = f"{a.root}/spot29"; os.makedirs(od, exist_ok=True)
    out = open(f"{od}/L{L}_{'exl3' if a.exl3 else f'{e0}_{e1}'}.jsonl", "a")
    sig = dict(NE.PROD["sigma"])
    for E in Es:
        t0 = time.time()
        ship = torch.load(f"{SHIP}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        new = torch.load(f"{a.root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        assert NH.width_of(new) == 512
        dm = cap.expert_data(L, E, "val", source=SRC)
        n = int((dm.capture["ids"] == E).sum())
        HG = NL.expert_HG(hcap, L, E)[0]
        Ws = [w.cpu() for w in dm.teacher]
        rec = dict(layer=L, expert=E, rows=n, bpw=dict(ship=bpw(ship), new=bpw(new)), eval={})
        arms = dict(ship=lambda lv: D.decode_expert(ship, lv), new=lambda lv: NH.decode_expert29(new, lv))
        if a.exl3:
            for K in (2, 4):
                dense, _ = T.exl3_arm(h, Ws, HG, f"X{K}", [], sig)
                arms[f"X{K}"] = (lambda q: (lambda lv: q))(dense[K])
        sc = T.Scorer(h, dm) if n else None
        for nm, f in arms.items():
            for lv in ((2, 4) if not nm.startswith("X") else (int(nm[1]),)):
                q = [w.cpu() for w in f(lv)]
                r = {}
                if sc is not None:
                    s = sc.score(q, Ws, HG); r = dict(routed=s["routed"], proj={p: s["proj"][p]["eV"] for p in PROJ})
                if n < FORCED_BELOW:
                    if sc is None:
                        sc = T.Scorer(h, dm)
                    r["forced"] = sc.forced(q)
                rec["eval"][f"{nm}/L{lv}" if not nm.startswith("X") else nm] = r
                del q
        key = "routed" if n >= FORCED_BELOW else "forced"
        ev = rec["eval"]
        rec["key"] = key
        rec["worse"] = [lv for lv in (2, 4) if ev[f"new/L{lv}"][key] > ev[f"ship/L{lv}"][key]]
        rec["s"] = round(time.time() - t0)
        if a.exl3:
            rec["role"] = "fixed" if E == sel[0] else "ordinary"
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"L{L} E{E} rows {n} {key} ship {ev['ship/L2'][key]:.2f}/{ev['ship/L4'][key]:.2f} "
              f"new {ev['new/L2'][key]:.2f}/{ev['new/L4'][key]:.2f}"
              + (f" X {ev['X2'][key]:.2f}/{ev['X4'][key]:.2f}" if a.exl3 else "")
              + f" worse {rec['worse']} {rec['s']}s", flush=True)
        del ship, new, dm, sc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
