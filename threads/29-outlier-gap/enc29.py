"""T29 drot4 refit: re-encode the down projection at in_had_down = WIDTH (gate/up byte-identical from shipped).
  python enc29.py --layer L --experts a:b [--width 512] [--score]
-> /tmp/nestquant/29-outlier-gap/nq-encode-h512/L{L}/experts/E{E}.pt
Same H (T12 nq_layer.expert_HG(open_stats(_stats, _stats_mm, 0.25))) and PROD encoder as the shipped nq-encode-v1.
Per expert checks: decode_expert29(new) gate/up == pinned decode of shipped; down == encoder dense (both levels).
--score: also t29 val routed score (vs the drot4 arm in res/)."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq29_had as NH
import nq_decode as D

SHIP = "/tmp/nestquant/nq-encode-v1"
SRC = "/tmp/nestquant/src/glm53-fp8"
OUT = "/tmp/nestquant/29-outlier-gap/nq-encode-h512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--experts", default="0:256")
    ap.add_argument("--width", type=int, default=512); ap.add_argument("--out", default=OUT)
    ap.add_argument("--gpu-gb", type=float, default=12); ap.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--ics", action="store_true", help="keep EXL3 ics (default flat)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq_layer as NL
    from orbit_duet.source import weights
    L = a.layer
    cfg = json.load(open(f"{SHIP}/L{L}/manifest.json"))["config"]
    cap = NL.open_stats(cfg["stats"], cfg["stats_mm"], cfg["mm_w"])
    od = f"{a.out}/L{L}/experts"; os.makedirs(od, exist_ok=True)
    if a.score:
        import t29_run as T, nq19_load, harness as h
        vcap = nq19_load.Capture(root=f"{SHIP}/_stats")
    e0, e1 = map(int, a.experts.split(":"))
    for E in range(e0, e1):
        dst = f"{od}/E{E}.pt"
        if os.path.exists(dst) and not a.force:
            continue
        t0 = time.time()
        art = torch.load(f"{SHIP}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        HG, _ = NL.expert_HG(cap, L, E)
        Ws = weights(SRC, L, E)
        new, dense = NH.refit_down(art, Ws[2].float(), HG, a.width, flat=not a.ics)
        # checks: gate/up tensors are the shipped objects; decode29 down == encoder dense; gate/up == shipped decode
        assert new["gate"] is art["gate"] and new["up"] is art["up"]
        chk = {}
        for Lv in (2, 4):
            q = NH.decode_expert29(new, Lv); q0 = D.decode_expert(art, Lv)
            perm = new["meta"].get("inter_perm")
            dd = dense[Lv].cuda()
            if perm is not None:
                dd = dd[:, torch.argsort(torch.as_tensor(perm, device="cuda"))]
            chk[Lv] = bool(torch.equal(q[0], q0[0]) and torch.equal(q[1], q0[1]) and torch.equal(q[2], dd))
        assert all(chk.values()), (L, E, chk)
        tmp = dst + ".tmp"; torch.save(new, tmp); os.replace(tmp, dst)
        msg = f"L{L} E{E} w{a.width} bits {new['meta']['info']['down']['bits'][2]:.4f}/{new['meta']['info']['down']['bits'][4]:.4f} " \
              f"rate {new['meta']['rate']:.4f} ship {art['meta']['rate']:.4f} chk ok"
        if a.score:
            dm = vcap.expert_data(L, E, "val", source=SRC)
            sc = T.Scorer(h, dm)
            ev = {Lv: sc.score(NH.decode_expert29(new, Lv), [w.cpu() for w in dm.teacher], HG)["routed"] for Lv in (2, 4)}
            msg += f" val L2 {ev[2]:.3f} L4 {ev[4]:.3f}"
            del dm, sc
        print(msg + f" {time.time()-t0:.0f}s", flush=True)
        del art, new, dense; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
