"""T29 flat-ics scan (down only; gate/up kept as shipped): per expert, val routed score of
  art      shipped nq-encode-v1 artifact
  f128     down re-encoded at Had128 with flat ics  (no format change)
  f512     down re-encoded at in_had_down 512 with flat ics
  i512     (--ics) down at 512 with the EXL3 ics  (for reference)
-> res/L{L}_E{E}/{arm}.json in the t29_run schema (EXL3 refs X2/X4 via t29_run.py), so sum29/table29 read them.
  python flat29.py [--arms f128,f512] L:E [L:E ...]
  python flat29.py --cv L:E ...        ics coefficient of variation per projection (prep only, no encode)"""
import os, sys, json, time, argparse, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq29_had as NH
import nq_decode as D

SHIP = "/tmp/nestquant/nq-encode-v1"
SRC = "/tmp/nestquant/src/glm53-fp8"
RES = "/tmp/nestquant/29-outlier-gap/res"
ARMS = dict(f128=(128, True), f512=(512, True), i512=(512, False), i128=(128, False))


@contextlib.contextmanager
def ics_probe(out):
    Q = NH._Q(); o = Q.block_rms

    def g(x, dim, keepdim=False, blocksize=32):
        r = o(x, dim, keepdim, blocksize)
        if dim == 1:
            rf = r.flatten().double()
            out.append(dict(k=int(x.shape[0]), cv=float(rf.std() / rf.mean()), min=float(rf.min() / rf.mean()),
                            max=float(rf.max() / rf.mean())))
        return r
    try:
        Q.block_rms = g
        yield
    finally:
        Q.block_rms = o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="f128,f512"); ap.add_argument("--cv", action="store_true")
    ap.add_argument("--force", action="store_true"); ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    import nq_layer as NL, nq_encode as NE
    from orbit_duet.source import weights
    cfg = json.load(open(f"{SHIP}/L3/manifest.json"))["config"]
    cap = NL.open_stats(cfg["stats"], cfg["stats_mm"], cfg["mm_w"])
    if a.cv:
        for pr in a.pairs:
            L, E = map(int, pr.split(":"))
            HG, _ = NL.expert_HG(cap, L, E); Ws = weights(SRC, L, E)
            o = {}
            for pi, pn in enumerate(NE.PROJ):
                rec = []
                with ics_probe(rec):
                    P = NE.prep(Ws[pi].float(), HG["H"][pi], 1, NE.PROD["sigma"][pn], G=HG["G"][pi],
                                ks=(2, float(NE.PROD["res_K"][pn])))
                NE.free(P)
                o[pn] = rec[0]
            print(json.dumps(dict(L=L, E=E, **{p: {k: round(v, 4) for k, v in d.items()} for p, d in o.items()})), flush=True)
            torch.cuda.empty_cache()
        return
    import t29_run as T, nq19_load, harness as h
    vcap = nq19_load.Capture(root=f"{SHIP}/_stats")
    arms = a.arms.split(",")
    for pr in a.pairs:
        L, E = map(int, pr.split(":"))
        od = f"{RES}/L{L}_E{E}"; os.makedirs(od, exist_ok=True)
        todo = [x for x in arms if a.force or not os.path.exists(f"{od}/{x}.json")]
        if not todo:
            continue
        art = torch.load(f"{SHIP}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        HG, _ = NL.expert_HG(cap, L, E); Ws = weights(SRC, L, E)
        dm = vcap.expert_data(L, E, "val", source=SRC); sc = T.Scorer(h, dm); Wt = [w.cpu() for w in dm.teacher]
        for arm in todo:
            t0 = time.time()
            w, flat = ARMS[arm]
            new, _ = NH.refit_down(art, Ws[2].float(), HG, w, flat=flat)
            ev = {f"L{Lv}": sc.score(NH.decode_expert29(new, Lv), Wt, HG) for Lv in (2, 4)}
            bits = new["meta"]["info"]["down"]["bits"]
            meta = dict(bpw={2: sum(new["meta"]["info"][p]["bits"][2] for p in NE.PROJ) / 3,
                             4: sum(new["meta"]["info"][p]["bits"][4] for p in NE.PROJ) / 3},
                        down_bits={2: bits[2], 4: bits[4]}, in_had_down=w, flat_ics=flat)
            json.dump(dict(layer=L, expert=E, arm=arm, eval=ev, meta=meta, s=round(time.time() - t0),
                           time=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{od}/{arm}.json", "w"), indent=1)
            print(f"L{L} E{E} {arm:5s} " + " ".join(f"{k} {v['routed']:.3f}" for k, v in ev.items()) +
                  f" bpw4 {meta['bpw'][4]:.4f} {time.time()-t0:.0f}s", flush=True)
            del new; torch.cuda.empty_cache()
        del art, dm, sc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
