"""T35 ft pilot: the tuned experts written by nq35_ftp.py must decode bit-exactly through the packed serving path (CPU).

For each tuned arm dir OUT/{arm}/L{L}/E{e}.pt (same codes as enc_b175, only su/sv/U/V changed):
  a. tuned proj meta == the layer manifest's proj_meta (the packer writes one meta per layer)
  b. nq_layer.split_expert -> TP8 shard parts -> nq_layer.assemble-style rebuild == tuned planes (tensor-equal), and
     nq_decode.decode_matrix of the rebuild == decode of the tuned artifact (the after-eval path), levels 2 and 4, bitwise
  c. a synthetic layer root PACK/L{L} (manifest copy + tp{s}.pt holding only the tuned experts) is run through
     res2_testvec.py for every tuned expert x TP4 rank: kernel planes (nqload.kernel_expert) -> moe.dense_W == nq_decode
     (gate|up, down x L2, L4), and the nq-res-v2 res .pt + P4 record round trip == decode, bitwise
Writes OUT/check.json.  Nothing leaves the box.
  python nq35_ftp_check.py --out /tmp/nestquant/35-nq15/ftp --root /tmp/nestquant/35-nq15/enc_b175 --layer 40
"""
import os, sys, json, glob, shutil, argparse, subprocess
T = "/home/coder/git/nestquant/threads"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, f"{T}/12-reference-encoder", f"{T}/05-exl3-harness"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch                        # noqa: E402
import nq15                         # noqa: E402,F401  base-K decoder
import nq_decode as D               # noqa: E402
import nq_layer as NL               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--root", default="/tmp/nestquant/35-nq15/enc_b175")
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--arms", default="")
ap.add_argument("--code", default="/home/coder/git/nestquant")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
L = a.layer
man = json.load(open(f"{a.root}/L{L}/manifest.json"))
PROJ = ("gate", "up", "down")


def teq(x, y, path="", bad=None):
    bad = [] if bad is None else bad
    if torch.is_tensor(x):
        if not (torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
            bad.append(path)
    elif isinstance(x, dict):
        for k in x:
            if k in y:
                teq(x[k], y[k], f"{path}.{k}", bad)
    elif isinstance(x, (list, tuple)):
        for i, (u, v) in enumerate(zip(x, y)):
            teq(u, v, f"{path}[{i}]", bad)
    return bad


def rebuild(parts):
    """nq_layer.assemble on in-memory shard parts"""
    art = {}
    for pn in PROJ:
        meta = dict(man["proj_meta"][pn])
        cat = lambda key: [p[pn][key] for p in parts]
        if pn == "down":
            suh2, svh2, suh4, svh4 = torch.cat(cat("suh2")), parts[0][pn]["svh2"], torch.cat(cat("suh4")), parts[0][pn]["svh4"]
        else:
            suh2, svh2, suh4, svh4 = parts[0][pn]["suh2"], torch.cat(cat("svh2")), parts[0][pn]["suh4"], torch.cat(cat("svh4"))
        base = dict(shards=cat("base"), suh=suh2, svh=svh2)
        if "var" in parts[0][pn]:
            base["var"] = cat("var")
        p4 = dict(shards=cat("p4"), word=cat("word"), suh=suh4, svh=svh4)
        if "lrU2" in parts[0][pn]:
            if pn == "down":
                V, U2, U4 = torch.cat(cat("lrV"), 1), parts[0][pn]["lrU2"], parts[0][pn]["lrU4"]
            else:
                V = parts[0][parts[0][pn].get("lrV_from", pn)]["lrV"]
                U2, U4 = torch.cat(cat("lrU2"), 1), torch.cat(cat("lrU4"), 1)
            base["lr"] = dict(V=V, U2=U2); p4["lr"] = dict(U4=U4)
        art[pn] = dict(base=base, p4=p4, meta=meta)
    return art


def dec(art, lv):
    rot = {p: D.rotated_levels(art[p], "cpu") for p in PROJ}
    return [D.decode_matrix(art[p], lv, "cpu", rot=rot[p]) for p in PROJ]


R = {}
dirs = sorted(glob.glob(f"{a.out}/n*_wrt*/L{L}"))
if a.arms:
    dirs = [d for d in dirs if d.split("/")[-2] in a.arms.split(",")]
for d in dirs:
    arm = d.split("/")[-2]
    Es = sorted(int(os.path.basename(f)[1:-3]) for f in glob.glob(f"{d}/E*.pt"))
    pack = f"{a.out}/pack/{arm}/L{L}"; os.makedirs(pack, exist_ok=True)
    shutil.copy(f"{a.root}/L{L}/manifest.json", f"{pack}/manifest.json")
    shards = [dict() for _ in range(NL.NSH)]
    rr = dict(experts=Es, meta_diff={}, rebuild_diff={}, decode_equal={}, changed_vs_enc={}, res2={})
    for E in Es:
        tu = torch.load(f"{d}/E{E}.pt", weights_only=False, map_location="cpu")
        en = torch.load(f"{a.root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        rr["meta_diff"][E] = {pn: sorted(k for k in man["proj_meta"][pn]
                                         if json.dumps(man["proj_meta"][pn][k], sort_keys=True, default=str)
                                         != json.dumps(tu[pn]["meta"].get(k), sort_keys=True, default=str)) for pn in PROJ}
        rr["changed_vs_enc"][E] = {pn: teq(en[pn], tu[pn], pn) for pn in PROJ}
        sp = NL.split_expert(tu)
        for s in range(NL.NSH):
            shards[s][E] = sp[s]
        rb = rebuild(sp)
        rr["rebuild_diff"][E] = {pn: teq({k: tu[pn][k] for k in ("base", "p4")}, {k: rb[pn][k] for k in ("base", "p4")}, pn)
                                 for pn in PROJ}
        eq = {}
        for lv in (2, 4):
            A, B = dec(tu, lv), dec(rb, lv)
            eq[lv] = all(torch.equal(x, y) for x, y in zip(A, B))
        rr["decode_equal"][E] = eq
        print(f"{arm} E{E}: meta_diff {rr['meta_diff'][E]} changed {rr['changed_vs_enc'][E]} "
              f"rebuild_diff {rr['rebuild_diff'][E]} decode_eq {eq}", flush=True)
    for s in range(NL.NSH):
        torch.save(shards[s], f"{pack}/tp{s}.pt")
    for E in Es:
        for r in range(4):
            p = subprocess.run([sys.executable, f"{HERE}/res2_testvec.py", "--root", f"{a.out}/pack/{arm}", "--layer", str(L),
                                "--expert", str(E), "--rank", str(r), "--out", f"{a.out}/pack/{arm}/testvec",
                                "--code", a.code], capture_output=True, text=True,
                               env=dict(os.environ, NT=str(a.threads)))
            ok = p.returncode == 0 and "round trip == decode, bitwise" in p.stdout
            rr["res2"][f"E{E}_r{r}"] = ok
            if not ok:
                print(p.stdout[-2000:], p.stderr[-3000:], flush=True)
            print(f"{arm} E{E} rank{r}: res2_testvec {'OK' if ok else 'FAIL'}", flush=True)
    rr["pass"] = (all(not any(v.values()) for v in rr["meta_diff"].values())
                  and all(not any(v.values()) for v in rr["rebuild_diff"].values())
                  and all(all(v.values()) for v in rr["decode_equal"].values()) and all(rr["res2"].values()))
    R[arm] = rr
    print(f"{arm}: PASS={rr['pass']}", flush=True)
    json.dump(R, open(f"{a.out}/check.json", "w"), indent=1, default=str)
print("all pass" if R and all(r["pass"] for r in R.values()) else "NOT all pass", flush=True)
