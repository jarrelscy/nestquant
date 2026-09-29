"""T29 finalize of a refit layer in ROOT29 (= nq25_finalize minus refcheck, + the in_had_down manifest fields):
  1. T12 nq_layer finalize (--experts 0:0) with the campaign's stats / fixed set / source -> tp{s}.pt + manifest.json
  2. manifest: config.in_had_down / ics_down / had_sign_seed + rotation block, campaign block carried from shipped + t29
  3. T25 nq25_st.convert (tp{s}.safetensors, round trip) + check_decode (assembled == E.pt, all experts)
  4. decode29 check: assemble29 from the safetensors == E.pt under decode_expert29 (every expert, L2 + L4), and
     gate/up of every expert byte-identical to shipped
  python fin29.py --layer L [--root ROOT29]
ROOT29 also gets L7..L77 (and L3..L6 not refit) as symlinks to the shipped nq-encode-v1 layers (--link)."""
import os, sys, json, time, argparse, subprocess
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq29_had as NH
import nq_decode as D

SHIP = "/tmp/nestquant/nq-encode-v1"
ROOT29 = "/tmp/nestquant/29-outlier-gap/nq-encode-h512"
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T25 = "/home/coder/git/nestquant/threads/25-campaign"
PY = sys.executable
REFIT = (3, 4, 5, 6)


def link(root):
    for L in range(3, 78):
        d = f"{root}/L{L}"
        if L in REFIT:
            continue
        if not os.path.lexists(d):
            os.symlink(f"{SHIP}/L{L}", d)
    for x in ("_stats", "_stats_mm"):
        if not os.path.lexists(f"{root}/{x}"):
            os.symlink(os.path.realpath(f"{SHIP}/{x}"), f"{root}/{x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int); ap.add_argument("--root", default=ROOT29); ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    if a.link:
        link(a.root); print("linked"); return
    L = a.layer; R = a.root; t0 = time.time()
    sman = json.load(open(f"{SHIP}/L{L}/manifest.json"))
    c = sman["campaign"]
    ed = f"{R}/L{L}/experts"
    have = sorted(int(f[1:-3]) for f in os.listdir(ed) if f.endswith(".pt"))
    assert have == list(range(256)), f"L{L}: {len(have)} experts"
    cmd = [PY, f"{T12}/nq_layer.py", "--layer", str(L), "--stats", f"{SHIP}/_stats", "--out", R, "--source", c["source"],
           "--fixed-set", c["fixed_set"]["path"], "--experts", "0:0", "--stats-mm", f"{SHIP}/_stats_mm", "--mm-w", "0.25"]
    print("+", " ".join(cmd), flush=True)
    assert subprocess.call(cmd) == 0
    p = f"{R}/L{L}/manifest.json"; man = json.load(open(p))
    assert man["default_allocation"] == sman["default_allocation"], "default allocation differs from shipped"
    for k in ("format", "rate", "base_var", "lam", "inner", "sigma", "res_K", "lr", "stats", "stats_mm", "mm_w"):
        if k in sman["config"]:
            assert man["config"].get(k) == sman["config"][k] or k == "rate", (k, man["config"].get(k), sman["config"][k])
    man["config"].update(in_had_down=512, ics_down="flat", had_sign_seed=91426)
    man["rotation"] = dict(
        down_in=dict(had=512, blocks="4 x 512 over k = 2048 = one block per TP4 rank (TP8 shard pair 2s, 2s+1)",
                     signs="random signs folded into suh2/suh4 exactly as for Had128 (encoder seed 91426, same RNG order)",
                     decode="W = diag(svh) Had128_n^T ... : nq_decode.dense_from_rotated with the k-side Hadamard at 512 "
                            "(threads/29-outlier-gap/nq29_had.decode_expert29)"),
        down_out=128, gate_up_in=128, gate_up_out=128,
        encoder="down: sign + Had512 input rotation, EXL3 input-channel scale (ics) replaced by its RMS (flat); "
                "gate/up: byte-identical to the shipped nq-encode-v1 layer")
    camp = dict(c)
    camp["t29"] = dict(refit="L3-L6 down in_had_down 512 + flat ics", shipped_manifest_sha256=__import__("hashlib").sha256(
        open(f"{SHIP}/L{L}/manifest.json", "rb").read()).hexdigest(), code="threads/29-outlier-gap/{nq29_had,enc29,fin29}.py",
        time=time.strftime("%Y-%m-%d %H:%M:%S"))
    man["campaign"] = camp
    json.dump(man, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    sys.path.insert(0, T25)
    import nq25_st as S
    assert S.convert(R, L), "st convert"
    assert S.check_decode(R, L, None), "st check_decode"
    man2 = json.load(open(p))
    assert man2["config"]["in_had_down"] == 512 and "rotation" in man2 and man2["campaign"].get("t29")
    # decode29 of the safetensors-assembled artifact == E.pt; gate/up == shipped (raw bytes)
    from gate29 import same_bytes
    bad = 0
    for E in range(256):
        art = torch.load(f"{ed}/E{E}.pt", weights_only=False, map_location="cpu")
        sa = S.assemble(R, L, E); sa["meta"] = {NH.FIELD: int(man2["config"]["in_had_down"])}
        shp = torch.load(f"{SHIP}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        ok = same_bytes(art["gate"], shp["gate"])[0] and same_bytes(art["up"], shp["up"])[0]
        if E % 16 == 0 or not ok:
            for Lv in (2, 4):
                ok &= all(torch.equal(x, y) for x, y in zip(NH.decode_expert29(art, Lv), NH.decode_expert29(sa, Lv)))
        bad += not ok
    print(f"[L{L}] fin29: st/E.pt decode29 + gate/up==shipped: {'PASS' if not bad else f'FAIL {bad}'} ({time.time()-t0:.0f}s)", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
