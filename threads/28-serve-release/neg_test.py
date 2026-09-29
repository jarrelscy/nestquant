"""Negative tests for nq_check.py on a copy of one layer of a local build: each tamper must make the check FAIL with the
expected message, and the untouched copy must PASS.
  python neg_test.py SRC_TPDIR REF_ROOT WORK [L=3]"""
import os, sys, json, shutil, subprocess, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nq_check as NC
src, ref, work = sys.argv[1:4]; L = int(sys.argv[4]) if len(sys.argv) > 4 else 3
PY = sys.executable

def fresh():
    d = f"{work}/serving/tp4"
    shutil.rmtree(work, ignore_errors=True); os.makedirs(d)
    for f in os.listdir(src):
        if f.endswith(".json") or f == "COMPLETE":
            shutil.copy(f"{src}/{f}", d)
    man = json.load(open(f"{d}/manifest.json")); tp = man["tp"]
    os.makedirs(f"{d}/layers"); shutil.copy(f"{src}/layers/L{L}.json", f"{d}/layers/")
    for r in range(tp):
        for sub in (f"rank{r}", f"res/rank{r}"):
            os.makedirs(f"{d}/{sub}", exist_ok=True)
        shutil.copy(f"{src}/rank{r}/L{L}.bin", f"{d}/rank{r}/"); shutil.copy(f"{src}/res/rank{r}/L{L}.pt", f"{d}/res/rank{r}/")
    return d, man

def run(d, extra=()):
    p = subprocess.run([PY, f"{HERE}/nq_check.py", d, "--ref", ref, "--allow-incomplete", "--layers", str(L), "--experts", "2",
                        *extra], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return p.returncode, p.stdout + p.stderr

def picked(d, man, r):
    idx = json.load(open(f"{d}/rank{r}.json")); e = idx["layers"][str(L)]; b = json.load(open(f"{d}/layers/L{L}.json"))
    rg = {int(k): v for k, v in e["rg"].items()}; rd = {int(k): v for k, v in e["rd"].items()}
    return NC.pick_experts(random.Random(1000 + L), b, rg, rd, man["NE"], 2)

def flip(p, off):
    with open(p, "r+b") as f:
        f.seek(off); b = f.read(1); f.seek(off); f.write(bytes([b[0] ^ 0x10]))

res = {}
ONLY = os.environ.get("NEG_ONLY")                         # substring filter on case names
def case(name, tamper, expect, extra=()):
    if ONLY and ONLY not in name:
        return
    d, man = fresh(); tamper(d, man); rc, out = run(d, extra)
    fails = [l for l in out.splitlines() if "FAIL" in l]
    ok = (rc == 0 and not expect) or (rc != 0 and all(any(x in l for l in fails) for x in expect))
    res[name] = ok; print(f"{'ok  ' if ok else 'BAD '} {name}: rc {rc}; " + " | ".join(fails[:4]), flush=True)
    if not ok:
        print(out[-3000:])

rb = lambda d: json.load(open(f"{d}/manifest.json"))["layout"]["rec_bytes"]
seg = lambda d: json.load(open(f"{d}/manifest.json"))["layout"]["seg"]
R = L % 4
case("clean", lambda d, m: None, [])
# a P4 byte of a checked expert on the checked rank: hash + record bytes + level-4 decode (level 2 stays exact)
case("record byte (gu.p4)", lambda d, m: flip(f"{d}/rank{R}/L{L}.bin", picked(d, m, R)[0] * rb(d) + seg(d)["gu.p4"][0] + 1000),
     ["sha256", "(a) record block bytes", "(c) level-4 gate|up decode"])
# a d4 block word of a checked expert
case("record byte (dn.d4)", lambda d, m: flip(f"{d}/rank{R}/L{L}.bin", picked(d, m, R)[-1] * rb(d) + seg(d)["dn.d4"][0] + 8),
     ["sha256", "(a) record block bytes", "(c) level-4 down decode"])
# a record byte of an unchecked expert on another rank: only the hash sees it
case("record byte, unsampled expert", lambda d, m: flip(f"{d}/rank{(R+1)%4}/L{L}.bin", 17 * rb(d) + 5), ["sha256"])
case("record byte, unsampled, --hash none (expected blind)", lambda d, m: flip(f"{d}/rank{(R+1)%4}/L{L}.bin", 17 * rb(d) + 5), [],
     ["--hash", "none"])
def bad_offset(d, m):
    p = f"{d}/rank{R}.json"; j = json.load(open(p)); j["layers"][str(L)]["offset"] += rb(d); json.dump(j, open(p, "w"))
case("wrong offset in index", bad_offset, ["offset"])
def bad_recbytes(d, m):
    p = f"{d}/rank{R}.json"; j = json.load(open(p)); j["rec_bytes"] += 4096; json.dump(j, open(p, "w"))
case("wrong rec_bytes in index", bad_recbytes, ["header != manifest"])
def truncate(d, m):
    p = f"{d}/rank{R}/L{L}.bin"; os.truncate(p, os.path.getsize(p) - 4096)
case("truncated block", truncate, ["size"])
def res_flip(d, m):
    import torch
    p = f"{d}/res/rank{R}/L{L}.pt"; x = torch.load(p, weights_only=False); E = picked(d, m, R)[0]
    x["gu_base"][E].view(-1)[123] ^= 1; torch.save(x, p)
case("resident base plane (re-saved)", res_flip, ["size", "(b) resident gu.base", "(c) level-2 gate|up decode"])
case("resident file byte (size kept)", lambda d, m: flip(f"{d}/res/rank{(R+2)%4}/L{L}.pt", os.path.getsize(f"{d}/res/rank{(R+2)%4}/L{L}.pt") // 2), ["sha256"])
def alloc(d, m):
    p = f"{d}/layers/L{L}.json"; j = json.load(open(p)); j["default_allocation"]["level4_experts"][0] ^= 1; json.dump(j, open(p, "w"))
case("fixed set edited in layer block", alloc, ["layer block hash", "fixed set != reference"])
def alloc_rehash(d, m):
    import nq_release as NR
    p = f"{d}/layers/L{L}.json"; j = json.load(open(p)); j["default_allocation"]["level4_experts"][0] ^= 1
    j["layer_hash"] = NR.layer_hash(j); json.dump(j, open(p, "w"))
case("fixed set edited + rehashed", alloc_rehash, ["fixed set != reference", "layer_hash differs"])
print("NEG TESTS", "PASS" if all(res.values()) else "FAIL", f"({sum(res.values())}/{len(res)})")
