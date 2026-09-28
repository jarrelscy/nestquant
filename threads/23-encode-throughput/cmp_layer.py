"""Compare two nq_layer output trees (reference nq_layer.py vs nq_layer_batch.py) for one layer.

  python cmp_layer.py REF_OUT BAT_OUT L
Checks: every experts/E*.pt tensor + meta value identical (common.compare), every tp{s}.pt file byte-identical
(sha256), manifest.json identical except the "time" stamp.
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import torch


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ra, rb, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
    da, db = f"{ra}/L{L}", f"{rb}/L{L}"
    ok = True
    ea = sorted(f for f in os.listdir(f"{da}/experts") if f.endswith(".pt"))
    eb = sorted(f for f in os.listdir(f"{db}/experts") if f.endswith(".pt"))
    if ea != eb:
        print("expert file sets differ", ea, eb); ok = False
    for f in ea:
        if f not in eb:
            continue
        nt, nb, bad = C.compare(torch.load(f"{da}/experts/{f}", weights_only=False),
                                torch.load(f"{db}/experts/{f}", weights_only=False))
        same_bytes = sha(f"{da}/experts/{f}") == sha(f"{db}/experts/{f}")
        print(f"  experts/{f:<8} tensors {nt} {'IDENTICAL' if not bad else 'MISMATCH ' + str(bad[:4])}"
              f"  file bytes {'same' if same_bytes else 'DIFFER'}")
        ok &= not bad
    for s in range(8):
        a, b = sha(f"{da}/tp{s}.pt"), sha(f"{db}/tp{s}.pt")
        print(f"  tp{s}.pt  {'IDENTICAL' if a == b else 'DIFFER'}  {a[:16]} {b[:16]}")
        ok &= a == b
    ma, mb = json.load(open(f"{da}/manifest.json")), json.load(open(f"{db}/manifest.json"))
    ma.pop("time", None); mb.pop("time", None)
    same = json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)
    if not same:
        for k in sorted(set(ma) | set(mb)):
            if ma.get(k) != mb.get(k):
                print(f"  manifest key {k} differs")
    print(f"  manifest.json (minus time) {'IDENTICAL' if same else 'DIFFER'}")
    ok &= same
    print(f"LAYER L{L}: {'ALL IDENTICAL' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
