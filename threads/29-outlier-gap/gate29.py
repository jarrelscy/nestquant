"""T29 in_had_down gates.
  python gate29.py dec L [a:b]      (a) decode_expert29 == pinned nq_decode.decode_expert on shipped L (both levels, torch.equal)
  python gate29.py enc128 L:E ...   (b) encode_down(width 128) reproduces the shipped down planes byte-exactly
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq29_had as NH
import nq_decode as D

ROOT = "/tmp/nestquant/nq-encode-v1"


def tensors(x, pre=""):
    if torch.is_tensor(x):
        yield pre, x
    elif isinstance(x, dict):
        for k in sorted(x):
            yield from tensors(x[k], f"{pre}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from tensors(v, f"{pre}[{i}]")


def same_bytes(a, b):
    ta, tb = dict(tensors(a)), dict(tensors(b))
    if ta.keys() != tb.keys():
        return False, f"keys {sorted(set(ta) ^ set(tb))[:5]}"
    for k in ta:
        x, y = ta[k], tb[k]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x.cpu().view(torch.uint8) if x.dtype != torch.bool else x.cpu(),
                                                                         y.cpu().view(torch.uint8) if y.dtype != torch.bool else y.cpu()):
            return False, k
    return True, ""


def gate_dec(L, rng):
    e0, e1 = map(int, rng.split(":"))
    bad = 0; n = 0; t0 = time.time()
    for E in range(e0, e1):
        p = f"{ROOT}/L{L}/experts/E{E}.pt"
        if not os.path.exists(p):
            continue
        art = torch.load(p, weights_only=False, map_location="cpu")
        assert NH.width_of(art) == 128
        for Lv in (2, 4):
            a = D.decode_expert(art, Lv); b = NH.decode_expert29(art, Lv)
            bad += not all(torch.equal(x, y) for x, y in zip(a, b))
        n += 1
        if n % 16 == 0:
            print(f'  L{L} E{E} n {n} bad {bad} {time.time()-t0:.0f}s', flush=True)
    print(f"GATE_A L{L} {rng}: {n} experts, {bad} mismatches -> {'PASS' if bad == 0 and n else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)


def gate_enc(pairs):
    import nq_layer as NL
    from orbit_duet.source import weights
    torch.backends.cuda.matmul.allow_tf32 = False
    man = json.load(open(f"{ROOT}/L3/manifest.json"))["config"]
    cap = NL.open_stats(man["stats"], man["stats_mm"], man["mm_w"])
    for pr in pairs:
        L, E = map(int, pr.split(":"))
        t0 = time.time()
        art = torch.load(f"{ROOT}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        HG, _ = NL.expert_HG(cap, L, E)
        Wd = weights("/tmp/nestquant/src/glm53-fp8", L, E)[2].float()
        planes, dense, info = NH.encode_down(Wd, HG, 128)
        ok, why = same_bytes(planes, art["down"])
        mok = planes["meta"] == art["down"]["meta"]
        print(f"GATE_B L{L} E{E}: planes byte-equal {ok} {why} meta-equal {mok} bits {info['bits']} "
              f"shipped {art['meta']['info']['down']['bits']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dec":
        gate_dec(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "0:256")
    else:
        gate_enc(sys.argv[2:])
