"""Bit-identity gate: batched encoder (nq_encode_batch) vs the LIVE thread-12 single-pass reference.

  python check_bitid.py ref   [--experts 3:0,16:36,...]      # encode with NE.encode_expert -> SCR/ref/L{L}_E{E}.pt
  python check_bitid.py batch [--experts ...] [--opts ...]    # encode with nq_encode_batch  -> SCR/batch/L{L}_E{E}.pt
  python check_bitid.py cmp   [--experts ...]                 # byte compare every tensor + meta value

The ref files record the sha of thread 12's nq_encode/nq_decode/nq_patvit at encode time; cmp refuses to pass if the
live thread-12 files changed since (re-run `ref` after T12 edits its encoder).
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import torch


def parse(s):
    return [tuple(map(int, x.split(":"))) for x in s.split(",")] if s else C.CHECK_EXPERTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ref", "batch", "cmp", "pair"))
    ap.add_argument("--experts")
    ap.add_argument("--tag", default="batch", help="output subdir for the batch arm")
    ap.add_argument("--group", type=int, default=8, help="experts per batch (nq_encode_batch)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ref-tag", default="ref", help="subdir of the reference arm")
    a = ap.parse_args()
    ex = parse(a.experts)
    os.makedirs(f"{C.SCR}/{a.ref_tag}", exist_ok=True); os.makedirs(f"{C.SCR}/{a.tag}", exist_ok=True)
    if a.cmd == "ref":
        C.setup()
        for L, E in ex:
            p = f"{C.SCR}/{a.ref_tag}/L{L}_E{E}.pt"
            if os.path.exists(p) and not a.force:
                continue
            torch.cuda.synchronize(); t = time.time()
            art = C.ref_encode(L, E)
            torch.cuda.synchronize(); dt = time.time() - t
            art["_ref"] = dict(shas=C.IMPORT_SHAS, dir=C.T12, seconds=dt)
            torch.save(art, p + ".tmp"); os.replace(p + ".tmp", p)
            print(f"ref L{L} E{E} {dt:.1f}s", flush=True)
    elif a.cmd == "batch":
        C.setup()
        import nq_encode_batch as NB
        import collections
        tm = collections.defaultdict(float)
        if os.environ.get("NQ23_PROF") == "1":           # sync-timed Viterbi (distorts overlap; diagnostics only)
            import nq_encode as NE
            _v = NE.viterbi
            def vit(r, K):
                torch.cuda.synchronize(); t0 = time.time(); o = _v(r, K); torch.cuda.synchronize()
                tm[f"viterbi K{K}"] += time.time() - t0; tm[f"rings K{K}"] += r.shape[0]; return o
            NE.viterbi = vit
        t = time.time()
        for L, E, art in NB.encode_experts(ex, group=a.group, stats=tm):
            p = f"{C.SCR}/{a.tag}/L{L}_E{E}.pt"
            art["_batch"] = dict(shas=C.IMPORT_SHAS, dir=C.T12)
            torch.save(art, p + ".tmp"); os.replace(p + ".tmp", p)
            print(f"batch L{L} E{E} done at {time.time()-t:.1f}s", flush=True)
        print(f"batch total {time.time()-t:.1f}s for {len(ex)} experts = {(time.time()-t)/len(ex):.2f} s/expert, "
              f"max mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tm.items()}, indent=1), flush=True)
    elif a.cmd == "pair":                                   # same process, back to back: ref then batch, then compare
        C.setup()
        import nq_encode_batch as NB
        import collections
        refs, tr = {}, []
        for L, E in ex:
            torch.cuda.synchronize(); t = time.time()
            refs[L, E] = C.ref_encode(L, E)
            torch.cuda.synchronize(); tr.append(time.time() - t)
            print(f"ref L{L} E{E} {tr[-1]:.1f}s", flush=True)
        tm = collections.defaultdict(float)
        torch.cuda.reset_peak_memory_stats()
        t = time.time(); bad_all = 0
        for L, E, art in NB.encode_experts(ex, group=a.group, stats=tm):
            nt, nb, bad = C.compare(refs.pop((L, E)), art)
            bad_all += len(bad)
            print(f"L{L:<2} E{E:<3} tensors {nt} {'IDENTICAL' if not bad else 'MISMATCH ' + str(bad[:6])}", flush=True)
        tb = time.time() - t
        tr_s = sorted(tr)
        print(f"PAIR ref median {tr_s[len(tr_s)//2]:.2f} s/expert (mean {sum(tr)/len(tr):.2f}) | batch group {a.group}: "
              f"{tb/len(ex):.2f} s/expert amortised, max mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB | "
              f"speedup {sum(tr)/tb:.2f}x | {'ALL BIT-IDENTICAL' if not bad_all else 'FAIL'} | T12 {C.IMPORT_SHAS}", flush=True)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tm.items()}), flush=True)
        sys.exit(0 if not bad_all else 1)
    else:
        live = C.ref_shas(C.T12_LIVE)
        ok_all = True
        for L, E in ex:
            r = torch.load(f"{C.SCR}/{a.ref_tag}/L{L}_E{E}.pt", weights_only=False)
            b = torch.load(f"{C.SCR}/{a.tag}/L{L}_E{E}.pt", weights_only=False)
            rs = r.pop("_ref")["shas"]; bs = b.pop("_batch", {}).get("shas")
            if bs is not None and bs != rs:
                print(f"L{L} E{E}: ref and batch arms imported DIFFERENT thread-12 code", flush=True); ok_all = False
            stale = rs != live
            nt, nb, bad = C.compare(r, b)
            ok = not bad
            ok_all &= ok
            print(f"L{L:<2} E{E:<3} tensors {nt} bytes {nb/2**20:.1f} MiB  {'IDENTICAL' if not bad else 'MISMATCH'}"
                  f"{'  (vs pinned T12, not live)' if stale else ''}", flush=True)
            for x in bad[:12]:
                print("    ", x)
        print(("ALL BIT-IDENTICAL" if ok_all else "FAIL") + f"  (thread-12 code: {rs})", flush=True)
        sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
