"""Thread 23: batched drop-in for thread 12's nq_layer.py (same CLI + --group G; same expert files, TP shards, manifest).

  python nq_layer_batch.py --layer L --out /tmp/nestquant/nq-encode [--experts 0:256] [--group 4] [nq_layer options]

The per-expert loop of nq_layer.main (H/G via nq_bnd.glm_H_bnd, unrouted-expert fallbacks, teacher, res_K parse,
meta fields) is mirrored here with the encode itself done G experts at a time by nq_encode_batch.encode_group; each
expert is written to OUT/L{L}/experts/E{E}.pt exactly as nq_layer writes it. Finalize (TP shard files + manifest)
is thread 12's own code: nq_layer.main() is then run with the same arguments (it skips the existing experts).
Resumable like nq_layer: finished experts are skipped; a killed group is re-encoded (group-composition invariant).
Optional background prefetch of the next group's teacher + H/G (--prefetch, off by default).
"""
import os, sys, time, argparse, threading, queue
os.environ.setdefault("OMP_NUM_THREADS", "16")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                   # sets up the thread-12 import path (NQ23_T12 pin or live)
import torch
import nq_encode as NE
import nq_bnd as NBND
import nq_layer as NL
import nq_encode_batch as NBAT


def parse():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--stats", default="/tmp/nestquant/19-capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", default="0:256")
    ap.add_argument("--rate", type=float)
    ap.add_argument("--source", default=os.environ.get("NQ19_SRC", "/tmp/nestquant/src/glm53-fp8"))
    ap.add_argument("--res-k")
    ap.add_argument("--bnd", default=str(NBND.DEFAULT_BND))
    ap.add_argument("--bnd-k", type=float, default=NBND.DEFAULT_K)
    ap.add_argument("--bnd-cap", type=float, default=NBND.DEFAULT_CAP)
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--stats-mm"); ap.add_argument("--mm-w", type=float, default=getattr(NL, "MM_W", 0.25))
    ap.add_argument("--lr-tau", type=float, default=NE.LR["tau"] if hasattr(NE, "LR") else None)
    ap.add_argument("--lr-rmax", type=int, default=NE.LR["rmax"] if hasattr(NE, "LR") else None)
    ap.add_argument("--no-lr", action="store_true")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--prefetch", action="store_true", help="load next group in a thread (off by default)")
    a, _ = ap.parse_known_args()
    return a


def load_one(cap, a, L, E):
    """nq_layer.main loop body up to the encode call (thread 12's own expert_HG)."""
    HG, flags = NL.expert_HG(cap, L, E, a.bnd, a.bnd_k, a.bnd_cap)
    return NL.teacher(a.source, L, E), HG, flags


def main():
    a = parse()
    C.setup()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    cap = NL.open_stats(a.stats, a.stats_mm, a.mm_w)
    lrc = None if a.no_lr else dict(NE.LR, tau=a.lr_tau, rmax=a.lr_rmax)
    L = a.layer
    bw = NBND.parse_bnd(a.bnd)
    d = f"{a.out}/L{L}"; ed = f"{d}/experts"
    os.makedirs(ed, exist_ok=True)
    e0, e1 = map(int, a.experts.split(":"))
    todo = [E for E in range(e0, e1) if not os.path.exists(f"{ed}/E{E}.pt")]
    groups = [todo[i:i + a.group] for i in range(0, len(todo), a.group)]
    rk = dict(zip(NE.PROJ, map(float, a.res_k.split(",")))) if a.res_k else None
    stream = torch.cuda.Stream()

    def load_group(g):
        with torch.cuda.stream(stream):
            out = [load_one(cap, a, L, E) for E in g]
        stream.synchronize()
        return out
    q = queue.Queue(maxsize=1)

    def producer():
        for g in groups:
            q.put((g, load_group(g)))
        q.put(None)
    if a.prefetch:
        threading.Thread(target=producer, daemon=True).start()
    seg = NBAT.Seg(True)
    t_all = time.time(); n_done = 0
    with NBAT.patched(NBAT.DEFAULT_OPTS):
        for gi in range(len(groups)):
            t0 = time.time()
            if not a.prefetch:
                g, data = groups[gi], load_group(groups[gi])
            else:
                g, data = q.get()
            torch.cuda.current_stream().wait_stream(stream)
            arts = NBAT.encode_group([(W, HG) for W, HG, _ in data], rate=a.rate, res_K=rk, seg=seg, lr=lrc)
            for E, (_, HG, flags), art in zip(g, data, arts):
                art["meta"].update(layer=L, expert=E, flags=flags, bnd=NBND.bnd_tag(bw), bnd_k=a.bnd_k, bnd_cap=a.bnd_cap,
                                   stats=a.stats, stats_mm=a.stats_mm, mm_w=a.mm_w if a.stats_mm else 0.0,
                                   hg_meta={k: (float(v) if torch.is_tensor(v) else v) for k, v in HG.get("meta", {}).items()
                                            if not isinstance(v, dict)})
                path = f"{ed}/E{E}.pt"
                torch.save(art, path + ".tmp"); os.replace(path + ".tmp", path)
            n_done += len(g)
            print(f"[L{L} E{g[0]}..E{g[-1]}] {time.time()-t0:.0f}s  {(time.time()-t_all)/n_done:.2f} s/expert amortised",
                  flush=True)
            del data, arts
            torch.cuda.empty_cache()
    if a.no_finalize:
        return
    argv = [a for a in sys.argv[1:]]
    for flag in ("--group",):                                   # T23-only options
        while flag in argv:
            i = argv.index(flag); del argv[i:i + 2]
    argv = [x for x in argv if x != "--prefetch"]
    sys.argv = [os.path.join(C.T12, "nq_layer.py")] + argv
    NL.main()                                                   # thread 12 finalize (all experts present -> skip)


if __name__ == "__main__":
    main()
