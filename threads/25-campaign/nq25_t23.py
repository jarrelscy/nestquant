"""Thread 25 adapter: T23's batched encoder behind nq_layer's per-expert file contract.

  python nq25_t23.py --layer L --experts a:b --stats SHIM --out ROOT [--source SRC] [--group 4] [--stats-mm M --mm-w w]

Writes ROOT/L{L}/experts/E{E}.pt (atomic tmp + replace, skip existing) with the artifact T23's encode_experts yields,
plus the meta keys nq_layer adds (bnd tag / bnd_k / bnd_cap / hg_meta), and prints nq_layer's progress line
"[L{L} E{E}] {s}s ..." (s = group wall / group size). T23 / T12 files are used unmodified; only T23's module-level
input locations (common.ROOT / STATS / SRC) are pointed at the campaign's pinned stats shim, and common.load_HG is
wrapped to keep each expert's HG meta for the artifact.
"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
T23 = "/home/coder/git/nestquant/threads/23-encode-throughput"
sys.path.insert(0, T23)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", default="0:256")
    ap.add_argument("--stats", required=True, help="capture root; its stats/ dir is used (campaign shim)")
    ap.add_argument("--stats-version", default="stats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="/tmp/nestquant/src/glm53-fp8")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--stats-mm", help="T26 vision root: H/G from T12's nq_layer.expert_HG(open_stats(stats, mm, w)) "
                                       "(the reference encoder's exact blend) instead of T23's text-only load_HG")
    ap.add_argument("--mm-w", type=float, default=0.25)
    a = ap.parse_args()
    import common as C
    C.ROOT, C.STATS, C.SRC = a.stats, a.stats_version, a.source
    C._CAP = None
    import torch
    import nq_encode_batch as B
    import nq_bnd as NB
    C.setup()
    hgm = {}
    orig = C.load_HG
    if a.stats_mm:
        import nq_layer as NL
        bcap = NL.open_stats(a.stats, a.stats_mm, a.mm_w)
        orig = lambda L, E, device="cuda": NL.expert_HG(bcap, L, E)

    def load_HG(L, E, device="cuda"):
        HG, flags = orig(L, E, device=device)
        hgm[(L, E)] = {k: (float(v) if torch.is_tensor(v) else v) for k, v in HG.get("meta", {}).items()
                       if not isinstance(v, dict)}
        return HG, flags
    C.load_HG = load_HG
    L = a.layer
    ed = f"{a.out}/L{L}/experts"
    os.makedirs(ed, exist_ok=True)
    e0, e1 = map(int, a.experts.split(":"))
    todo = [(L, E) for E in range(e0, e1) if not os.path.exists(f"{ed}/E{E}.pt")]
    bw = NB.parse_bnd(str(NB.DEFAULT_BND))
    t_last = time.time(); pend = []
    for i, (L_, E, art) in enumerate(B.encode_experts(todo, group=a.group)):
        art["meta"].update(layer=L_, expert=E, bnd=NB.bnd_tag(bw), bnd_k=NB.DEFAULT_K, bnd_cap=NB.DEFAULT_CAP,
                           hg_meta=dict(hgm.pop((L_, E), {}), bnd="none(w=1)"), encoder="t23-batch",
                           stats=a.stats, stats_mm=a.stats_mm, mm_w=a.mm_w if a.stats_mm else 0.0)
        p = f"{ed}/E{E}.pt"
        torch.save(art, p + ".tmp"); os.replace(p + ".tmp", p)
        pend.append((E, art["meta"]))
        if len(pend) == a.group or i == len(todo) - 1:        # one group finished
            dt = (time.time() - t_last) / len(pend); t_last = time.time()
            for E_, m in pend:
                print(f"[L{L_} E{E_}] {dt:.0f}s {m.get('flags')} "
                      f"{ {q: round(m['info'][q]['bits'][4], 4) for q in ('gate', 'up', 'down')} }", flush=True)
            pend = []


if __name__ == "__main__":
    main()
