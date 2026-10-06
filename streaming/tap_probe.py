"""nq-tapad boot probe: time random record reads on this rank's drive(s) before serving -> starting read rate for the tap
scheduler (TapScheduler.probe_rate). Plain O_DIRECT preads (the engine reads O_DIRECT through io_uring; on DGX Spark
unified mode straight into host-mapped slots, same drive path) from qd threads per drive, so the drive sees the
engine's queue depth; with a second drive copy (nq-io dual path) both drives run at once with their own qd and reads
split by qd (the engine sends each read to the drive with fewer in flight, so the rank rate = the sum). CPU only, no GPU.
  probe(path, rec_bytes, n=256, qd=8, alt=None, qd_alt=0, max_s=3.0, seed=0)
    -> dict(GBps, n, bytes, s, p50_ms, p99_ms, direct, drives=[dict(path, GBps, n)])
  n reads in total (stops early at max_s: GBps is then over the reads done); GBps = bytes / wall time.
  python streaming/tap_probe.py <rank.bin> <rec_bytes> [n] [qd] [alt.bin qd_alt]"""
import os, sys, time, mmap, threading
import numpy as np


def _drive(path, rb, n, qd, deadline, rng, out):
    try: fd = os.open(path, os.O_RDONLY | os.O_DIRECT); dio = True
    except OSError: fd = os.open(path, os.O_RDONLY); dio = False          # no O_DIRECT here (tmpfs, tests): buffered
    nrec = os.fstat(fd).st_size // rb; recs = rng.integers(0, max(nrec, 1), n); lat = []; nb = [0]; lk = threading.Lock(); k = [0]
    def run():
        buf = mmap.mmap(-1, rb)                                            # page-aligned, as O_DIRECT needs
        while time.monotonic() < deadline:
            with lk:
                if k[0] >= n: return
                j = k[0]; k[0] += 1
            t = time.monotonic()
            try: r = os.preadv(fd, [buf], int(recs[j]) * rb)
            except OSError: r = -1
            dt = time.monotonic() - t
            with lk:
                if r == rb: nb[0] += rb; lat.append(dt)
    t0 = time.monotonic(); th = [threading.Thread(target=run, daemon=True) for _ in range(max(1, qd))]
    for x in th: x.start()
    for x in th: x.join()
    w = time.monotonic() - t0; os.close(fd)
    out.append(dict(path=path, n=len(lat), bytes=nb[0], s=w, GBps=nb[0] / max(w, 1e-9) / 1e9, lat=lat, direct=dio))


def probe(path, rb, n=256, qd=8, alt=None, qd_alt=0, max_s=3.0, seed=0):
    rng = np.random.default_rng(seed); dv = [(path, qd)] + ([(alt, qd_alt or qd)] if alt else [])
    qt = sum(q for _, q in dv); ns = [max(1, n * q // qt) for _, q in dv]
    out = []; t0 = time.monotonic(); dl = t0 + max_s
    th = [threading.Thread(target=_drive, args=(p, rb, m, q, dl, np.random.default_rng(rng.integers(1 << 62)), out)) for (p, q), m in zip(dv, ns)]
    for x in th: x.start()
    for x in th: x.join()
    w = time.monotonic() - t0; nb = sum(d['bytes'] for d in out); lat = np.array([x for d in out for x in d['lat']] or [np.nan])
    return dict(GBps=nb / max(w, 1e-9) / 1e9, n=sum(d['n'] for d in out), bytes=nb, s=w, p50_ms=float(np.median(lat)) * 1e3,
                p99_ms=float(np.percentile(lat, 99)) * 1e3, direct=all(d['direct'] for d in out),
                drives=[dict(path=d['path'], GBps=d['GBps'], n=d['n']) for d in out])


if __name__ == '__main__':
    a = sys.argv[1:]
    r = probe(a[0], int(a[1]), n=int(a[2]) if len(a) > 2 else 256, qd=int(a[3]) if len(a) > 3 else 8,
              alt=a[4] if len(a) > 4 else None, qd_alt=int(a[5]) if len(a) > 5 else 0)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
