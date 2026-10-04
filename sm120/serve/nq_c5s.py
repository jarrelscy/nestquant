"""step 3b c5s live admission (rank 0 only, NQ_C5S=<C3k ckpt>, e.g. /nq/streaming/ckpt/C3k.pt; nq_vllm.py never imports
this when unset). Needs NQ_SCHED=tap + NQ_PREDICTOR=joint; composes with NQ_KVEC; replaces NQ_TFCAP (both use RT.CAP).

Row source = the tfcap device ring (nq_tfcap.Capture: ids / w / xn / hp written by index_copy inside the decode graph,
tok / pos by the runner hook), without logits or files; a drain thread (every NQ_C5S_POLL_MS, default 10) copies the
finished steps' rows to host and feeds streaming/c5s.py: row finalizer (one row per (request, position), keep last, pos 0
and prefilling steps dropped; prefill steps give the request's pf) -> incremental p-seq features -> C3k forward on the
newest ready refresh -> published mC. TapScheduler (scheduler_tap.py, s.c5) blends the latest mC with the current jF S
at every tap refresh; it never waits: a late forward reuses the previous mC, and without a usable mC (new request, or
older than NQ_C5S_MAXLAG rows) the refresh uses plain jF exactly like prod.

Env: NQ_C5S (ckpt), NQ_C5S_THREADS (torch intra-op threads for the CPU forward, default 4; 0 = leave as is),
NQ_C5S_DEV (cpu | cuda = this rank's GPU on a side stream, untested), NQ_C5S_MAXLAG (rows, default 64), NQ_C5S_RING
(device ring rows, default 4096), NQ_C5S_POLL_MS (default 10), NQ_C5S_LOG_S (stats log period, default 60)."""
import os, time, threading, logging
import numpy as np, torch
import nq_tfcap as TC
try:
    from vllm.logger import init_logger; log = init_logger('vllm.nestquant.c5s')
except Exception:
    log = logging.getLogger('nestquant.c5s')


class C5Capture(TC.Capture):
    """tfcap ring without logits / file output; drained into c5s.C5S"""
    def __init__(s, rt, layers, H, dev, P):
        s.rt, s.dev, s.P = rt, dev, P
        s.layers = list(layers); s.li = {L: i for i, L in enumerate(s.layers)}; NL = len(s.layers)
        s.pl = {L: k for k, L in enumerate([L for L in TC.PLAYERS if L in s.li])}
        assert NL == 75 and len(s.pl) == 8, f'c5s needs 75 MoE layers and 8 hp layers here (got {NL}, {list(s.pl)})'
        g = torch.Generator().manual_seed(1234)            # same projection as the tfcap traces C3k was trained on
        s.R = (torch.randn(len(s.pl), H, TC.PD, generator=g) / H ** 0.5).to(dev, torch.bfloat16)
        s.RING = int(os.environ.get('NQ_C5S_RING') or 4096)
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=dev)       # noqa: E731
        s.r_ids = z((s.RING, NL, 8), torch.uint8); s.r_w = z((s.RING, NL, 8), torch.float16)
        s.r_xn = z((s.RING, NL), torch.float32); s.r_hp = z((s.RING, len(s.pl), TC.PD), torch.float16)
        s.r_tok = z((s.RING,), torch.int32); s.r_pos = z((s.RING,), torch.int32)
        s.keys = ('ids', 'w', 'xn', 'hp', 'tok', 'pos')
        s.gate = {}                                         # no router logits (Capture.layer checks LOGITS and L in s.gate)
        s.idx = z((TC.CAP_MAXT,), torch.long); s.ar = torch.arange(TC.CAP_MAXT, device=dev)
        s.pf = z((NL, 256), torch.int32)
        s.Tp = 0; s.head = 0; s.pend = []; s.lock = threading.Lock(); s.reqmap = {}; s.lost = 0
        s.poll = float(os.environ.get('NQ_C5S_POLL_MS') or 10) / 1e3; s.logp = float(os.environ.get('NQ_C5S_LOG_S') or 60)
        s.stream = torch.cuda.Stream(dev)
        mb = (s.r_ids[0].numel() + 2 * s.r_w[0].numel() + 4 * s.r_xn[0].numel() + 2 * s.r_hp[0].numel() + 8) * s.RING / 2 ** 20
        log.info('c5s: ring %d rows (%.0f MiB), hp layers %s, ckpt %s, dev %s, threads %d, maxlag %d', s.RING, mb, list(s.pl),
                 os.environ.get('NQ_C5S'), P.dev, P.threads, P.maxlag)
        s.thread = threading.Thread(target=s.drain_loop, name='nq-c5s', daemon=True); s.thread.start()

    # tfcap's RING is a module constant; the hooks index with it
    def pre(s, runner, input_ids, positions):
        Tp = int(input_ids.shape[0]) if input_ids is not None else int(positions.shape[-1])
        s.Tp = Tp
        if Tp > TC.CAP_MAXT:
            return
        h = s.head % s.RING
        s.idx[:Tp].copy_((s.ar[:Tp] + h) % s.RING)
        idx = s.idx[:Tp]
        if input_ids is not None:
            s.r_tok.index_copy_(0, idx, input_ids.to(torch.int32))
        p = positions[0] if positions.dim() == 2 else positions
        s.r_pos.index_copy_(0, idx, p[:Tp].to(torch.int32))

    def drain_loop(s):
        torch.cuda.set_device(s.dev)
        if s.P.dev.type == 'cuda': s.P.stream = torch.cuda.Stream(s.P.dev)
        tail = 0; tlog = time.time(); bad = 0
        while True:
            time.sleep(s.poll)
            try:
                if s.rt.ncap > 0:
                    continue
                ev_ = []
                with s.lock:
                    while s.pend and s.pend[0][0].query():
                        ev_.append(s.pend.pop(0))
                if ev_:
                    ends = [e for _, e, _ in ev_ if e is not None]
                    blk = None
                    if ends:
                        end = ends[-1]; n = end - tail; t0 = tail
                        if n > s.RING:
                            s.lost += n - s.RING; log.warning('c5s: drain fell behind, %d rows lost', n - s.RING); t0 = end - s.RING; n = s.RING
                        a = t0 % s.RING
                        ixn = np.concatenate([np.r_[a:min(a + n, s.RING)], np.r_[0:max(0, a + n - s.RING)]])
                        with torch.cuda.stream(s.stream):
                            ix = torch.from_numpy(ixn).to(s.dev)
                            blk = {k: getattr(s, 'r_' + k).index_select(0, ix).cpu().numpy() for k in s.keys}
                        tail = end
                    for _, e, rec in ev_:                       # stream order: prefill counts before the request's decode rows
                        if e is None:
                            q, Tp, _, _, h = rec; s.P.R.prefill(q, Tp, h.numpy().astype(np.int64)); continue
                        q, Tp, _ = rec; T = abs(Tp); lo = e - T - t0
                        if lo < 0: continue                     # rows lost
                        r = slice(lo, lo + T)
                        s.P.R.step(q, Tp < 0, blk['pos'][r], blk['tok'][r], blk['ids'][r], blk['w'][r], blk['xn'][r], blk['hp'][r])
                    s.P.run_pending()
                if s.logp > 0 and time.time() - tlog > s.logp:
                    tlog = time.time(); st = s.P.st; nf = max(1, st['fwd'])
                    log.info('c5s: rows %d refreshes %d fwd %d (skipped %d) | feat %.2f ms fwd %.2f ms /refresh, blk %.2f ms | '
                             'tap used %d fallback %d | lost %d', st['rows'], st['refresh'], st['fwd'], st['skipped'],
                             1e3 * st['t_feat'] / nf, 1e3 * st['t_fwd'] / nf, 1e3 * st['t_blk'] / max(1, st['refresh']),
                             st['used'], st['fallback'], s.lost)
            except Exception:
                bad += 1; log.exception('c5s drain error (%d)', bad); time.sleep(5)


def install(rt, layers, H, dev):
    """rank 0 (leader): load C3k, create the capture (hooks via nq_tfcap.install), return it (-> RT.CAP); rt.S.c5 = predictor"""
    import c5s as C5
    e = lambda k, d: os.environ.get(k) or d       # noqa: E731  (empty = default)
    pdev = e('NQ_C5S_DEV', 'cpu'); pdev = dev if pdev == 'cuda' else pdev
    P = C5.C5S(e('NQ_C5S', ''), dev=pdev, threads=int(e('NQ_C5S_THREADS', '4')), maxlag=int(e('NQ_C5S_MAXLAG', '64')))
    cap = TC.install(rt, layers, H, dev, cls=lambda rt_, L_, H_, d_: C5Capture(rt_, L_, H_, d_, P))
    rt.S.c5 = P
    log.info('c5s: tap admission = c5s(jF, C3k) blend, fallback jF')
    return cap
