"""nq-tfpred decode-trace capture (rank 0 only, NQ_TFCAP=<out dir>; off when unset, and nq_vllm.py never imports it then).

Per decode row (every row of a model step with <= CAP_MAXT padded tokens, i.e. MTP verify rows incl. rejected drafts and
cudagraph padding; dedupe offline by (request, position), keep last), written into a device ring inside the CUDA graph:
  ids  uint8  [75, 8]   topk expert ids per MoE layer (L3..L77)
  w    fp16   [75, 8]   topk gate weights as passed to the MoE (before routed_scaling_factor)
  xn   fp32   [75]      sum(x^2) of the MoE input (fp32), so salience = (rsf*w)^2 * xn
  hp   fp16   [P, 256]  fixed random projection (seed 1234, N(0,1/H)) of the MoE input at layers CAP_PLAYERS
  tok, pos int32        input token id / position (written by the runner hook, outside the graph)
The runner hook (GPUModelRunner._model_forward) sets the row indices for the step before the forward (ring offset +
arange(Tp)) and records an event after it; a drain thread copies completed rows to host and writes chunk npz files.
Prefill steps (Tp > CAP_MAXT) write no rows; they log the request's per-layer expert counts instead (pf, int32 [75,256])."""
import os, time, threading, logging, hashlib, json
import numpy as np, torch
try:
    from vllm.logger import init_logger; log = init_logger('vllm.nestquant.tfcap')
except Exception:
    log = logging.getLogger('nestquant.tfcap')

CAP_MAXT = 16
RING = int(os.environ.get('NQ_TFCAP_RING', '16384'))
PLAYERS = [int(v) for v in os.environ.get('NQ_TFCAP_PLAYERS', '3,13,23,33,43,53,63,77').split(',')]
PD = 256
LOGITS = os.environ.get('NQ_TFCAP_LOGITS', '0') == '1'   # step3: full router logits [NL, 256] fp16 per row (raw gate output, pre-sigmoid/bias)
CHUNK = int(os.environ.get('NQ_TFCAP_CHUNK', '65536'))


class Capture:
    def __init__(s, rt, layers, H, dev):
        s.rt, s.dev, s.out = rt, dev, os.environ['NQ_TFCAP']
        os.makedirs(s.out, exist_ok=True)
        s.layers = list(layers); s.li = {L: i for i, L in enumerate(s.layers)}; NL = len(s.layers)
        s.pl = {L: k for k, L in enumerate([L for L in PLAYERS if L in s.li])}
        g = torch.Generator().manual_seed(1234)
        s.R = (torch.randn(len(s.pl), H, PD, generator=g) / H ** 0.5).to(dev, torch.bfloat16)
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=dev)       # noqa: E731
        s.r_ids = z((RING, NL, 8), torch.uint8); s.r_w = z((RING, NL, 8), torch.float16)
        s.r_xn = z((RING, NL), torch.float32); s.r_hp = z((RING, len(s.pl), PD), torch.float16)
        s.r_tok = z((RING,), torch.int32); s.r_pos = z((RING,), torch.int32)
        s.keys = ('ids', 'w', 'xn', 'hp', 'tok', 'pos') + (('lg',) if LOGITS else ())
        if LOGITS:
            import nq_lookahead as LAH
            s.r_lg = z((RING, NL, 256), torch.float16); s.gate = {L: LAH._run[L].gate for L in s.layers if L in LAH._run}
            miss = [L for L in s.layers if L not in s.gate]
            if miss: log.warning('tfcap: no router for layers %s (logits stay 0)', miss[:8])
            s.save_bias()
        s.idx = z((CAP_MAXT,), torch.long); s.ar = torch.arange(CAP_MAXT, device=dev)
        s.pf = z((NL, 256), torch.int32)
        s.Tp = 0; s.head = 0                       # host: rows enqueued so far (monotone)
        s.pend = []; s.lock = threading.Lock()     # [(event, row_end, step record)]
        s.steps = []; s.rows = []; s.nchunk = 0; s.boot = time.strftime('%Y%m%dT%H%M%S'); s.done = 0; s.lost = 0
        s.reqmap = {}; s.pfrec = []
        s.stream = torch.cuda.Stream(dev)
        s.mb = (s.r_ids[0].numel() + 2 * s.r_w[0].numel() + 4 * s.r_xn[0].numel() + 2 * s.r_hp[0].numel() + 8 + (2 * s.r_lg[0].numel() if LOGITS else 0)) * RING / 2 ** 20
        log.info('tfcap: ring %d rows (%.0f MiB), proj layers %s, out %s', RING, s.mb, list(s.pl), s.out)
        s.thread = threading.Thread(target=s.drain_loop, name='nq-tfcap', daemon=True); s.thread.start()

    def save_bias(s):
        """step3: per-layer e_score_correction_bias (if the router exposes it) -> <out>/router_bias.npz, once per boot"""
        try:
            import nq_lookahead as LAH
            B = {}
            for L in s.layers:
                r = LAH._run.get(L)
                for o in (r, getattr(r, 'router', None), getattr(r, 'gate', None)):
                    b = getattr(o, 'e_score_correction_bias', None) if o is not None else None
                    if b is not None:
                        B[f'L{L}'] = b.detach().float().cpu().numpy(); break
            np.savez(f'{s.out}/router_bias-{s.boot}.npz', **B)
            log.info('tfcap: router bias for %d layers saved', len(B))
        except Exception:
            log.exception('tfcap: router bias save failed')

    # ---------------------------------------------------------------- in the MoE forward (inside graphs for decode)
    def layer(s, L, x, w, ids):
        T = x.shape[0]
        if T > CAP_MAXT:
            if not torch.cuda.is_current_stream_capturing():
                s.pf[s.li[L]] += torch.bincount(ids.reshape(-1).long(), minlength=256).int()
            return
        i = s.li[L]; idx = s.idx[:T]
        s.r_ids[:, i].index_copy_(0, idx, ids.to(torch.uint8))
        s.r_w[:, i].index_copy_(0, idx, w.to(torch.float16))
        s.r_xn[:, i].index_copy_(0, idx, x.float().square().sum(-1))
        k = s.pl.get(L)
        if k is not None:
            s.r_hp[:, k].index_copy_(0, idx, (x.to(torch.bfloat16) @ s.R[k]).to(torch.float16))
        if LOGITS and L in s.gate:
            lg, _ = s.gate[L](x)
            s.r_lg[:, i].index_copy_(0, idx, lg.to(torch.float16))

    # ---------------------------------------------------------------- runner hook (eager, every model step)
    def pre(s, runner, input_ids, positions):
        Tp = int(input_ids.shape[0]) if input_ids is not None else int(positions.shape[-1])
        s.Tp = Tp
        if Tp > CAP_MAXT:
            return
        h = s.head % RING
        s.idx[:Tp].copy_((s.ar[:Tp] + h) % RING)
        idx = s.idx[:Tp]
        if input_ids is not None:
            s.r_tok.index_copy_(0, idx, input_ids.to(torch.int32))
        p = positions[0] if positions.dim() == 2 else positions
        s.r_pos.index_copy_(0, idx, p[:Tp].to(torch.int32))

    def post(s, runner, rid=None, prefilling=False):
        Tp = s.Tp
        if rid is None:
            try:
                rid = runner.input_batch.req_ids[0] if runner.input_batch.req_ids else ''
            except Exception:
                rid = ''
        q = s.reqmap.get(rid)
        if q is None:
            q = s.reqmap[rid] = len(s.reqmap)
        if Tp > CAP_MAXT:                       # prefill step: per-layer expert counts, copied in stream order
            h = torch.empty(s.pf.shape, dtype=torch.int32).pin_memory()
            h.copy_(s.pf, non_blocking=True); s.pf.zero_()
            ev = torch.cuda.Event(); ev.record()
            with s.lock:
                s.pend.append((ev, None, (q, Tp, time.time(), s.head, h)))
            return
        s.head += Tp
        ev = torch.cuda.Event(); ev.record()
        with s.lock:
            s.pend.append((ev, s.head, (q, Tp if not prefilling else -Tp, time.time())))

    # ---------------------------------------------------------------- drain (own thread, own stream)
    def drain_loop(s):
        torch.cuda.set_device(s.dev)
        s.tail = 0
        while True:
            time.sleep(0.25)
            try:
                if s.rt.ncap > 0:
                    continue
                done = []
                with s.lock:
                    while s.pend and s.pend[0][0].query():
                        ev, end, rec = s.pend.pop(0)
                        if end is None:
                            q, Tp, wl, head, h = rec
                            s.pfrec.append((q, Tp, wl, head, h.numpy().copy()))
                        else:
                            done.append((end, rec))
                if done:
                    end = done[-1][0]; n = end - s.tail; t0 = s.tail
                    if n > RING:
                        s.lost += n - RING; log.warning('tfcap: drain fell behind, %d rows lost', n - RING); t0 = end - RING; n = RING
                    a = t0 % RING
                    ixn = np.concatenate([np.r_[a:min(a + n, RING)], np.r_[0:max(0, a + n - RING)]])
                    with torch.cuda.stream(s.stream):
                        ix = torch.from_numpy(ixn).to(s.dev)
                        blk = {k: getattr(s, 'r_' + k).index_select(0, ix).cpu().numpy() for k in s.keys}
                    blk['abs'] = np.arange(t0, end, dtype=np.int64)
                    s.rows.append(blk)
                    for e2, (q, Tp, wl) in done:
                        s.steps.append((q, Tp, wl, e2))
                    s.tail = end; s.done += n
                s.maybe_flush()
            except Exception:
                log.exception('tfcap drain error'); time.sleep(5)

    def maybe_flush(s, force=False):
        if not s.rows and not s.pfrec:
            s.t_first = time.time(); return
        nrow = sum(len(b['tok']) for b in s.rows)
        if nrow < CHUNK and not force and time.time() - getattr(s, 't_first', time.time()) < 900:
            return
        K = s.keys + ('abs',)
        cat = {k: (np.concatenate([b[k] for b in s.rows]) if s.rows else np.zeros((0,) + getattr(s, 'r_' + k, s.r_tok).shape[1:])) for k in K}
        st = np.array([(q, t, e) for q, t, _, e in s.steps], dtype=np.int64).reshape(-1, 3)
        sw = np.array([w for _, _, w, _ in s.steps], dtype=np.float64)
        pfi = np.array([(q, t, hd) for q, t, _, hd, _ in s.pfrec], dtype=np.int64).reshape(-1, 3)
        pfw = np.array([w for _, _, w, _, _ in s.pfrec], dtype=np.float64)
        pfc = np.stack([c for *_, c in s.pfrec]) if s.pfrec else np.zeros((0, len(s.layers), 256), np.int32)
        f = f'{s.out}/cap-{s.boot}-{s.nchunk:05d}.npz'
        np.savez(f + '.part.npz', **cat, step_req=st[:, 0], step_T=st[:, 1], step_end=st[:, 2], step_wall=sw,
                 pf_req=pfi[:, 0], pf_T=pfi[:, 1], pf_head=pfi[:, 2], pf_wall=pfw, pf_counts=pfc,
                 layers=np.array(s.layers), players=np.array(list(s.pl)), lost=np.array(s.lost),
                 req_names=np.array(json.dumps({str(v): k for k, v in s.reqmap.items()})))
        os.replace(f + '.part.npz', f)
        log.info('tfcap: wrote %s (%d rows, %d steps, %d prefill recs, lost %d)', f, len(cat['tok']), len(st), len(s.pfrec), s.lost)
        s.rows = []; s.steps = []; s.pfrec = []; s.nchunk += 1; s.t_first = time.time()


def install_v2(rt):
    """V2 model runner (vllm/v1/worker/gpu/model_runner.py, the one this image uses): pre after prepare_inputs (inside
    execute_model, so the tok/pos copies are stream-ordered before the forward), post after execute_model.  Steps with a
    prefilling request are flagged (step_T < 0)."""
    from vllm.v1.worker.gpu import model_runner as MR2
    R = MR2.GPUModelRunner
    if getattr(R, '_nq_tfcap', False):
        return
    p0, e0 = R.prepare_inputs, R.execute_model

    def prepare_inputs(self, *a, **k):
        ib = p0(self, *a, **k)
        c = rt.CAP
        if c is not None and getattr(self, '_nq_tf_in_exec', False) and not torch.cuda.is_current_stream_capturing():
            try:
                c.pre(self, ib.input_ids, ib.positions)
                self._nq_tf_ib = ib
            except Exception:
                log.exception('tfcap pre failed, capture off'); rt.CAP = None
        return ib

    def execute_model(self, *a, **k):
        self._nq_tf_in_exec = True; self._nq_tf_ib = None
        try:
            out = e0(self, *a, **k)
        finally:
            self._nq_tf_in_exec = False
        ib = self._nq_tf_ib; self._nq_tf_ib = None
        c = rt.CAP
        if ib is not None and c is not None:
            try:
                c.post(self, rid=ib.req_ids[0] if ib.req_ids else '', prefilling=bool(np.any(ib.is_prefilling_np)))
            except Exception:
                log.exception('tfcap post failed, capture off'); rt.CAP = None
        return out
    R.prepare_inputs, R.execute_model, R._nq_tfcap = prepare_inputs, execute_model, True
    log.info('tfcap: hooked V2 model runner')


def install(rt, layers, H, dev):
    """rank 0: create the capture and hook the model runner (V2 if importable, plus V1 GPUModelRunner._model_forward)."""
    from vllm.v1.worker import gpu_model_runner as GMR
    cap = Capture(rt, layers, H, dev)
    try:
        install_v2(rt)
    except Exception:
        log.exception('tfcap: V2 runner hook failed')
    R = GMR.GPUModelRunner
    if not getattr(R, '_nq_tfcap', False):
        f0 = R._model_forward

        def _model_forward(self, input_ids=None, positions=None, *a, **k):
            c = rt.CAP
            live = c is not None and not torch.cuda.is_current_stream_capturing() and positions is not None
            if live:
                try:
                    c.pre(self, input_ids, positions)
                except Exception:
                    log.exception('tfcap pre failed, capture off'); rt.CAP = None; live = False
            out = f0(self, input_ids, positions, *a, **k)
            if live and rt.CAP is not None:
                try:
                    c.post(self)
                except Exception:
                    log.exception('tfcap post failed, capture off'); rt.CAP = None
            return out
        R._model_forward = _model_forward; R._nq_tfcap = True
    return cap
