"""NQ_S3_EXT example / test hooks (streaming/scheduler_tap.py TapExtCtx).
  replay: X = NQ_S3X_REPLAY .npy [refresh, 75, 256] at ctx.tok // 16 (an offline prediction file, e.g. p-online E1 preds)
  none:   returns None (hook cost floor)
  feats:  GBDT-like feature cost probe: per-expert counts over the last 64 / 256 / 1024 rows + last-use recency, returns
          the 64-row counts (not a model)"""
import os, numpy as np
_E = {}


def replay(ctx):
    p = os.environ['NQ_S3X_REPLAY']; E = _E.get(p)
    if E is None: E = _E[p] = np.load(p, mmap_mode='r')
    b = ctx.tok // 16
    return None if b >= len(E) else np.asarray(E[b], np.float32)


def none(ctx): return None


def feats(ctx):
    ids = ctx.ids; n = len(ids); NL = ids.shape[1]
    flat = (ids.astype(np.int32) + (np.arange(NL, dtype=np.int32) * 256)[None, :, None])
    out = []
    for w in (64, 256, 1024):
        out.append(np.bincount(flat[max(0, n - w):].ravel(), minlength=NL * 256).reshape(NL, 256))
    last = np.full(NL * 256, -1, np.int64); r = np.repeat(np.arange(n), NL * 8)
    last[flat.ravel()] = r                                  # fancy assignment: the last write wins = most recent row
    ctx.state['rec'] = n - 1 - last
    return out[0].astype(np.float32)
