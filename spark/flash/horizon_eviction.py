"""Causal replacement utility from all future jMT heads; no weight/state mutation."""
import math
import numpy as np

def replacement_scores(forecasts,ema,mix,rate,pending,tps,protect_k=16):
    f=np.asarray(forecasts,dtype=np.float64)
    if f.ndim!=3 or f.shape[0]!=16 or f.shape[2]!=288 or not np.isfinite(f).all() or np.any(f<0):
        raise ValueError('Invalid sixteen-head jMT shares')
    if not np.allclose(f.sum(-1),1,rtol=.003,atol=.003):raise ValueError('jMT heads must be normalized separately')
    ema=np.asarray(ema,dtype=np.float64)
    if ema.shape!=f.shape[1:] or not np.isfinite(ema).all() or np.any(ema<0):raise ValueError('Invalid EMA')
    if not 0<=mix<=1 or not 0<=protect_k<=288:raise ValueError('Invalid horizon policy parameters')
    if not rate or rate<=0:return None
    # Conservative queue-service estimate. Forecast origin is the last committed
    # input n. Head index0 predicts n+1; index15 predicts n+16.
    arrival=max(1,math.ceil((max(0,pending)+1)/rate*max(1,tps)))
    weighted=mix*f+(1-mix)*(ema/np.maximum(ema.sum(-1,keepdims=True),1e-30))[None]
    cost=weighted.sum(0)  # lose resident's benefit throughout the forecast window
    gain=weighted[arrival-1:].sum(0) if arrival<=16 else np.zeros_like(cost)
    near=f[:min(arrival,16)].sum(0)
    protect=np.zeros_like(cost,dtype=bool)
    if protect_k:
        order=np.argsort(-near,axis=-1,kind='stable')[:,:protect_k]
        np.put_along_axis(protect,order,True,axis=-1)
    return dict(gain=gain,cost=cost,protect=protect,arrival=arrival)
