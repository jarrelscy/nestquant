"""Retain/reuse the existing profile-only sentinel; no serving allocation."""
import json,os,pathlib,time
_active=False
_tensor=None
_runner=None
_ever=False
_events=[]
EXPECTED_BYTES=512*1024*1024

def record(event,**kw):
    _events.append(dict(event=event,time=time.time(),**kw))

def begin():
    global _active,_ever
    if _active or _ever:raise RuntimeError('Startup sentinel scope repeated')
    _active=True;_ever=True;record('begin')

def acquire(size,device):
    import torch
    global _tensor
    if not _active:return torch.empty(size,dtype=torch.uint8,device=device)
    if size!=EXPECTED_BYTES:raise RuntimeError('Unexpected startup profile sentinel size')
    if _tensor is None:
        _tensor=torch.empty(size,dtype=torch.uint8,device=device)
        record('allocated',bytes=size,pointer=_tensor.data_ptr(),device=str(device))
    else:
        if _tensor.device!=torch.device(device):raise RuntimeError('Sentinel device changed')
        record('reused',bytes=size,pointer=_tensor.data_ptr(),device=str(device))
    return _tensor

def mark_accounted(runner):
    global _runner
    if not _active or _tensor is None or _runner is not None:raise RuntimeError('Invalid sentinel accounting')
    _runner=runner;record('accounted',bytes=EXPECTED_BYTES)

def release():
    import torch
    global _active,_tensor,_runner
    if not _active:return
    torch.cuda.synchronize()
    if _runner is not None:
        if _runner.model_memory_usage<EXPECTED_BYTES:raise RuntimeError('Sentinel accounting underflow')
        _runner.model_memory_usage-=EXPECTED_BYTES
    record('released',bytes=EXPECTED_BYTES if _tensor is not None else 0,accounting_corrected=_runner is not None)
    _tensor=None;_runner=None;_active=False
    out=pathlib.Path('/data/kld-sampled-ahead-sentinel')
    out.mkdir(exist_ok=True)
    (out/f'sentinel-{os.getpid()}.json').write_text(json.dumps({'events':_events,'released':True},indent=2))
