"""Build/load the nqstream io_uring upgrade engine (nqstream.cu) and a thin wrapper over one rank's record file."""
import os,sys,json,torch
os.environ.setdefault('CUDA_HOME','/home/jarrelscy/cuda128')
os.environ['PATH']=os.environ['CUDA_HOME']+'/bin:'+os.path.dirname(sys.executable)+':'+os.environ['PATH']
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0a')
from torch.utils.cpp_extension import load
D=os.path.dirname(os.path.abspath(__file__));URING=os.environ.get('LIBURING','/data/Jarrel/liburing')
_m=None
def mod():
    global _m
    if _m is None:
        b='/data/Jarrel/nq-build/nqstream';os.makedirs(b,exist_ok=True)
        _m=load('nqstream',[D+'/nqstream.cu'],extra_include_paths=[URING+'/include'],extra_ldflags=[URING+'/lib/liburing.a'],
                extra_cuda_cflags=['-O2'],build_directory=b,verbose=False)
    return _m

class RankFile:
    """record file of one TP rank (repack.py). rec(L, E) -> record index; engine(...) -> nqstream.Engine."""
    def __init__(s,repack_dir,rank):
        s.path=f'{repack_dir}/rank{rank}.bin';s.idx=json.load(open(f'{repack_dir}/rank{rank}.json'))
        s.rb=s.idx['rec_bytes'];s.lay=dict(seg=s.idx['seg'],rec_bytes=s.rb)
    def rec(s,L,E):return (L-s.idx['L0'])*s.idx['NE']+E
    def engine(s,n_host=64,qd=8,device=None):
        return mod().Engine(s.path,s.rb,n_host,qd,torch.cuda.current_device() if device is None else device)
