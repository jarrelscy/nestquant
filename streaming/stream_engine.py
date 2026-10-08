"""Build/load the nqstream io_uring upgrade engine (nqstream.cu) and a thin wrapper over one rank's record file."""
import os,sys,json,torch,glob
os.environ.setdefault('CUDA_HOME','/home/jarrelscy/cuda128')
os.environ['PATH']=os.environ['CUDA_HOME']+'/bin:'+os.path.dirname(sys.executable)+':'+os.environ['PATH']
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0a')
from torch.utils.cpp_extension import load
D=os.path.dirname(os.path.abspath(__file__));URING=os.environ.get('LIBURING','/data/Jarrel/liburing')
CUDA_HEADER_FLAGS=[f for p in glob.glob(os.path.join(os.path.dirname(torch.__file__),'..','nvidia','*','include')) for f in ('-isystem',p)]
_m=None
def mod():
    global _m
    if _m is None:
        b=os.environ.get('NQ_BUILD','/data/Jarrel/nq-build')+'/nqstream';os.makedirs(b,exist_ok=True)
        _m=load('nqstream',[D+'/nqstream.cu'],extra_include_paths=[URING+'/include'],extra_cflags=CUDA_HEADER_FLAGS,extra_ldflags=[URING+'/lib/liburing.a'],
                extra_cuda_cflags=CUDA_HEADER_FLAGS+['-O2'],build_directory=b,verbose=False)
    return _m

class RankFile:
    """record file of one TP rank (repack.py). rec(L, E) -> record index; engine(...) -> nqstream.Engine."""
    def __init__(s,repack_dir,rank):
        s.path=f'{repack_dir}/rank{rank}.bin';s.idx=json.load(open(f'{repack_dir}/rank{rank}.json'))
        s.rb=s.idx['rec_bytes'];s.lay=dict(seg=s.idx['seg'],rec_bytes=s.rb)
    def rec(s,L,E):return (L-s.idx['L0'])*s.idx['NE']+E
    def engine(s,n_host=64,qd=8,device=None,alt_path='',qd_alt=0,direct=False):
        """alt_path: identical copy of this record file on a second drive (nq-io dual path; '' = one drive);
        direct: unified memory, reads land straight in host-mapped slots (Engine.alloc_slots)"""
        return mod().Engine(s.path,s.rb,n_host,qd,torch.cuda.current_device() if device is None else device,alt_path,qd_alt,direct)

def unified_default(dev=None):
    """NQ_UNIFIED=1 / 0 forces direct-to-slot streaming on / off; unset = on for an integrated GPU (GB10, DGX Spark)"""
    v=os.environ.get('NQ_UNIFIED','')
    if v in ('0','1'):return v=='1'
    p=torch.cuda.get_device_properties(torch.cuda.current_device() if dev is None else dev)
    return bool(getattr(p,'is_integrated',False))
