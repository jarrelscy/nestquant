import os,torch,glob
os.environ.setdefault('CUDA_HOME','/home/jarrelscy/cuda128')
import sys;os.environ['PATH']=os.environ['CUDA_HOME']+'/bin:'+os.path.dirname(sys.executable)+':'+os.environ['PATH']
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0a')   # Spark (GB10): 12.1a; A100 test builds: 8.0
from torch.utils.cpp_extension import load
D=os.path.dirname(os.path.abspath(__file__))
# Compiler runtime headers must precede pip CUDA headers (CRT versions can differ).
CUDA_INCLUDES=glob.glob(os.path.join(os.path.dirname(torch.__file__),'..','nvidia','*','include'))
CUDA_HEADER_FLAGS=[f for p in CUDA_INCLUDES for f in ('-isystem',p)]
def get(defs=None):
    """defs: extra -D macros (diagnostic variants get their own module name / build dir)."""
    defs=defs if defs is not None else [d for d in os.environ.get('NQ_DEFS','').split(',') if d]
    tag=('_'+'_'.join(d.replace('=','') for d in defs)) if defs else ''
    b=os.environ.get('NQ_BUILD','/data/Jarrel/nq-build')+'/sm120'+tag;os.makedirs(b,exist_ok=True)
    return load('nqmoe120'+tag,[os.path.join(D,'nqmoe.cu')],extra_cflags=CUDA_HEADER_FLAGS,extra_cuda_cflags=CUDA_HEADER_FLAGS+['-O3','--use_fast_math','-lineinfo']+['-D'+d for d in defs],build_directory=b,verbose=False)
def get_sal():
    """nqsal.cu: decode salience export (GBDT x mps128 predictor); own module / build dir, independent of nqmoe."""
    b=os.environ.get('NQ_BUILD','/data/Jarrel/nq-build')+'/sm120_sal';os.makedirs(b,exist_ok=True)
    return load('nqsal120',[os.path.join(D,'nqsal.cu')],extra_cflags=CUDA_HEADER_FLAGS,extra_cuda_cflags=CUDA_HEADER_FLAGS+['-O3','-lineinfo'],build_directory=b,verbose=False)
if __name__=='__main__':get();print('built')
