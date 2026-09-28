import os,torch
os.environ['CUDA_HOME']='/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13'
os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
from torch.utils.cpp_extension import load
D=os.path.dirname(os.path.abspath(__file__))
def get(count=False,G=int(os.environ.get('NQ15_G',2))):
    sfx=('_count' if count else '')+('' if G==2 else f'_g{G}')
    bd='/tmp/nestquant/15-level4-decode/build'+sfx;os.makedirs(bd,exist_ok=True)
    return load('nqk15'+('c' if count else '')+('' if G==2 else f'g{G}'),[os.path.join(D,'nqk15.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo']+(['-DNO_WDBG'] if count else [])+([f'-DNQ_G={G}'] if G!=2 else []),
                build_directory=bd,verbose=False)
if __name__=='__main__':
    import sys;get(len(sys.argv)>1);print('built')
