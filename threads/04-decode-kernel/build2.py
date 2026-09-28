import os,torch
os.environ['CUDA_HOME']='/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13'
os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
from torch.utils.cpp_extension import load
D=os.path.dirname(os.path.abspath(__file__))
def get():
    os.makedirs('/tmp/nestquant/04-decode-kernel/build2',exist_ok=True)
    return load('nqk2',[os.path.join(D,'nqk2.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo'],
                build_directory='/tmp/nestquant/04-decode-kernel/build2',verbose=False)
if __name__=='__main__':get();print('built')
