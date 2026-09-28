import os,torch
os.environ['CUDA_HOME']='/home/coder/git/glm52/.venv/lib/python3.12/site-packages/nvidia/cu13'
os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
from torch.utils.cpp_extension import load
D=os.path.dirname(os.path.abspath(__file__))
def get():
    return load('nqk',[os.path.join(D,'nqk.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-Xptxas=-v'],
                build_directory='/tmp/nestquant/04-decode-kernel/build',verbose=False)
if __name__=='__main__':
    os.makedirs('/tmp/nestquant/04-decode-kernel/build',exist_ok=True);get();print('built')
