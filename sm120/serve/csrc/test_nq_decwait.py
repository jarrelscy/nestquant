# spin kernel: flag set by a host thread after D ms -> GPU wait ~D; timeout path; run in the serve container
import os,time,threading,torch
from torch.utils.cpp_extension import load
load(name='nq_decwait',sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),'nq_decwait.cu')],extra_cuda_cflags=['-O3'],is_python_module=False)
f=torch.zeros(1,dtype=torch.int64,pin_memory=True);fv=f.numpy();st=torch.zeros(3,dtype=torch.int64,pin_memory=True)
x=torch.randn(1024,1024,device='cuda')
for n,(delay,tmo) in enumerate([(0.005,0.05),(0.020,0.05),(0.2,0.01),(0.0,0.01)],1):
    torch.cuda.synchronize();t=threading.Timer(delay,lambda n=n:fv.__setitem__(0,n));t0=time.perf_counter();t.start()
    torch.ops.nq_decwait.spin(f.data_ptr(),n,int(tmo*1e9),st.data_ptr());y=x@x;th=(time.perf_counter()-t0)*1e3
    torch.cuda.synchronize();tg=(time.perf_counter()-t0)*1e3;t.join()
    print(f'n={n} delay {delay*1e3:.0f} ms timeout {tmo*1e3:.0f} ms: host enqueue {th:.2f} ms, gpu done {tg:.1f} ms, stat {st.tolist()}')
