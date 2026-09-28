import torch,sys,json
torch.cuda.set_per_process_memory_fraction(12/80)
from orbit_duet.exl3_adapter import EXL3Expert
from torch.profiler import profile, ProfilerActivity
R='/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/'
for bits in [2,4]:
    e=EXL3Expert(R+f'expert_{bits}.bin')
    for B in [1,4]:
        x=torch.randn(B,6144,device='cuda')
        for _ in range(10): e.forward_batch(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(50): e.forward_batch(x)
            torch.cuda.synchronize()
        print(f'=== bits {bits} B{B}')
        print(p.key_averages().table(sort_by='cuda_time_total',row_limit=15,max_name_column_width=90))
