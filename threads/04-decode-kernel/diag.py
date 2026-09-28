import torch;torch.cuda.set_per_process_memory_fraction(12/80)
from nq import *
from timing import bench
from orbit_duet.exl3_adapter import EXL3Expert
NC=6
bufs=[torch.randint(0,100,(6291456//4,),device='cuda',dtype=torch.int32) for _ in range(NC)]
big=[torch.randint(0,100,(4*6291456//4,),device='cuda',dtype=torch.int32) for _ in range(NC)]
o=torch.empty(1,device='cuda',dtype=torch.int64)
ex=[EXL3Expert('/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/exl3_e36/expert_4.bin') for _ in range(NC)]
x=torch.randn(1,6144,device='cuda').half();g=torch.empty(1,2048,device='cuda')
ps=[Proj('UNI2',4096,6144) for _ in range(NC)];y=torch.zeros(1,4096,device='cuda')
p4=[Proj('UNI4',2048,6144) for _ in range(NC)];y2=torch.zeros(1,2048,device='cuda')
fns={'sum6MB':lambda:[torch.sum(b,dim=0,out=o) for b in bufs],'sum25MB':lambda:[torch.sum(b,dim=0,out=o) for b in big],
     'exl3_gate4':lambda:[e.objects[0].bc.run(x,g) for e in ex],
     'uni2_gu':lambda:[p(x,y,(1,8,1,3)) for p in ps],'uni4_gate':lambda:[p(x,y2,(1,8,1,3)) for p in p4],
     'noop':lambda:[torch.cuda._sleep(0) for _ in range(NC)]}
for it in range(2):
    med,r=bench(fns,blocks=40,repeats=1)
    print({k:round(v/NC,2) for k,v in med.items()},{k:round(v[1]/NC,2) for k,v in r.items()})
