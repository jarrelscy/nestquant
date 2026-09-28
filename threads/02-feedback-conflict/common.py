import os, sys, torch
sys.path.insert(0, '/home/coder/git/orbit-duet'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.cuda.set_per_process_memory_fraction(12/80)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(8)
GLM_STATS = '/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l16/statistics/l16_e36.pt'
MIMO_STATS = '/home/coder/git/orbit-duet/runs/full55_statistics/l55_e70.pt'
def glm():
    from orbit_duet.source import weights
    Ws = weights('/tmp/nestquant/glm53-fp8-experts', 16, 36)
    st = torch.load(GLM_STATS, map_location='cpu', mmap=True, weights_only=True)
    return Ws, [st['grams'][0].cuda(), st['grams'][0].cuda(), st['grams'][1].cuda()]
def mimo_H():
    st = torch.load(MIMO_STATS, map_location='cpu', mmap=True, weights_only=True)
    return [st['grams'][0].cuda(), st['grams'][0].cuda(), st['grams'][1].cuda()]
