"""run CODE/streaming/repack.py on CPU (CUDA kernel build stubbed: repack never launches it). argv: CODE SRC OUT LAYER [RANKS]"""
import sys, types, runpy, torch
code, src, out, L = sys.argv[1:5]
sys.modules['build'] = types.SimpleNamespace(get=lambda *a, **k: None, get_sal=lambda *a, **k: None)
sys.path[:0] = [f'{code}/streaming', f'{code}/sm120']
torch.cuda.empty_cache = lambda: None
torch.set_num_threads(int(__import__('os').environ.get('NT', '4')))
import nqload as NQ
NQ.RankLayer.__init__.__defaults__ = (4, None, 'cpu')
sys.argv = [f'{code}/streaming/repack.py', src, out, '4', L] + ([__import__('os').environ['REC_BYTES']] if __import__('os').environ.get('REC_BYTES') else [])
runpy.run_path(f'{code}/streaming/repack.py', run_name='__main__')
