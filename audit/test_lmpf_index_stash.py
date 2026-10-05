"""CPU schedule oracle for the actual cg_layer_run inner loop; no torch/CUDA.

Deferred index merge is modeled as an explicit attention-side callback, matching
sparse_attn_indexer._merge_dcp_topk_global_async. This proves host scheduling,
not GPU collective/kernel parity. Executes both committed and proposed loops.
"""
import ast
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATH = 'sm120/serve/nq_lmpf.py'

def extract(source):
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'LM')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'cg_layer_run')
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == 'prog')
    stub = ast.parse('def run(prog, inf, tib, stash, o, nk, env, G, sp, cv):\n pass').body[0]
    stub.body = [copy.deepcopy(loop)]
    tree = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
    ns = {}
    exec(compile(tree, PATH, 'exec'), ns)
    return ns['run']

class Buffer:
    def __init__(self, n, data=None, start=0, stop=None):
        self.data = [None] * n if data is None else data
        self.start, self.stop = start, n if stop is None else stop
    def __getitem__(self, key):
        lo, hi, step = key.indices(self.stop - self.start)
        assert step == 1
        return Buffer(0, self.data, self.start + lo, self.start + hi)
    def values(self):
        return self.data[self.start:self.stop]
    def copy_(self, other):
        values = other.values() if isinstance(other, Buffer) else list(other)
        assert len(values) == self.stop - self.start
        self.data[self.start:self.stop] = values

def run_case(loop, counts, starts, overlap=True):
    tib = Buffer(4096)
    pending = [None]
    bad_indexed = bad_shared = 0
    for turn, (count, start) in enumerate(zip(counts, starts)):
        # Model the previous turn/MTP touching unrelated scratch: the first
        # target indexer must overwrite all current rows independently.
        tib.copy_([('previous', turn)] * 4096)
        stash = Buffer(count)
        chunks = [(o, min(4096, count - o)) for o in range(0, count, 4096)]
        for layer, owns_index in enumerate((True, False, False)):
            for o, nk in chunks:
                # Absolute IDs retain arbitrary cached-prefix start. Distinct
                # per-token tuples stand for a deterministic global top-k.
                expected = [(start + o + j - 17, start + o + j - 9, start + o + j - 1) for j in range(nk)]
                observed = []
                def indexer():
                    if overlap and nk <= 4:
                        tib[:nk].copy_([tuple(v // 4 for v in row) for row in expected])
                        pending[0] = lambda: tib[:nk].copy_(expected)
                    else:
                        tib[:nk].copy_(expected)
                def attention():
                    if pending[0] is not None:
                        pending[0]()
                        pending[0] = None
                    observed.extend(tib[:nk].values())
                prog = []
                if owns_index:
                    prog.append(('idx_out', indexer, [], {}, 'idx'))
                prog.append(('attn_out', attention, [], {}, 'attn'))
                loop(prog, {'idx': owns_index}, tib, stash, o, nk, {}, [], {}, lambda x, *_: x)
                wrong = sum(a != b for a, b in zip(observed, expected))
                if owns_index:
                    bad_indexed += wrong
                else:
                    bad_shared += wrong
        assert pending[0] is None
    return {'wrong_indexed_rows': bad_indexed, 'wrong_shared_rows': bad_shared}

def main():
    baseline = extract(subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + PATH], text=True))
    fixed = extract((ROOT / PATH).read_text())
    results = {}
    for tail in (0, 1, 2, 3, 4, 5, 4095):
        for overlap in (False, True):
            key = f'tail={tail},overlap={overlap}'
            args = ([4096 + tail], [300013], overlap)
            old, new = run_case(baseline, *args), run_case(fixed, *args)
            assert old['wrong_indexed_rows'] == new['wrong_indexed_rows'] == 0
            assert old['wrong_shared_rows'] == (2 * tail if overlap and 1 <= tail <= 4 else 0)
            assert new['wrong_shared_rows'] == 0
            results[key] = {'baseline': old, 'fixed': new}
    # Repeated per-turn windows, varied absolute cached-prefix offsets, full
    # chunks, bad small tails, and ordinary tails. Includes context >300K.
    counts = [4096 + [0, 1, 2, 3, 4, 5, 91, 1023][i % 8] for i in range(250)]
    starts = [300013 + i * 1709 for i in range(250)]
    old, new = run_case(baseline, counts, starts), run_case(fixed, counts, starts)
    assert old['wrong_shared_rows'] > 0 and new['wrong_shared_rows'] == 0
    results['250_turns'] = {'baseline': old, 'fixed': new}
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
