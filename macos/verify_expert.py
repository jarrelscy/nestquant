import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from experts import ExpertStore


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--resident-base', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'sm120'), str(root / 'threads/12-reference-encoder')]
    # group_art is a CPU layout oracle; importing its sibling kernels must not compile CUDA.
    with patch.dict(sys.modules, {'build': types.SimpleNamespace(get=lambda: None)}):
        loader = import_file('nq_cpu_loader', root / 'sm120/nqload.py')
    container = import_file('nq_cpu_container', root / 'threads/25-campaign/nq25_st.py')
    ref = loader.D
    layer = args.model / 'layers/L3'
    manifest = json.loads((layer / 'manifest.json').read_text())
    parts = {}
    for shard in range(8):
        with safe_open(layer / f'tp{shard}.safetensors', framework='pt', device='cpu') as handle:
            tree = json.loads(handle.metadata()['tree'])
            node = next(node for expert, node in tree['D'] if expert == args.expert)
            parts[shard] = {args.expert: container._untree(node, handle.get_tensor)}
    artifact = loader.group_art(parts, manifest, args.expert, list(range(8)))
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 128 ** 0.5
    rotated = {name: ref.rotated_levels(projection, 'cpu') for name, projection in artifact.items()}

    def projection(x, name, level):
        plane = artifact[name]['base' if level == 2 else 'p4']
        shape = x.shape
        transformed = ((x * plane['suh'].float()).reshape(-1, 128) @ h).reshape(shape).half().float()
        y = transformed @ rotated[name][level].float()
        y = (y.reshape(-1, 128) @ h).reshape(x.shape[0], -1) * plane['svh'].float()
        lr = artifact[name]['base'].get('lr')
        if lr is not None:
            low = x @ lr['V'].float().T
            y = y + low @ lr['U2'].float()
            if level == 4:
                y = y + low @ artifact[name]['p4']['lr']['U4'].float()
        return y

    torch.set_num_threads(8)
    torch.manual_seed(414)
    store = ExpertStore(args.model, cache_gib=0.05, resident_base=args.resident_base)
    checks = []
    try:
        for level in (2, 4):
            for batch_size in (1, 5):
                x = torch.randn(batch_size, 4096).half().float()
                gate = projection(x, 'gate', level).clamp(max=10)
                up = projection(x, 'up', level).clamp(min=-10, max=10)
                expected = projection(torch.nn.functional.silu(gate) * up, 'down', level).numpy()
                actual = store.apply(3, args.expert, mx.array(x.numpy()), level)
                mx.eval(actual)
                a = np.asarray(actual)
                relative = float(np.linalg.norm(a - expected) / np.linalg.norm(expected))
                check = dict(level=level, tokens=batch_size, relative_rms=relative,
                             max_abs=float(np.max(np.abs(a - expected))))
                print(json.dumps(check), flush=True)
                checks.append(check)
                assert relative < 0.001, check
        assert store.cache_bytes <= store.cache_limit
    finally:
        store.close()
    args.output.write_text(json.dumps(dict(status='passed', expert=args.expert, layer=3, resident_base=args.resident_base,
                                          scope='Full eight-shard expert against upstream CPU layout and decoder', checks=checks), indent=2) + '\n')


if __name__ == '__main__':
    main()
