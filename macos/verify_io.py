import argparse
from collections import Counter
import json
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open

from backbone import parameter_sources, text_config
from positional_reader import PositionalTensorReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    tensors = 0
    total_bytes = 0
    calls = Counter()
    for shard in range(8):
        path = args.model / 'layers/L3' / f'tp{shard}.safetensors'
        with safe_open(path, framework='numpy', backend='pread') as reference, \
                closing(PositionalTensorReader(path)) as grouped:
            for expert in ('E0', 'E7', 'E287'):
                keys = [key for key in reference.keys() if key.startswith(expert + '.')]
                for phase in ('base', 'refinement', 'all'):
                    selected = [key for key in keys if phase == 'all' or
                                (key.rsplit('.', 1)[-1] in ('p4', 'word', 'suh4', 'svh4', 'lrU4')) == (phase == 'refinement')]
                    with patch('positional_reader.os.preadv', wraps=os.preadv) as read:
                        values = grouped.get_tensors(selected)
                        calls[f'{phase}_grouped'] += read.call_count
                    calls[f'{phase}_individual'] += len(selected)
                    for key, actual in values.items():
                        expected = reference.get_tensor(key)
                        assert actual.dtype == expected.dtype and actual.shape == expected.shape
                        assert actual.tobytes() == expected.tobytes()
                        tensors += 1
                        total_bytes += actual.nbytes
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    wanted = {name for names in parameter_sources(args.model, text_config(args.model)).values() for name in names}
    wanted.update(name.removesuffix('.weight') + '.weight_scale_inv' for name in list(wanted)
                  if name.removesuffix('.weight') + '.weight_scale_inv' in index)
    backbone = []
    dtypes = Counter()
    for filename in sorted({index[name] for name in wanted}):
        with safe_open(args.model / filename, framework='pt', device='cpu', backend='mmap') as mapped, \
                safe_open(args.model / filename, framework='pt', device='cpu', backend='pread') as streamed:
            for key in mapped.keys():
                if key not in wanted:
                    continue
                tensor_slice = mapped.get_slice(key)
                shape, dtype = tensor_slice.get_shape(), tensor_slice.get_dtype()
                if np.prod(shape) > 16 * 1024**2 or dtypes[dtype] >= 3:
                    continue
                expected, actual = mapped.get_tensor(key), streamed.get_tensor(key)
                assert expected.shape == actual.shape and expected.dtype == actual.dtype
                assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
                backbone.append(dict(name=key, dtype=str(actual.dtype), bytes=actual.numel() * actual.element_size()))
                dtypes[dtype] += 1
                if len(backbone) == 9:
                    break
        if len(backbone) == 9:
            break
    assert tensors > 0 and len(backbone) == 9
    assert {item['dtype'] for item in backbone} == {'torch.bfloat16', 'torch.float32', 'torch.float8_e4m3fn'}
    report = dict(status='passed', expert_tensors=tensors, expert_tensor_bytes=total_bytes, backbone=backbone,
                  expert_read_calls=dict(calls), scope='Exact grouped/individual expert bytes for three layer 3 experts across eight shards, and exact mmap/pread agreement for nine backbone tensors; no throughput claim')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
