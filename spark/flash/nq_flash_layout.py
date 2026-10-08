"""Flash checkpoint/repack checks that do not import torch or initialize CUDA."""
import json
from pathlib import Path

FLASH_DEFS = 'NQ_RK_CODES=0x405,NQ_RK_GU=0x5,NQ_RK_DN=0x401,NQ_BK_CODES=0x21,NQ_SWIGLU_LIMIT=10'


def validate_checkpoint(root):
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text())
    text = cfg.get('text_config', cfg)
    expected = dict(hidden_size=4096, moe_intermediate_size=2048,
                    n_routed_experts=288, num_hidden_layers=45,
                    first_k_dense_replace=3, swiglu_limit=10.0)
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f'{key}: expected {value}, got {text.get(key)}')
    for layer in range(3, 45):
        path = root/'layers'/f'L{layer}'
        man = json.loads((path/'manifest.json').read_text())
        if man['format'] != 'nestquant-v1' or sorted(man['experts']) != list(range(288)):
            raise ValueError(f'Invalid L{layer} expert manifest')
        for proj, (n, k, residual) in dict(gate=(2048,4096,2.5),
                                         up=(2048,4096,2.5), down=(4096,2048,2.8125)).items():
            meta = man['proj_meta'][proj]
            if (meta['n'], meta['k'], meta['base_K'], meta['res_rule']) != (
                    n, k, 1.5, dict(kind='uniform', K=residual)):
                raise ValueError(f'Unsupported L{layer} {proj} format: {meta}')
        for shard in range(8):
            if not (path/f'tp{shard}.safetensors').is_file():
                raise FileNotFoundError(path/f'tp{shard}.safetensors')
    return cfg


def validate_repack(root):
    root = Path(root)
    idx = json.loads((root/'rank0.json').read_text())
    for key, value in dict(format='nq-p4rec-v1',tp=1,rank=0,L0=3,NE=288).items():
        if idx.get(key) != value:
            raise ValueError(f'Invalid repack {key}: {idx.get(key)} != {value}')
    if set(idx['layers']) != {str(i) for i in range(3,45)}:
        raise ValueError('Flash requires all 42 routed layers; partial repacks cannot serve')
    for layer in range(3,45):
        if sorted(idx['layers'][str(layer)]['experts']) != list(range(288)):
            raise ValueError(f'Incomplete L{layer}')
        if not (root/'res/rank0'/f'L{layer}.pt').is_file():
            raise FileNotFoundError(f'Resident L{layer}')
    expected_bytes = 42*288*idx['rec_bytes']
    if (root/'rank0.bin').stat().st_size != expected_bytes:
        raise ValueError('Record file length does not match Flash layout')
    return idx


def tile_config(n, k):
    # Conservative valid tile, independent of the full-GLM tuned table (K=6144).
    cfg = [1,8,2]
    if n%16 or k%(cfg[1]*cfg[2]*128):
        raise ValueError(f'Unsupported Flash tile dimensions N={n} K={k}')
    return cfg


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--repack')
    a=p.parse_args();validate_checkpoint(a.checkpoint)
    if a.repack:validate_repack(a.repack)
    print('Flash checkpoint format validated'+('; TP1 repack validated' if a.repack else ''))
