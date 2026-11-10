import argparse
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from decoder import decode_rings, dense_projection, rotated_matrix, projection_matvec, projection_matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    path = Path(__file__).resolve().parents[1] / 'threads/12-reference-encoder/nq_decode.py'
    spec = importlib.util.spec_from_file_location('nq_reference', path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h *= 128 ** -0.5
    manifest = json.loads((args.layer / 'manifest.json').read_text())
    checks = []
    rng = np.random.default_rng(2718)
    with safe_open(args.layer / 'tp0.safetensors', framework='pt', device='cpu') as f:
        for expert in (0, 7, 287):
            for projection in ('gate', 'up', 'down'):
                prefix = f'E{expert}.{projection}'
                get = lambda field: f.get_tensor(prefix + '.' + field)
                meta = dict(manifest['proj_meta'][projection])
                for dim in (('k', 'tk') if projection == 'down' else ('n', 'tn')):
                    meta[dim] //= 8
                planes = [get(p) for p in ('base', 'p4', 'word', 'var')]
                outputs = decode_rings(*[mx.array(t.numpy()) for t in planes], meta['base_K'], meta['res_rule']['K'])
                art = dict(meta=meta, base=dict(shards=[planes[0]], var=[planes[3]]),
                           p4=dict(shards=[planes[1]], word=[planes[2]]))
                reference = ref.rotated_levels(art, device='cpu')
                for level in (2, 4):
                    q = reference[level].half()
                    k, n = q.shape
                    w = (h @ q.float().reshape(-1, 128, n)).reshape(k, n).half()
                    w *= get(f'suh{level}')[:, None]
                    w = (w.float().reshape(-1, 128) @ h).reshape(k, n).half()
                    w *= get(f'svh{level}')[None, :]
                    expected = w.float().T
                    v = u2 = u4 = None
                    if prefix + '.lrU2' in f.keys():
                        vkey = prefix + '.lrV'
                        if vkey not in f.keys():
                            assert projection == 'up'
                            vkey = f'E{expert}.gate.lrV'
                        v, u2 = f.get_tensor(vkey), get('lrU2')
                        expected = expected + u2.float().T @ v.float()
                        if level == 4:
                            u4 = get('lrU4')
                            expected = expected + u4.float().T @ v.float()
                    convert = lambda x: None if x is None else mx.array(x.numpy())
                    actual = dense_projection(outputs[0 if level == 2 else 1], meta,
                                              convert(get(f'suh{level}')), convert(get(f'svh{level}')),
                                              convert(v), convert(u2), convert(u4))
                    mx.eval(actual)
                    a, e = np.asarray(actual), expected.numpy()
                    relative_rms = float(np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-30))
                    rotated = np.asarray(rotated_matrix(outputs[0 if level == 2 else 1], meta))
                    check = dict(expert=expert, projection=projection, level=level,
                                 rotated_mismatches=int(np.count_nonzero(rotated != reference[level].numpy())),
                                 dense_relative_rms=relative_rms, dense_max_abs=float(np.max(np.abs(a-e))))
                    packed = {name: convert(get(name)) for name in ('base', 'p4', 'word', 'var', 'suh2', 'svh2', 'suh4', 'svh4')}
                    if v is not None:
                        packed.update(lrV=convert(v), lrU2=convert(u2), lrU4=convert(get('lrU4')))
                    matvec_errors = []
                    for _ in range(3):
                        x = rng.standard_normal(meta['k']).astype(np.float32)
                        result = projection_matvec(packed, meta, mx.array(x), level)
                        mx.eval(result)
                        oracle = expected.numpy() @ x
                        error = float(np.linalg.norm(np.asarray(result) - oracle) / np.linalg.norm(oracle))
                        matvec_errors.append(error)
                    batch = rng.standard_normal((5, meta['k'])).astype(np.float32)
                    batched = projection_matmul(packed, meta, mx.array(batch), level)
                    mx.eval(batched)
                    batch_oracle = batch @ expected.numpy().T
                    batch_error = float(np.linalg.norm(np.asarray(batched) - batch_oracle) / np.linalg.norm(batch_oracle))
                    check['batch_matmul_relative_rms'] = batch_error
                    check['fused_matvec_relative_rms'] = matvec_errors
                    checks.append(check)
                    print(json.dumps(check), flush=True)
                    assert check['rotated_mismatches'] == 0
                    assert relative_rms < 0.001, check
                    assert max(matvec_errors) < 0.001, check
                    assert batch_error < 0.001, check
    args.output.write_text(json.dumps(dict(status='passed', scope='L3 TP0 partial projections with Hadamard transforms, scales and low-rank corrections', tolerance_relative_rms=0.001, checks=checks), indent=2) + '\n')


if __name__ == '__main__':
    main()
