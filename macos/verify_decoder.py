import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from decoder import decode_rings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(8), default=0)
    parser.add_argument('--experts', type=int, nargs='+', default=[0, 7, 98, 177, 287])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[1] / 'threads/12-reference-encoder/nq_decode.py'
    spec = importlib.util.spec_from_file_location('nq_reference', reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch.set_num_threads(8)
    manifest = json.loads((args.layer / 'manifest.json').read_text())
    shard = args.layer / f'tp{args.shard}.safetensors'
    with shard.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == manifest['files'][shard.name]['sha256'], 'Checkpoint checksum mismatch'
    report = {'status': 'running', 'shard_sha256': digest, 'device': mx.device_info(),
              'scope': f'{args.layer.name} TP{args.shard} ring decoding only; no full-model inference', 'checks': []}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    with safe_open(shard, framework='pt', device='cpu') as tensors:
        for expert in args.experts:
            for projection in ('gate', 'up', 'down'):
                key = f'E{expert}.{projection}'
                base, residual, words, variants = [tensors.get_tensor(f'{key}.{p}') for p in ('base', 'p4', 'word', 'var')]
                meta = manifest['proj_meta'][projection]
                bk = float(meta['base_K'])
                rk = float(meta['res_rule']['K'])
                assert meta['base_var'] == 'sign' and meta['res_rule']['kind'] == 'uniform'
                units = words.numel()
                sb = ref.hsum(ref.stream_states(base.reshape(-1, int(bk * 32)), bk)).reshape(units, 8, 256)
                sr = ref.hsum(ref.stream_states(residual.reshape(-1, int(rk * 32)), rk)).reshape(units, 8, 256)
                sign = ref.variant_table('sign', 'cpu')[variants.long()].reshape(units, 8, 1)
                qb = ref.q2_values(sb, sign).half().numpy()
                qh = ref.fold(sb, sr, words.long() & 255, (words.long() >> 8) & 255, sign).half().numpy()
                inputs = [mx.array(x.numpy()) for x in (base, residual, words, variants)]
                out = decode_rings(*inputs, bk, rk)
                mx.eval(out)
                expected = [qb, qh, sb.numpy().astype(np.uint16), sr.numpy().astype(np.uint16)]
                diffs = [int(np.count_nonzero(np.asarray(a) != b)) for a, b in zip(out, expected)]
                times = []
                for _ in range(3):
                    t = time.perf_counter()
                    measured = decode_rings(*inputs, bk, rk)
                    mx.eval(measured)
                    times.append((time.perf_counter() - t) * 1000)
                check = {'expert': expert, 'projection': projection, 'weights_per_level': qb.size,
                         'mismatch_counts_base_hot_sb_sr': diffs, 'kernel_ms': times}
                report['checks'].append(check)
                print(json.dumps(check), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                if any(diffs):
                    report['status'] = 'failed'
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    raise AssertionError(check)
    report.update(status='passed', elapsed_seconds=time.monotonic() - started,
                  decoded_weight_values=sum(c['weights_per_level'] * 2 for c in report['checks']))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    main()
