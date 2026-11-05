import argparse
import json
from pathlib import Path

import mlx.core as mx

from decoder import projection_matvec_banked
from metrics import MemoryGuard
from native_assets import REVISION, fingerprint, paths
from packed_expert_bank import BaseReadStaging, PackedExpertBank
from resources import process_usage, system_pressure


def verify(source, archive, layer, guard, staging=None):
    banks = []
    try:
        for index, directory in enumerate((source, archive)):
            file, sidecar = paths(directory, layer)
            evidence = json.loads(sidecar.read_text())
            if evidence['revision'] != REVISION or evidence['state'] != fingerprint(file):
                raise RuntimeError('Prepared archive fingerprint changed')
            banks.append(PackedExpertBank(file, layer, 32, REVISION, 512 * 1024**2,
                                          staging if index == 1 else None))
        reference, candidate = banks
        if reference.expert_row_bytes is not None or candidate.expert_row_bytes is None:
            raise ValueError('Expected a field-major reference and expert-major candidate')
        tensors, checked_bytes = 0, 0
        for expert in range(288):
            left, left_bytes, _ = reference.read_payload(expert)
            right, right_bytes, _ = candidate.read_payload(expert)
            if left_bytes != right_bytes or set(left) != set(right):
                raise RuntimeError('Expert payload fields or sizes differ')
            for name in left:
                if left[name] != right[name]:
                    raise RuntimeError(f'Expert bytes differ at layer {layer}, expert {expert}, plane {name}')
                tensors += 1
                checked_bytes += len(left[name])
            del left, right
            guard.check(force=True)
        projections = []
        for first in (0, 80, 160, 240, 16, 0):
            experts = tuple(range(first, first + 8))
            routes = [bank.select(experts) for bank in banks]
            outputs = [[], []]
            for name, (_, meta) in reference.planes.items():
                inputs = mx.random.normal((8, meta['k'])).astype(mx.float16).astype(mx.float32)
                values = [projection_matvec_banked(*bank.planes[name], inputs, route, 2)
                          for bank, route in zip(banks, routes, strict=True)]
                mx.eval(values)
                if not mx.array_equal(*values).item() or not mx.isfinite(values[0]).all().item():
                    raise RuntimeError(f'Projection differs at layer {layer}, plane {name}')
                for destination, value in zip(outputs, values, strict=True):
                    destination.append(value.sum())
                projections.append(dict(first_expert=first, projection=name, identical=True))
            for bank, output in zip(banks, outputs, strict=True):
                bank.retain(mx.stack(output))
            guard.check(force=True)
        return dict(layer=layer, experts=288, tensors=tensors, checked_bytes=checked_bytes,
                    projections=projections, evictions=[bank.stats['evictions'] for bank in banks],
                    candidate_stats=dict(candidate.stats))
    finally:
        for bank in banks:
            bank.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', choices=range(3, 45), default=list(range(3, 45)))
    parser.add_argument('--base-read-mode', choices=('scatter', 'staged'), default='scatter')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    mx.random.seed(42)
    mx.set_cache_limit(128 * 1024**2)
    mx.set_memory_limit(3 * 1024**3)
    guard = MemoryGuard(4)
    staging = BaseReadStaging() if args.base_read_mode == 'staged' else None
    checks = [verify(args.source, args.archive, layer, guard, staging) for layer in args.layers]
    usage = process_usage()
    memory = dict(physical_bytes=usage.phys_footprint,
                  lifetime_peak_physical_bytes=usage.lifetime_max_phys_footprint,
                  mlx_peak_bytes=mx.get_peak_memory(), pressure_level=system_pressure())
    with args.output.open('x') as stream:
        json.dump(dict(status='passed', checks=checks, memory=memory, base_read_mode=args.base_read_mode,
                       base_staging_bytes=len(staging.storage) if staging is not None else 0), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='passed', checked_bytes=sum(row['checked_bytes'] for row in checks))))
