import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from experts import ExpertStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = ExpertStore(args.model, cache_gib=0)
    resident = ExpertStore(args.model, cache_gib=0.01, resident_base=True)
    checks = []
    try:
        for expert in range(288):
            resident.get(3, expert, 2)
        base_bytes = resident.base_bytes
        base_reads = resident.metrics['base_tensor_bytes']
        assert resident.metrics['refinement_tensor_bytes'] == 0
        assert resident.cache_bytes == 0
        assert len(resident.base_cache) == 288
        checks.append('All layer 3 base experts loaded without refinement reads')
        x = mx.array(np.random.default_rng(42).normal(size=(5, 4096)).astype(np.float16))
        for expert in (0, 7, 287):
            for level in (2, 4):
                expected = reference.get(3, expert, level)
                actual = resident.get(3, expert, level)
                for projection, (planes, meta) in actual.items():
                    assert meta == expected[projection][1]
                    assert planes.keys() == expected[projection][0].keys()
                    for field, value in planes.items():
                        np.testing.assert_array_equal(np.asarray(value), np.asarray(expected[projection][0][field]))
                    if level == 4:
                        base = resident.base_cache[3, expert][projection][0]
                        for field in ('base', 'var', 'lrV', 'lrU2', 'suh2', 'svh2'):
                            if field in base:
                                assert planes[field] is base[field]
                wanted = reference.apply(3, expert, x, level)
                got = resident.apply(3, expert, x, level)
                mx.eval(wanted, got)
                np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))
                assert resident.cache_bytes <= resident.cache_limit
            checks.append(f'Expert {expert}: tensors and outputs match, base buffers are shared')
        assert resident.metrics['base_tensor_bytes'] == base_reads
        assert resident.base_bytes == base_bytes
        assert (3, 0, 4) not in resident.cache
        resident.get(3, 0, 4)
        assert resident.metrics['base_tensor_bytes'] == base_reads
        checks.append('Evicted refinement reloads without rereading base tensors')
        report = dict(status='passed', checks=checks, layer3_resident_base_bytes=base_bytes,
                      refinement_cache_bytes=resident.cache_bytes, metrics=resident.metrics.copy(),
                      scope='All layer 3 base experts; three full experts at both levels; bounded refinement eviction')
    finally:
        reference.close()
        resident.close()
    assert resident.base_bytes == resident.cache_bytes == 0
    assert not resident.base_cache and not resident.cache and not resident.handles
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
