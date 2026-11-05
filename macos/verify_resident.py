"""Compare fully preloaded device routes with the real per-expert reference."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from checkpoint import validate
from decoder import projection_matvec_banked
from experts import ExpertStore
from metrics import MemoryGuard
from native_assets import REVISION, fingerprint, paths
from packed_expert_bank import PackedExpertBank


EXPERTS = (0, 7, 98, 177, 218, 241, 277, 287)


def compare(root, archive, layer, guard):
    file, sidecar = paths(archive, layer)
    evidence = json.loads(sidecar.read_text())
    if evidence['revision'] != REVISION or evidence['state'] != fingerprint(file):
        raise RuntimeError('Prepared expert archive identity differs')
    bank = PackedExpertBank(file, layer, 288, REVISION, remaining_bytes=2 * 1024**3)
    store = ExpertStore(root, cache_gib=0.25)
    try:
        bank.preload(lambda: guard.check(force=True))
        before = dict(bank.stats)
        inputs = mx.random.normal((len(EXPERTS), 4096)).astype(mx.float16).astype(mx.float32)
        expected = mx.stack([store.apply(layer, expert, inputs[row], 2)
                             for row, expert in enumerate(EXPERTS)])
        mx.eval(expected)
        with patch('packed_expert_bank.os.preadv', side_effect=AssertionError('Unexpected base read')):
            route = bank.select_device(mx.array(EXPERTS, dtype=mx.uint32))
            gate = mx.minimum(projection_matvec_banked(*bank.planes['gate'], inputs, route, 2), 10.0)
            up = mx.clip(projection_matvec_banked(*bank.planes['up'], inputs, route, 2), -10.0, 10.0)
            actual = projection_matvec_banked(*bank.planes['down'], gate * mx.sigmoid(gate) * up, route, 2)
            bank.retain(actual)
            mx.eval(actual)
        error = (mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item()
        if not mx.isfinite(actual).all().item() or not error < 0.001:
            raise RuntimeError(f'Resident expert comparison failed: {error}')
        if bank.stats['read_bytes'] != before['read_bytes'] or bank.stats['misses'] != 0:
            raise RuntimeError('A fully resident computation loaded a missing base')
        if evidence['state'] != fingerprint(file):
            raise RuntimeError('Prepared expert archive changed during validation')
        guard.check(force=True)
        return dict(layer=layer, relative_rms=error, base_bytes=bank.bytes,
                    preload_read_bytes=before['read_bytes'], decode_read_bytes=0,
                    base_misses=bank.stats['misses'], resident_routes=bank.stats['resident_routes'])
    finally:
        store.close()
        bank.close()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    validate(args.model)
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(6 * 1024**3)
    mx.random.seed(42)
    guard = MemoryGuard(6)
    checks = [compare(args.model, args.archive, layer, guard) for layer in (3, 44)]
    report = dict(status='passed', scope='Real eight-expert comparisons at layers 3 and 44, base precision only',
                  relative_rms_tolerance=0.001, checks=checks)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
