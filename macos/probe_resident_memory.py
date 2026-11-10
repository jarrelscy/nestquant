"""Measure the packed NestQuant base separately from the inference runtime."""

import argparse
import json
import os
from pathlib import Path
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from native_assets import bank_storage_bytes, read_bank
from resources import process_usage, system_pressure, vm_snapshot


def sample(event, bank_bytes, **fields):
    usage = process_usage()
    record = dict(event=event, logical_bank_bytes=bank_bytes, physical_bytes=usage.phys_footprint,
                  lifetime_physical_peak_bytes=usage.lifetime_max_phys_footprint,
                  resident_bytes=usage.resident_size, wired_bytes=usage.wired_size,
                  mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                  mlx_peak_bytes=mx.get_peak_memory(), read_bytes=usage.diskio_bytesread,
                  pressure=system_pressure(), system_vm=vm_snapshot(), **fields)
    print(json.dumps(record), flush=True)
    return record


def touch_pages(banks):
    checks = []
    seen = set()
    for bank in banks:
        for tensors, _ in bank.values():
            for array in tensors.values():
                if id(array) in seen:
                    continue
                seen.add(id(array))
                indices = mx.arange(0, array.size, 16384 // array.itemsize)
                checks.append(array.reshape(-1)[indices].astype(mx.float32).sum())
    mx.eval(checks)
    return dict(checks=len(checks), sample_sum=mx.stack(checks).sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reader', choices=('metal', 'mlx', 'mlx-nocache', 'mmap'), default='metal')
    parser.add_argument('--wire-base', action='store_true')
    parser.add_argument('--touch-each-layer', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--wired-gib', type=int, choices=(56, 68), default=56)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if (args.model is None) != (args.backbone is None):
        parser.error('Provide both --model and --backbone')
    if args.wired_gib != 56 and not args.wire_base:
        parser.error('--wired-gib requires --wire-base')
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(76 * 1024**3)
    if args.wire_base:
        if args.reader not in ('mlx', 'mlx-nocache'):
            parser.error('Use --reader mlx or mlx-nocache for this wired probe')
        mx.set_wired_limit(args.wired_gib * 1024**3)
    banks = []
    bank_bytes = 0
    records = [sample('start', bank_bytes, reader=args.reader, wired_limit_gib=args.wired_gib if args.wire_base else 0,
                      touch_each_layer=args.touch_each_layer,
                      residency_refresh_ms=os.environ.get('MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS', '0'))]
    backbone_bytes = 0
    if args.backbone is not None:
        from metrics import MemoryGuard
        from native_backbone import load_prepared

        guard = MemoryGuard(82)
        model = load_prepared(args.model, args.backbone, lambda *_: guard.check(force=True))
        weights = dict(tree_flatten(model.parameters()))
        backbone_bytes = sum(array.nbytes for array in weights.values())
        touched = touch_pages([{'backbone': (weights, {})}])
        records.append(sample('backbone-loaded', bank_bytes, logical_backbone_bytes=backbone_bytes, **touched))
        guard.check(force=True)
    for layer in range(3, 45):
        bank = read_bank(args.archive, layer, reader=args.reader)
        banks.append(bank)
        bank_bytes += bank_storage_bytes(bank)
        touched = touch_pages([bank]) if args.touch_each_layer else {}
        record = sample('layer', bank_bytes, layer=layer, logical_backbone_bytes=backbone_bytes, **touched)
        records.append(record)
        if record['pressure'] != 1 or max(record[key] for key in (
                'physical_bytes', 'lifetime_physical_peak_bytes', 'resident_bytes', 'wired_bytes')) > 82 * 1024**3:
            raise MemoryError('Resident bank load crossed the inner memory guard')
    for sweep in range(2):
        records.append(sample('page-sweep', bank_bytes, sweep=sweep,
                              logical_backbone_bytes=backbone_bytes, **touch_pages(banks)))
    time.sleep(2)
    records.append(sample('all-banks-live', bank_bytes, logical_backbone_bytes=backbone_bytes))
    with args.output.open('x') as handle:
        json.dump(records, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
