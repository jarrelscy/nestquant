"""Pack binary signs and alias identical corrections without changing weights."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import mlx.core as mx
import numpy as np

from native_assets import COMPACT_FORMAT, REVISION, bank_storage_bytes, fingerprint, paths, read_bank


def compact(bank):
    if set(bank) != {'gate', 'up', 'down'}:
        raise ValueError('Expected three expert projections')
    arrays, geometry = {}, {}
    for name, (planes, meta) in bank.items():
        if 'var_bits' in meta or planes['var'].dtype != mx.uint8:
            raise ValueError('Expected unpacked uint8 signs')
        signs = np.array(planes['var'])
        if (signs.shape != (planes['base'].shape[0], meta['tk'] * meta['tn'] * 8)
                or np.any(signs > 1)):
            raise ValueError('Sign values must be binary with eight signs per unit')
        arrays.update({f'{name}.{key}': value for key, value in planes.items()})
        arrays[f'{name}.var'] = mx.array(np.packbits(signs, axis=-1, bitorder='little'))
        geometry[name] = dict(meta, var_bits=1)
    gate, up = arrays['gate.lrV'], arrays['up.lrV']
    if (gate.shape != up.shape or gate.dtype != up.dtype
            or not mx.array_equal(gate.view(mx.uint8), up.view(mx.uint8)).item()):
        raise ValueError('Gate and up corrections differ')
    del arrays['up.lrV']
    mx.eval(arrays)
    return arrays, geometry


def verify(original, loaded):
    compared = 0
    for name, (planes, meta) in original.items():
        actual, geometry = loaded[name]
        if geometry != dict(meta, var_bits=1) or planes.keys() != actual.keys():
            raise RuntimeError('Compact archive geometry or fields changed')
        for field, expected in planes.items():
            value = actual[field]
            if field == 'var':
                value = mx.array(np.unpackbits(np.array(value), axis=-1, bitorder='little'))
            if (value.dtype != expected.dtype or value.shape != expected.shape
                    or not mx.array_equal(value.view(mx.uint8), expected.view(mx.uint8)).item()):
                raise RuntimeError(f'Compact archive changed {name}.{field}')
            compared += expected.nbytes
    if loaded['gate'][0]['lrV'] is not loaded['up'][0]['lrV']:
        raise RuntimeError('Compact archive duplicated its correction alias')
    return compared


def convert_layer(source, destination, layer, check):
    file, sidecar = paths(source, layer)
    target, proof = paths(destination, layer)
    if target.exists() or proof.exists():
        raise FileExistsError('Use a fresh compact archive destination')
    evidence = json.loads(sidecar.read_text())
    with file.open('rb', buffering=0) as handle:
        fcntl.fcntl(handle, fcntl.F_NOCACHE, 1)
        if hashlib.file_digest(handle, 'sha256').hexdigest() != evidence['sha256']:
            raise RuntimeError('Source archive checksum mismatch')
    original = read_bank(source, layer, 'mlx-nocache')
    check()
    arrays, geometry = compact(original)
    stored = sum(value.nbytes for value in arrays.values())
    if shutil.disk_usage(destination).free < stored + 100 * 1024**3:
        raise RuntimeError('Compact archive would cross the disk reserve')
    temporary = target.with_suffix('.tmp.safetensors')
    mx.save_safetensors(temporary, arrays, metadata=dict(
        format=COMPACT_FORMAT, revision=REVISION, layer=str(layer), geometry=json.dumps(geometry),
        aliases=json.dumps({'up.lrV': 'gate.lrV'})))
    del arrays
    with temporary.open('rb', buffering=0) as handle:
        fcntl.fcntl(handle, fcntl.F_NOCACHE, 1)
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    temporary.replace(target)
    stage = proof.with_suffix('.tmp')
    stage.write_text(json.dumps(dict(revision=REVISION, state=fingerprint(target), sha256=digest), indent=2) + '\n')
    stage.replace(proof)
    check()
    loaded = read_bank(destination, layer, 'mlx-nocache')
    compared = verify(original, loaded)
    check()
    return dict(layer=layer, original_bytes=bank_storage_bytes(original), stored_bytes=bank_storage_bytes(loaded),
                compared_bytes=compared, sha256=digest, source_sha256=evidence['sha256'])


def main():
    from metrics import MemoryGuard

    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    mx.set_cache_limit(256 * 1024**2)
    guard = MemoryGuard(6)
    records = []
    for layer in range(3, 45):
        record = convert_layer(args.archive, args.output_dir, layer, lambda: guard.check(force=True))
        records.append(record)
        print(json.dumps(dict(event='compact_base_verified', **record)), flush=True)
        mx.clear_cache()
    (args.output_dir / 'manifest.json').write_text(json.dumps(dict(
        format=COMPACT_FORMAT, revision=REVISION, layers=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
