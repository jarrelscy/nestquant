"""Pack one layer at a time for bounded NestQuant resident loading."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import mlx.core as mx

FORMAT = 'nestquant-resident-base-v1'
REVISION = '4023870a6f2c6097e37fa5d2e54040db64b011e9'


def fingerprint(path):
    stat = path.stat()
    return dict(bytes=stat.st_size, inode=stat.st_ino, device=stat.st_dev, mtime_ns=stat.st_mtime_ns)


def paths(directory, layer):
    return directory / f'L{layer}.safetensors', directory / f'L{layer}.json'


def build(model, directory):
    from experts import ExpertStore
    from metrics import MemoryGuard
    from packed_expert_bank import pack
    from checkpoint import validate

    validate(model)
    directory.mkdir(exist_ok=True)
    mx.set_cache_limit(256 * 1024**2)
    guard = MemoryGuard(7)
    store = ExpertStore(model, cache_gib=0)
    store.memory_check = guard.check
    try:
        for number in range(3, 45):
            file, sidecar = paths(directory, number)
            if file.exists() and sidecar.exists():
                data = json.loads(sidecar.read_text())
                if data['revision'] == REVISION and data['state'] == fingerprint(file):
                    continue
                raise RuntimeError(f'Resident archive changed: {file.name}')
            if shutil.disk_usage(directory).free < 102 * 1024**3:
                raise RuntimeError('Insufficient disk reserve for another resident bank')
            experts = [store.get(number, expert, 2) for expert in range(288)]
            arrays = {}
            geometry = {}
            for name, (_, meta) in experts[0].items():
                if any(expert[name][1] != meta for expert in experts):
                    raise RuntimeError(f'Projection geometry differs within layer {number}')
                geometry[name] = meta
                arrays.update({f'{name}.{key}': value
                               for key, value in pack([expert[name][0] for expert in experts]).items()})
            mx.eval(arrays)
            del experts
            guard.check(force=True)
            temporary = file.with_suffix('.tmp.safetensors')
            mx.save_safetensors(temporary, arrays, metadata=dict(
                format=FORMAT, revision=REVISION, layer=str(number), geometry=json.dumps(geometry)))
            del arrays
            with temporary.open('rb') as handle:
                fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            temporary.replace(file)
            stage = sidecar.with_suffix('.tmp')
            stage.write_text(json.dumps(dict(revision=REVISION, state=fingerprint(file), sha256=digest), indent=2) + '\n')
            stage.replace(sidecar)
            print(json.dumps(dict(event='resident_archive_written', layer=number, bytes=file.stat().st_size)), flush=True)
            store.close()
    finally:
        store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    build(args.model, args.output_dir)
