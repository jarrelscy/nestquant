"""Prepare a second record copy with bounded RAM, before workers start.

Matching metadata and sizes allow a fast repeat launch; this is not a bit-rot audit.
On a changed/incomplete release every record file is recopied. Metadata is published
last, so an interrupted copy cannot be mistaken for a complete new release.
"""
import os
from pathlib import Path
import shutil
import sys


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('dual-drive directories must differ')
    bins = [f'rank{r}.bin' for r in range(4)]
    meta = [f'rank{r}.json' for r in range(4)] + ['artifact_stamp.json']
    for name in bins + meta:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / '.nq-copy-incomplete'
    identical = not marker.exists() and all(
        (destination / n).is_file() and (source / n).read_bytes() == (destination / n).read_bytes()
        for n in meta)
    complete = all((destination / n).is_file() and
                   (source / n).stat().st_size == (destination / n).stat().st_size for n in bins)
    if identical and complete:
        print('Second record copy: metadata and record sizes match.')
        return
    # Each replacement temporarily needs space for its new file beside the old one.
    growth = 0
    required = 0
    for name in bins + meta:
        size = (source / name).stat().st_size
        old = (destination / name).stat().st_size if (destination / name).exists() else 0
        required = max(required, growth + size)
        growth += size - old
    if shutil.disk_usage(destination).free < required + 1024**3:
        raise RuntimeError(f'Insufficient second-drive space: need {required / 1024**3:.1f} GiB + 1 GiB margin')
    marker.write_text('Copy in progress; do not start workers against this directory.\n')
    for name in bins + meta:
        tmp = destination / (name + '.nq-partial')
        with (source / name).open('rb') as src, tmp.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024**2)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, destination / name)
        print(f'Second record copy: installed {name}', flush=True)
    marker.unlink()


if __name__ == '__main__':
    prepare(*sys.argv[1:])
