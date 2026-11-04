import argparse
import hashlib
import json
from pathlib import Path

from checkpoint import validate


def expected_files(root: Path):
    result = []
    for layer in range(3, 45):
        directory = Path('layers') / f'L{layer}'
        manifest = json.loads((root / directory / 'manifest.json').read_text())
        for shard in range(8):
            name = f'tp{shard}.safetensors'
            item = manifest['files'][name]
            result.append(dict(file=str(directory / name), bytes=item['bytes'], sha256=item['sha256']))
    nonexpert = json.loads((root / 'nonexpert_manifest.json').read_text())
    files = {item['file']: item for item in nonexpert['files']}
    for number in range(1, 5):
        name = f'nonexpert-{number:05d}-of-00004.safetensors'
        item = files[name]
        result.append(dict(file=name, bytes=item['bytes'], sha256=item['sha256']))
    predictor = json.loads((root / 'serving/predictor/predictor.json').read_text())
    result.append(dict(file='serving/predictor/jt/jT.safetensors', sha256=predictor['model_sha256']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file to preserve earlier verification results')
    root = args.model.expanduser().resolve()
    validate(root)
    verified = []
    for item in expected_files(root):
        path = root / item['file']
        size = path.stat().st_size
        if 'bytes' in item and size != item['bytes']:
            raise ValueError(f'Incomplete file: {item["file"]}')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'SHA-256 mismatch: {item["file"]}')
        verified.append(dict(file=item['file'], bytes=size, sha256=digest))
        print(json.dumps(dict(verified=len(verified), file=item['file'])), flush=True)
    report = dict(status='passed', files=verified, bytes=sum(item['bytes'] for item in verified),
                  scope='Expert, backbone, and predictor weights against checkpoint manifests; reviewed predictor code hashes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='passed', files=len(verified), bytes=report['bytes'])), flush=True)


if __name__ == '__main__':
    main()
