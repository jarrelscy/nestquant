import hashlib
import importlib.util
import json
from pathlib import Path


PREDICTOR_CODE = {
    'serving/predictor/jt/jt_model.py': 'c6ed8a2c241f50c32fc55051f0ad797493abe836e885ba794e49e545405f01a2',
    'serving/predictor/jt/policy.py': '109327fbd2d8fb173083f575bd241f8eadb6c2adce04f0dd9a79c444e4ad979d',
}


def validate(root: Path):
    source = Path(__file__).resolve().parents[1] / 'spark/flash/nq_flash_layout.py'
    spec = importlib.util.spec_from_file_location('nq_flash_layout_cpu', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_checkpoint(root)
    for layer in range(3, 45):
        directory = root / 'layers' / f'L{layer}'
        manifest = json.loads((directory / 'manifest.json').read_text())
        if int(manifest.get('config', {}).get('in_had_down', 128)) != 128:
            raise ValueError('The Mac kernels require 128-wide Hadamard blocks')
        for shard in range(8):
            name = f'tp{shard}.safetensors'
            if (directory / name).stat().st_size != manifest['files'][name]['bytes']:
                raise ValueError(f'Incomplete checkpoint shard: L{layer}/{name}')
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    for name, file in index.items():
        if name.startswith('model.language_model.') or name == 'lm_head.weight':
            path = (root / file).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f'Missing or invalid backbone file: {file}')
    for name, expected in PREDICTOR_CODE.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Predictor source differs from the reviewed checkpoint revision: {name}')
