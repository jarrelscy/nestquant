import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


def staged_predictor():
    source = Path(__file__).resolve().parents[1] / 'spark/flash'
    staging = tempfile.TemporaryDirectory(prefix='nestquant-macos-predictor-')
    destination = Path(staging.name)
    shutil.copyfile(source / 'sampled_ahead_sentinel/jt_runtime.py', destination / 'jt_runtime.py')
    for path in source.glob('u_distribution*.json'):
        shutil.copyfile(path, destination / path.name)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('nestquant_macos_jt_runtime', destination / 'jt_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, staging


def slot_budgets(canonical: dict[int, int], fraction: float) -> dict[int, int]:
    if set(canonical) != set(range(3, 45)) or sum(canonical.values()) != 2630:
        raise ValueError('Expected the canonical 42-layer, 2630-slot allocation')
    if not 2 / 288 <= fraction <= 1:
        raise ValueError('Hot fraction must be between 2/288 and 1')
    total = round(fraction * 12096)
    if total == 2630:
        return canonical.copy()
    budgets = {layer: 2 for layer in canonical}
    for _ in range(total - sum(budgets.values())):
        layer = max((layer for layer in budgets if budgets[layer] < 288),
                    key=lambda layer: (canonical[layer] / (budgets[layer] + 1), -layer))
        budgets[layer] += 1
    return budgets


def configure_slots(predictor, module, root: Path, fraction: float):
    canonical = module.layer_budgets({}, 'spark_256K_U_2630_flat50')
    predictor.budgets = slot_budgets(canonical, fraction)
    predictor.fixed_counts = module.fixed_counts_for_fraction(predictor.budgets, 0.2)
    predictor.n_fixed = sum(predictor.fixed_counts.values())
    manifest = json.loads((root / 'fixed_set.json').read_text())
    predictor.defaults, predictor.fixed_ids = module.initial_expert_pools(
        manifest, predictor.layers, predictor.budgets, predictor.fixed_counts)
    predictor.reset()
