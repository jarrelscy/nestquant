import argparse
import hashlib
import json
import os
from pathlib import Path

from checkpoint import PREDICTOR_CODE
from predictor import configure_slots, slot_budgets, staged_predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    module, staging = staged_predictor()
    canonical = module.layer_budgets({}, 'spark_256K_U_2630_flat50')
    checks = []
    for fraction in (2 / 288, 0.01, 0.10, 2630 / 12096, 0.35, 0.75, 1.0):
        budgets = slot_budgets(canonical, fraction)
        assert sum(budgets.values()) == round(fraction * 12096)
        assert all(2 <= n <= 288 for n in budgets.values())
        fixed = module.fixed_counts_for_fraction(budgets, 0.2)
        assert sum(fixed.values()) == int(sum(budgets.values()) * 0.2 + 0.5)
        assert all(fixed[layer] < budgets[layer] for layer in budgets)
        checks.append(dict(fraction=fraction, slots=sum(budgets.values()), fixed=sum(fixed.values())))
    assert slot_budgets(canonical, 2630 / 12096) == canonical
    for invalid in (-1, 0, 0.001, 1.01, float('nan')):
        try:
            slot_budgets(canonical, invalid)
        except ValueError:
            continue
        raise AssertionError(f'Accepted invalid hot fraction: {invalid}')
    if args.model:
        import numpy as np
        import torch

        for name, expected in PREDICTOR_CODE.items():
            assert hashlib.sha256((args.model / name).read_bytes()).hexdigest() == expected
        torch.set_num_threads(8)
        os.environ['NQ_FLASH_JT_PARAMS'] = json.dumps(dict(mix=0.75, hm=4.0, G=1, half_life=64.0))
        predictor = module.CommittedPredictor(args.model, preset='spark_256K_U_2630_flat50',
                                              device='cpu', dtype=torch.float32, fixed_fraction=0.2)
        ids = np.tile(np.arange(8, dtype=np.int64), (42, 1))
        weights = np.full((42, 8), 1 / 8, dtype=np.float32)
        norms = np.ones(42, dtype=np.float32)
        for check in checks:
            configure_slots(predictor, module, args.model, check['fraction'])
            for position, token in enumerate((42, 43, 44)):
                wanted = predictor.preforward(token, position)
                assert wanted.shape == (42, 288)
                assert int(wanted.sum()) == check['slots']
                for i, layer in enumerate(predictor.layers):
                    assert wanted[i].sum() == predictor.budgets[layer]
                    assert wanted[i, predictor.fixed_ids[i]].all()
                predictor.commit([(position, token, ids, weights, norms)])
            check['committed_tokens'] = predictor.t
    report = dict(status='passed', allocations=checks, trained_predictor_checked=args.model is not None,
                  scope='Slot allocation and optional trained jT execution with synthetic routing history')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    staging.cleanup()


if __name__ == '__main__':
    main()
