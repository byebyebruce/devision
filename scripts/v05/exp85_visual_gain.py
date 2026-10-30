"""Supplemental difference-in-differences; leaves original exp85 gates unchanged."""
import json
from pathlib import Path

import numpy as np


def main():
    data = Path('data/v05/exp85/ui_dev.jsonl')
    groups = {r['id']: r['counterfactual_group'] for r in map(json.loads, data.read_text().splitlines())}
    keys = [('baseline', 'real'), ('baseline', 'mismatched'), ('raw', 'real'), ('raw', 'mismatched')]
    rows = []
    for stage, control in keys:
        path = Path('runs/exp85') / stage / f'ui.{control}.details.jsonl'
        records = list(map(json.loads, path.read_text().splitlines()))
        indexed = {r['sample_id']: r for r in records}
        assert len(indexed) == len(records) == 256
        assert set(indexed) == set(groups)
        rows.append(indexed)
    clustered = {}
    for sid in sorted(groups):
        clustered.setdefault(groups[sid], []).append([float(rs[sid]['correct']) for rs in rows])
    matrix = np.array(list(clustered.values()))
    assert matrix.shape == (64, 4, 4)
    means = matrix.mean(1)
    rng = np.random.default_rng(86)
    boot = np.array([means[rng.integers(0, 64, 64)].mean(0) for _ in range(5000)])
    values = means.mean(0)
    differences = (boot[:, 2] - boot[:, 3]) - (boot[:, 0] - boot[:, 1])
    result = dict(protocol='supplemental; not a replacement of exp85 gates',
                  baseline_real=values[0], baseline_rotated=values[1],
                  final_real=values[2], final_rotated=values[3],
                  visual_gap_change=(values[2]-values[3])-(values[0]-values[1]),
                  ci95=np.quantile(differences, [.025, .975]).tolist(), groups=64,
                  seed=86, resamples=5000)
    Path('runs/v05-prep/exp85_visual_gain_supplement.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
