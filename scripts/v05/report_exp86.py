"""Fixed UI gates with paired bootstrap over original four-screen groups."""
import json
from pathlib import Path
import sys
import numpy as np


def main():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/exp86-smol-ui-baseline-seq64')
    assert json.loads((work / 'STATUS.json').read_text())['state'] == 'completed'
    def rows(path):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        result = {r['sample_id']: r for r in records}
        assert len(result) == len(records) == 256
        return result
    base = rows(Path('runs/exp85/baseline/ui.real.details.jsonl'))
    real = rows(work / 'ui.real.details.jsonl')
    wrong = rows(work / 'ui.mismatched.details.jsonl')
    source = [json.loads(line) for line in Path('data/v05/exp85/ui_dev.jsonl').read_text().splitlines()]
    groups = {}
    assert set(base) == set(real) == set(wrong) == {r['id'] for r in source}
    for s in source:
        key = s['id']
        assert base[key]['gold'] == real[key]['gold'] == wrong[key]['gold']
        groups.setdefault(s['counterfactual_group'], []).append(key)
    assert len(groups) == 64 and all(len(ids) == 4 for ids in groups.values())
    def compare(a, b):
        diffs = np.array([np.mean([float(b[k]['correct']) - float(a[k]['correct']) for k in ids]) for ids in groups.values()])
        rng = np.random.default_rng(86)
        boot = diffs[rng.integers(0, 64, size=(5000, 64))].mean(axis=1)
        return dict(difference=float(diffs.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), quadruples=64, questions=256)
    p1, p2 = compare(base, real), compare(wrong, real)
    passed = p1['difference'] >= .05 and p1['ci95'][0] > 0 and p2['difference'] >= .03 and p2['ci95'][0] > 0
    report = dict(P1_vs_exp81=p1, P2_real_minus_rotation=p2,
                  accuracy={name:sum(r['correct'] for r in rs.values())/256 for name,rs in [('exp81',base),('smol_real',real),('smol_rotation',wrong)]},
                  decision='advance_to_other_panels' if passed else 'validate_readout_before_training',
                  bootstrap_seed=86, resamples=5000, scope='development UI only; no v0.5 release qualification')
    (work / 'report.json').open('x').write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
