"""Summarize the prewritten paired calibration prompt diagnostic; never overwrite."""
import json
from pathlib import Path
import numpy as np

source = [json.loads(x) for x in Path('data/v05/exp85/calib.jsonl').read_text().splitlines()]
groups = {}
for row in source:
    groups.setdefault(row['counterfactual_group'], []).append(row['id'])
assert len(groups) == 8 and all(len(v) == 4 for v in groups.values())
reports = {}
records = {}
for label in ('empty', 'answer'):
    folder = Path('runs/exp86-prompt-calib-' + label)
    assert json.loads((folder / 'STATUS.json').read_text())['state'] == 'completed'
    controls = {}
    for control in ('real', 'reversed', 'mismatched'):
        rows = [json.loads(x) for x in (folder / f'ui.{control}.details.jsonl').read_text().splitlines()]
        keyed = {x['sample_id']: x for x in rows}
        assert len(keyed) == len(rows) == len(source)
        assert set(keyed) == {x['id'] for x in source}
        assert all(keyed[x['id']]['gold'] == x['gold']['q']['probabilities'] for x in source)
        controls[control] = keyed
    acc = {c: float(np.mean([x['correct'] for x in rows.values()])) for c, rows in controls.items()}
    reports[label] = dict(accuracy=acc, order_mean=(acc['real']+acc['reversed'])/2,
        visual_gap=acc['real']-acc['mismatched'],
        semantic_change_rate=float(np.mean([controls['real'][k]['prediction'] != controls['reversed'][k]['prediction'] for k in controls['real']])))
    records[label] = controls

diffs = np.array([np.mean([sum(float(records['answer'][c][k]['correct'])-float(records['empty'][c][k]['correct']) for c in ('real','reversed'))/2 for k in ids]) for ids in groups.values()])
rng = np.random.default_rng(86)
ci = np.quantile(diffs[rng.integers(0,len(diffs),size=(5000,len(diffs)))].mean(1), [.025,.975]).tolist()
a,b=reports['answer'],reports['empty']
passed=a['order_mean']-b['order_mean'] >= .05 and a['visual_gap']>0 and a['semantic_change_rate']<b['semantic_change_rate']
report=dict(protocols=reports, order_mean_difference=float(diffs.mean()), ci95=ci,
    groups=8, questions=32, bootstrap_seed=86, resamples=5000,
    decision='expand_fixed_prefix_on_development' if passed else 'stop_prompt_search_move_to_next_backbone',
    scope='prompt selection diagnostic; calibration data now monitoring; no release qualification')
path=Path('runs/exp86-prompt-calib-report.json')
with path.open('x') as f:
    f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
