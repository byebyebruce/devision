"""Compare semantic predictions for the fixed exp89 order diagnostic."""
import json
from pathlib import Path

root = Path('runs/exp89-d3-protect-order')
assert json.loads((root/'STATUS.json').read_text())['state'] == 'completed'
# Launcher exit code is unavailable; completeness is checked against fixed inputs below.
a = {r['sample_id']: r for r in map(json.loads, Path('runs/exp88-d3-panels-local/protect.real.details.jsonl').read_text().splitlines())}
b = list(map(json.loads, (root/'protect.reversed.details.jsonl').read_text().splitlines()))
assert len(b) == 128 and len({r['sample_id'] for r in b}) == 128
result = {}
for source in ['coco-position', 'coco-size']:
    rows = [r for r in b if r['source'] == source]
    assert len(rows) == 64
    original = sum(a[r['sample_id']]['correct'] for r in rows)/len(rows)
    reverse = sum(r['correct'] for r in rows)/len(rows)
    changed = sum(a[r['sample_id']]['prediction'] != r['prediction'] for r in rows)/len(rows)
    result[source] = dict(n=len(rows), original_accuracy=original, reversed_accuracy=reverse,
                          mean_accuracy=(original+reverse)/2, semantic_change_rate=changed)
result['decision'] = 'inspect_order_robustness' if any(v['semantic_change_rate'] > .2 for v in result.values()) else 'inspect_error_images_and_label_semantics'
(root/'report.json').open('x').write(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
