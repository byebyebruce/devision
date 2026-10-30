"""Offline integrity and paired-result recomputation for exp99."""
import json
import math
from pathlib import Path
from devision.train.compare import paired_metric

w=Path('runs/exp106-d3-fp32-scienceqa')
inputs=[json.loads(x) for x in (w/'inputs.jsonl').read_text().splitlines()]
report=json.loads((w/'report.json').read_text())
rows={arm:[json.loads(x) for x in (w/f'{arm}.details.jsonl').read_text().splitlines()] for arm in ('real','noimage','reversed','v04')}
for arm,records in rows.items():
    assert len(records)==256
    assert [r['sample_id'] for r in records]==[s['id'] for s in inputs]
    for s,r in zip(inputs,records):
        assert r['gold']==s['gold']['q']['probabilities']
        p=r['probabilities']
        assert set(p)==set(r['gold'])
        assert all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
        assert r['correct']==(max(p,key=p.get)==max(r['gold'],key=r['gold'].get))
        assert abs(r['nll']+sum(v*math.log(max(p[k],1e-30)) for k,v in r['gold'].items()))<1e-6
    if arm!='v04':
        assert abs(report['summary'][arm]['accuracy']-sum(r['correct'] for r in records)/256)<1e-12
for key,arm in [('visual_gain','noimage'),('v04_comparison','v04')]:
    result=paired_metric(rows[arm],rows['real'],metric='accuracy',resamples=5000,seed=99)
    assert result==report[key]
verification=dict(verified=True,arms={arm:len(r) for arm,r in rows.items()},pictures=len({s['image_id'] for s in inputs}),note='all probabilities, labels, NLL, point metrics and paired bootstrap recomputed; historical baseline reused')
(w/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification))

assert (w/'report.json').is_file()  # launcher exit status unavailable; verify complete artifacts
assert (w/'inputs.jsonl').read_bytes()==Path('runs/exp99-d3-scienceqa/inputs.jsonl').read_bytes()
old=[json.loads(x) for x in Path('runs/exp99-d3-scienceqa/real.details.jsonl').read_text().splitlines()]
assert [r['sample_id'] for r in old]==[r['sample_id'] for r in rows['real']]
comparison=dict(max_probability_difference=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(old,rows['real']) for k in a['probabilities']),semantic_changes=sum(a['prediction']!=b['prediction'] for a,b in zip(old,rows['real'])),paired_accuracy=paired_metric(old,rows['real'],metric='accuracy',resamples=5000,seed=99),regression_persists=report['v04_comparison']['ci95'][1]<0)
with (w/'fp32-comparison.json').open('x') as f: json.dump(comparison,f,indent=2)
print(json.dumps(comparison,indent=2))
