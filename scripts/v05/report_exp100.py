"""Offline integrity and paired-result recomputation for exp100."""
import json
import math
from pathlib import Path
import hashlib
from PIL import Image
from devision.train.compare import paired_metric

w=Path('runs/exp100-d3-countbench-retry1')
inputs=[json.loads(x) for x in (w/'inputs.jsonl').read_text().splitlines()]
report=json.loads((w/'report.json').read_text())
rows={arm:[json.loads(x) for x in (w/f'{arm}.details.jsonl').read_text().splitlines()] for arm in ('real','noimage','reversed','v04')}
for arm,records in rows.items():
    assert len(records)==491
    assert [r['sample_id'] for r in records]==[s['id'] for s in inputs]
    for s,r in zip(inputs,records):
        assert r['gold']==s['gold']['q']['probabilities']
        p=r['probabilities']
        assert set(p)==set(r['gold'])
        assert all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
        assert r['correct']==(max(p,key=p.get)==max(r['gold'],key=r['gold'].get))
        assert abs(r['nll']+sum(v*math.log(max(p[k],1e-30)) for k,v in r['gold'].items()))<1e-6
    if arm!='v04':
        assert abs(report['summary'][arm]['accuracy']-sum(r['correct'] for r in records)/491)<1e-12
for key,arm in [('visual_gain','noimage'),('v04_comparison','v04')]:
    result=paired_metric(rows[arm],rows['real'],metric='accuracy',resamples=5000,seed=100)
    assert result==report[key]
for s,r in zip(inputs,rows['v04']):
    with Image.open(Path('data')/s['image']) as im:
        digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
    assert digest==r['picture_pixels_sha256'], s['id']
verification=dict(historical_baseline_pixel_identity_verified=True,verified=True,arms={arm:len(r) for arm,r in rows.items()},pictures=len({s['image_id'] for s in inputs}),note='all probabilities, labels, NLL, point metrics and paired bootstrap recomputed; historical baseline reused')
(w/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification))
