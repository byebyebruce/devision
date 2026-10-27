"""Independent probability/identity checks and document-cluster comparison."""
import json
import math
from pathlib import Path
from devision.train.compare import load_records,paired_metric
root=Path('runs/exp98-d3-funsd-binding')
samples=[json.loads(x) for x in Path('data/v05/funsd/derived/dev.jsonl').read_text().splitlines()]
keys={s['id'] for s in samples}
checks={}
records={}
for name,file in [('real',root/'real.details.jsonl'),('noimage',root/'noimage.details.jsonl'),('reversed',root/'reversed.details.jsonl'),('v04',Path('runs/exp98-v04-funsd/real.details.jsonl'))]:
 rows=load_records(str(file));assert len(rows)==len(samples) and {r['sample_id'] for r in rows}==keys
 for r in rows:
  p=r['probabilities'];g=r['gold'];assert set(p)==set(g)
  assert all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
  correct=max(p,key=p.get)==max(g,key=g.get)
  assert correct==r['correct']
  assert abs(r['nll']+sum(v*math.log(max(p[k],1e-30)) for k,v in g.items()))<1e-5
 checks[name]=dict(n=len(rows),correct=sum(r['correct'] for r in rows),nll=sum(r['nll'] for r in rows)/len(rows))
 records[name]=rows
report=dict(verification=checks,d3_vs_v04=paired_metric(records['v04'],records['real'],metric='accuracy',resamples=5000,seed=98),visual_gain=paired_metric(records['noimage'],records['real'],metric='accuracy',resamples=5000,seed=98))
(root/'verification.json').open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
