"""Independent semantic replication; fixed per-source criterion."""
import json
from pathlib import Path
from devision.train.compare import load_records, paired_metric
root=Path('runs/exp91-d3-semantics-independent')
a=load_records(str(root/'protect.real.details.jsonl'));b=load_records(str(root/'protect.semantic.details.jsonl'))
assert len(a)==len(b)==128
result={}
for source in ('coco-size','coco-position'):
 x=[r for r in a if r['source']==source];y=[r for r in b if r['source']==source]
 assert len(x)==len(y)==64
 assert {r['sample_id'] for r in x}=={r['sample_id'] for r in y}
 d=paired_metric(x,y,metric='accuracy',resamples=5000,seed=91)
 old={r['sample_id']:r for r in x}
 d['semantic_change_rate']=sum(old[r['sample_id']]['prediction']!=r['prediction'] for r in y)/64
 d['nll_original']=sum(r['nll'] for r in x)/64;d['nll_semantic']=sum(r['nll'] for r in y)/64
 result[source]=d
d=result['coco-size'];result['size_replication_pass']=d['difference']>=.05 and d['ci95'][0]>0
(root/'report.json').open('x').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
