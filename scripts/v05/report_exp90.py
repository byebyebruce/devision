"""Paired, picture-clustered semantic diagnostic on frozen 128 samples."""
import json
from pathlib import Path
from devision.train.compare import load_records, paired_metric
from structure_report import occurrences
root=Path('runs/exp90-d3-semantics')
# Background launcher did not persist EXIT_CODE; do not infer exit0.
if (root/'EXIT_CODE').exists():
 assert (root/'EXIT_CODE').read_text().strip()=='0'
assert json.loads((root/'STATUS.json').read_text())['state']=='completed'
a=occurrences(load_records('runs/exp88-d3-panels-local/protect.real.details.jsonl'))
b=occurrences(load_records(str(root/'protect.semantic.details.jsonl')))
assert len(b)==128 and len({r['sample_id'] for r in b})==128
result={}
for source in ('coco-position','coco-size'):
 x=[r for r in a if r['source']==source];y=[r for r in b if r['source']==source]
 assert len(x)==len(y)==64
 assert {r['sample_id'] for r in x}=={r['sample_id'] for r in y}
 d=paired_metric(x,y,metric='accuracy',resamples=5000,seed=90)
 old={r['sample_id']:r for r in x}
 d['semantic_change_rate']=sum(old[r['sample_id']]['prediction']!=r['prediction'] for r in y)/len(y)
 result[source]=d
result['decision']='validate_semantics_independently' if any(d['difference']>=.05 and d['ci95'][0]>0 for d in result.values()) else 'stop_prompt_search_prepare_recovery_pilot'
(root/'report.json').open('x').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
