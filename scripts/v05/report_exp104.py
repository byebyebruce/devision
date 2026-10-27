"""Offline verification and image-group paired comparison for exp104."""
import json,math,random
from pathlib import Path
W=Path('runs/exp104-d3-fp32-realworld-retry1')
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
assert (W/'EXIT_CODE').read_text().strip()=='0'
rows=read(W/'details.jsonl');assert len(rows)==587
refs={'bf16':read('runs/exp88-d3-panels-local/realworld.real.details.jsonl'),'v04':read('runs/v05-prep/v04_realworld_raw.details.jsonl')}
for r in rows:
 p=r['probabilities'];g=r['gold'];assert set(p)==set(g)
 assert all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
 assert r['correct']==(max(p,key=p.get)==max(g,key=g.get))
 assert abs(r['nll']+sum(v*math.log(max(p[k],1e-30)) for k,v in g.items()))<1e-8
report={'n':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),'nll':sum(r['nll'] for r in rows)/len(rows),'comparisons':{}}
for name,ref in refs.items():
 by={r['sample_id']:r for r in ref};assert set(by)=={r['sample_id'] for r in rows}
 groups={};changes=0;maxdiff=0;refnll=0;refacc=0
 for r in rows:
  b=by[r['sample_id']];assert r['gold']==b['gold'] and r['image_id']==b['image_id']
  p=b['probabilities'];ok=max(p,key=p.get)==max(b['gold'],key=b['gold'].get)
  refacc+=ok;refnll-=sum(v*math.log(max(p[k],1e-30)) for k,v in b['gold'].items())
  groups.setdefault(r['image_id'],[]).append(int(r['correct'])-int(ok))
  changes+=r['prediction']!=max(p,key=p.get);maxdiff=max(maxdiff,max(abs(r['probabilities'][k]-p[k]) for k in p))
 rng=random.Random(104);keys=list(groups);draws=[]
 for _ in range(5000):
  chosen=rng.choices(keys,k=len(keys));draws.append(sum(sum(groups[k]) for k in chosen)/sum(len(groups[k]) for k in chosen))
 draws.sort();report['comparisons'][name]=dict(accuracy=refacc/len(rows),nll=refnll/len(rows),accuracy_delta=report['accuracy']-refacc/len(rows),nll_delta=report['nll']-refnll/len(rows),image_bootstrap95=[draws[125],draws[4874]],semantic_changes=changes,max_probability_delta=maxdiff)
b=report['comparisons']['bf16'];report['deployment_screen_pass']=b['accuracy_delta']>=-.01 and b['nll_delta']<=.02
report['scope']='monitoring native RealWorldQA choice/noul subset, historical comparisons; not calibration, statistical equivalence, or official admission'
(W/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
