"""Tuple-clustered counterfactual UI decision; never alter original prediction files."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from devision.train.compare import load_records,paired_metric
from devision.train.evaluate import order_sensitivity
from structure_report import occurrences


def main():
 data=Path('data/v05/exp85');work=Path('runs/exp85');base=work/'baseline';end=work/'raw';init=Path('runs/exp81/raw')
 groups={r['id']:r['counterfactual_group'] for line in (data/'ui_dev.jsonl').open() if (r:=json.loads(line))}
 def records(folder,name,control='real'):
  return occurrences(load_records(str(folder/f'{name}.{control}.details.jsonl')))
 def cluster(rs):
  out=[]
  for r in rs:
   out.append(dict(r,actual_image_id=r['image_id'],image_id=groups[r['sample_id'].rsplit('#occ',1)[0]]))
  return out
 def pair(a,b,metric='accuracy',quad=False):
  d=paired_metric(cluster(a) if quad else a,cluster(b) if quad else b,metric,seed=85)
  if quad:d['quadruples']=d.pop('pictures');d['resampling_unit']='four-screen tuple'
  return d
 def harm(d):return d['difference']<=-.03 and d['ci95'][1]<0
 ui=records(end,'ui');p1=pair(records(base,'ui'),ui,quad=True)
 p2=pair(records(end,'ui','mismatched'),ui,quad=True)
 bygroup=defaultdict(list)
 for r in ui:bygroup[groups[r['sample_id'].rsplit('#occ',1)[0]]].append(float(r['correct']))
 matrix=np.array(list(bygroup.values()));assert matrix.shape[1]==4
 rng=np.random.default_rng(85);boot=np.array([matrix[rng.integers(0,len(matrix),len(matrix))].mean() for _ in range(2000)])
 lo,hi=np.quantile(boot,[.025,.975]);accuracy=dict(value=float(matrix.mean()),ci95=[float(lo),float(hi)],quadruples=len(matrix),chance_bound=.25)
 chart=pair(records(init,'chart'),records(end,'chart'));old,new=records(init,'protect'),records(end,'protect')
 sources={s:pair([r for r in old if r['source']==s],[r for r in new if r['source']==s]) for s in sorted({r['source'] for r in old})}
 bad=[s for s,d in sources.items() if harm(d)];nll=pair(old,new,'nll');photo=pair(records(init,'realworld'),records(end,'realworld'))
 gates=dict(P1=p1['difference']>=.05 and p1['ci95'][0]>0,
            P2=lo>.25 and p2['difference']>=.03 and p2['ci95'][0]>0,
            P3=not harm(chart) and not bad and nll['difference']<=.05 and not harm(photo))
 order=order_sensitivity(ui,records(end,'ui','reversed'));assert order['n']==len(ui)
 result=dict(P1_ui=p1,P2_real_minus_within_tuple_rotation=p2,accuracy_vs_text_only_bound=accuracy,
             all_four_correct_fraction=float((matrix.sum(1)==4).mean()),at_least_three_correct_fraction=float((matrix.sum(1)>=3).mean()),
             chart_protection=chart,old_protect_nll=nll,protect_sources=sources,harmful_sources=bad,realworld=photo,
             order_sensitivity=order,pass_conditions=gates,decision='retain_candidate' if all(gates.values()) else 'reject',
             bootstrap_seed=85,control_provenance='Mismatched JSONL rotates images within each tuple; CLI none adds no shuffle; actual encoded image identities remain in raw records.')
 (work/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='protect_sources'},indent=2))


if __name__=='__main__':main()
