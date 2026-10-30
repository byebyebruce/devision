"""Verify same-text four-screen invariants and full mixed-image isolation."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
from chart_data import digest,pixels
from audit_exp80 import read


def main():
 root=Path('data');data=root/'v05/exp85';manifest=json.loads((data/'MANIFEST.json').read_text())
 sets={n:read(data/(n+'.jsonl')) for n in ['train','ui_train','ui_dev','calib']}
 for n,p in dict(old_ui_dev='v05/exp82v2/ui_dev.jsonl',chart='v05/exp80v2/chart_dev.jsonl',protect='v05/exp80v2/protect.jsonl',realworld='v05/external/realworld_native.jsonl',examples='v05/external/jev_examples.jsonl').items():sets[n]=read(root/p)
 for n,m in manifest['files'].items():assert digest((data/(n+'.jsonl')).read_bytes())==m['sha256']
 for name in ['ui_train','ui_dev','calib']:
  bygroup=defaultdict(list)
  for r in sets[name]:bygroup[r['counterfactual_group']].append(r)
  for group in bygroup.values():
   assert len(group)==4 and len({r['image_id'] for r in group})==4
   assert len({json.dumps(r['questions'],sort_keys=False) for r in group})==1
   assert all(not r.get('state_text') for r in group)
   gold=[max(r['gold']['q']['probabilities'],key=r['gold']['q']['probabilities'].get) for r in group]
   assert len(set(gold))==4 and set(gold)==set(group[0]['questions']['q']['criteria'])
  assert len({r['image_id'] for r in sets[name]})==len(sets[name])
 original={r['id']:r for r in sets['ui_dev']}
 rotated=read(data/'ui_dev_rotated.jsonl')
 assert len(rotated)==len(original)
 for r in rotated:
  orig=original[r['id']]
  assert r['questions']==orig['questions'] and r['gold']==orig['gold'] and r['image_id']==orig['image_id']
  assert r['image']!=orig['image'] and r['dataset_control']=='within_quad_image_rotation'
  peers=[x for x in sets['ui_dev'] if x['counterfactual_group']==r['counterfactual_group']]
  assert r['image'] in {x['image'] for x in peers}
 assert Counter(r['image'] for r in rotated)==Counter(r['image'] for r in original.values())
 paths=sorted({r['image'] for rows in sets.values() for r in rows})
 def hashed(p):return p,pixels((root/p).read_bytes())[0]
 with ThreadPoolExecutor(max_workers=4) as pool:h=dict(pool.map(hashed,paths))
 ids={n:{h[r['image']] for r in rows} for n,rows in sets.items()}
 overlap={n:len(ids['train']&p) for n,p in ids.items() if n not in ['train','ui_train']}
 # Empirical arbitrary fixed first-option rule must hit exactly one of four for each tuple.
 first=sum(r['gold']['q']['probabilities'][next(iter(r['questions']['q']['criteria']))] for r in sets['ui_dev'])/len(sets['ui_dev'])
 assert first==.25
 out=dict(manifest_sha256=digest((data/'MANIFEST.json').read_bytes()),images_hashed=len(paths),overlaps=overlap,
          identical_text_balanced_quadruples=True,text_only_deterministic_accuracy_bound=first,train_presentations=len(sets['train']),
          ui_train_questions=len(sets['ui_train']),ui_dev_questions=len(sets['ui_dev']))
 out['pass']=not any(overlap.values()) and len(sets['train'])==8192 and len(sets['ui_train'])==256
 Path('runs/v05-prep/exp85_data_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
 if not out['pass']:raise RuntimeError('Counterfactual data audit failed')


if __name__=='__main__':main()
